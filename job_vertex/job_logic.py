import builtins
import datetime
import inspect
import io
import json
import os
import shutil
import smtplib
import time
import traceback
import warnings
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.cloud import storage as _gcs


# Always flush prints so Cloud Run logs appear immediately
print = lambda *a, **kw: builtins.print(*a, **{**kw, 'flush': True})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as _permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

import google.auth
from googleapiclient.discovery import build


PACIFIC = ZoneInfo('America/Los_Angeles')

def _pt():
    return datetime.datetime.now(PACIFIC).strftime('%H:%M:%S PT')

class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cols_to_keep_ = X.columns[X.notna().any(axis=0)].tolist()
        return self

    def transform(self, X):
        return X[self.cols_to_keep_]

warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

RANDOM_SEED = 12345
CV_FOLDS = 5
N_BOOT = 20

LASSO_PARAM_GRID = {"model__alpha": list(np.logspace(-4, 1, 7))}
RIDGE_PARAM_GRID = {"model__alpha": list(np.logspace(-4, 1, 7))}

RF_PARAM_GRID = {
    "model__n_estimators": [200, 500, 1000, 2000, 4000],
    "model__max_depth": [3, 5, 10, 20, 40],
    "model__min_samples_split": [ 5, 10, 20,40],
    "model__min_samples_leaf": [1, 2, 5, 10],
    "model__max_features": ["sqrt", "log2", 0.5],
}
RF_N_ITER = 25  # RandomizedSearchCV samples

LGBM_N_TRIALS = 30  # Optuna trials per outer fold, matching the notebook

LGBM_SEARCH_BOUNDS = {
    'learning_rate':     (0.01, 0.5),
    'max_depth':         (1, 10),
    'num_leaves':        (5, 60),
    'min_child_samples': (5, 300),
}

TORCH_GRID = list(ParameterGrid({
    "lr": [1e-3, 3e-4],
    "weight_decay": [1e-3, 1e-5],
    "patience": [10, 50, 200],
}))
TORCH_LIGHT_PARAMS = {"lr": 1e-3, "weight_decay": 1e-4, "patience": 10}

# Single-point grids used when toy_param_grids=True — one value near the middle of each range.
TOY_LASSO_PARAM_GRID = {"model__alpha": [0.0316]}  # middle of logspace(-4, 1, 7)
TOY_RF_PARAM_GRID = {
    "model__n_estimators": [1000],
    "model__max_depth": [10],
    "model__min_samples_split": [10],
    "model__min_samples_leaf": [5],
    "model__max_features": ["log2"],
}
TOY_LGBM_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'num_leaves': 30,
    'min_child_samples': 40,
}
TOY_TORCH_PARAMS = [{"lr": 5e-4, "weight_decay": 1e-4, "patience": 50}]

LOAD_EXISTING_FEATURES = True
NN_LIGHT_MODE = False  # True: skip inner CV for NN (~6x faster); False: full nested CV

SHEET_ID = '13vZkBNoI1TNKEuWJhtFospr_XNR23DpZTobjaKAJneA'
FINAL_EVAL_SHEET_ID = '1a_evWLy8NxFcaD89knoc4F3mCHqfeX0AiSBmOrIEqJ0'
SHEET_TAB = 'Results log'
LEADERBOARD_STANDALONE_TAB = 'Leaderboard: Stand-alone'
LEADERBOARD_MERGED_TAB = 'Leaderboard: Along with existing features'
INDIVIDUAL_FEATURES_TAB = 'Log: Individual Features'
LEADERBOARD_SIZE = 10

GMAIL_FROM = 'featurizationtestserver@gmail.com'

# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------

class _TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class _TorchRegressorWrapper:
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3, weight_decay=1e-4, epochs=1000, patience=20, device='cpu'):
        self.device = device
        self.model = _TorchMLP(input_dim, hidden_dim).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y, val_frac=0.15, batch_size=64, sample_weight=None):
        # Initialise the output bias to the training mean so the network starts
        # predicting at the correct scale.  Without this the bias is near 0
        # while log_consumption has mean ~6.5.  Adam's per-parameter gradient
        # normalisation means it closes this gap at a fixed rate (~lr per batch
        # update, regardless of gradient magnitude).
        with torch.no_grad():
            self.model.net[-1].bias.fill_(float(y.mean()))

        n = len(X)
        n_val = max(1, int(n * val_frac))
        perm = np.random.default_rng(RANDOM_SEED).permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1).to(self.device)

        if sample_weight is not None:
            sw_train = sample_weight[train_idx]
            sw_train = sw_train / sw_train.mean()
            sw_train_t = torch.tensor(sw_train, dtype=torch.float32).unsqueeze(1)
            sw_val = sample_weight[val_idx]
            sw_val = sw_val / sw_val.mean()
            sw_val_t = torch.tensor(sw_val, dtype=torch.float32).unsqueeze(1).to(self.device)
            dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t, sw_train_t)
        else:
            sw_val_t = None
            dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        best_loss, patience_left, best_state = float('inf'), self.patience, None

        for epoch in range(self.epochs):
            self.model.train()
            for batch in loader:
                if sample_weight is not None:
                    X_b, y_b, sw_b = batch
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)
                    sw_b = sw_b.to(self.device)
                    self.opt.zero_grad()
                    pred = self.model(X_b)
                    loss = (sw_b * (pred - y_b) ** 2).mean()
                else:
                    X_b, y_b = batch
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    self.opt.zero_grad()
                    loss = self.loss_fn(self.model(X_b), y_b)
                loss.backward()
                self.opt.step()

            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(X_val_t)
                if sw_val_t is not None:
                    val_loss = (sw_val_t * (pred_val - y_val_t) ** 2).mean().item()
                else:
                    val_loss = self.loss_fn(pred_val, y_val_t).item()
            if val_loss < best_loss:
                best_loss = val_loss
                patience_left = self.patience
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            ).cpu().numpy().ravel()


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def _bootstrap_metrics(y_true, y_pred):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    results = {}
    for name, fn in [
        ('r2', r2_score),
        ('pearson', lambda a, b: pearsonr(a, b)[0]),
        ('spearman', lambda a, b: spearmanr(a, b)[0]),
    ]:
        vals = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            v = fn(y_true[idx], y_pred[idx])
            if not np.isnan(v):
                vals.append(v)
        lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
        results[name] = {'mean': float(np.nanmean(vals)), 'ci_low': float(lo), 'ci_high': float(hi)}
    return results


def _cv_sklearn(X, y, pipeline, param_grid, outer_kf, inner_kf, model_type_name='unspecified', sample_weight=None, n_iter=None):
    """Nested CV for sklearn pipelines; returns (y_true, y_pred, params_per_fold).

    If n_iter is given, uses RandomizedSearchCV instead of GridSearchCV.
    """
    fold_preds = []
    params_per_fold = []
    for i, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
        print(f'{_pt()} running grid search fold {i} for model type {model_type_name}')
        if n_iter is not None:
            gs = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, cv=inner_kf,
                                    n_jobs=-1, verbose=0, random_state=RANDOM_SEED)
        else:
            gs = GridSearchCV(pipeline, param_grid, cv=inner_kf, n_jobs=-1, verbose=0)
        fit_params = {}
        if sample_weight is not None:
            fit_params['model__sample_weight'] = sample_weight[train_idx]
        gs.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)
        fold_preds.append((y.iloc[test_idx].values, gs.best_estimator_.predict(X.iloc[test_idx])))
        params_per_fold.append(gs.best_params_)
    y_true = np.concatenate([t for t, _ in fold_preds])
    y_pred = np.concatenate([p for _, p in fold_preds])
    return y_true, y_pred, params_per_fold


def _torch_inner_cv(X, y, param_grid, inner_kf, sample_weight=None):
    """Inner CV for torch hyperparameter selection."""
    best_params, best_score = None, -np.inf
    for params in param_grid:
        scores = []
        for tr, va in inner_kf.split(X):
            m = _TorchRegressorWrapper(X.shape[1], **params)
            sw_tr = sample_weight[tr] if sample_weight is not None else None
            m.fit(X[tr], y[tr], sample_weight=sw_tr)
            scores.append(r2_score(y[va], m.predict(X[va])))
        if np.mean(scores) > best_score:
            best_score, best_params = np.mean(scores), params
    return best_params


def _cv_torch(X, y, param_grid, outer_kf, inner_kf, light=False, sample_weight=None):
    """Nested CV for the torch MLP; scales inside each outer fold.

    If light=True, skips inner CV and uses TORCH_LIGHT_PARAMS directly (~18x faster).
    """
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y if isinstance(y, np.ndarray) else np.asarray(y)
    fold_preds = []
    for i, (train_idx, test_idx) in enumerate(outer_kf.split(X_arr)):
        print(f'{_pt()} running NN fold {i}')
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        # Replace NaN/inf that may remain after scaling
        X_train_s = np.nan_to_num(X_train_s)
        X_test_s = np.nan_to_num(X_test_s)
        if light:
            best_params = TORCH_LIGHT_PARAMS
        else:
            best_params = _torch_inner_cv(X_train_s, y_train, param_grid, inner_kf, sample_weight=sw_train)
        model = _TorchRegressorWrapper(X_train_s.shape[1], **best_params)
        model.fit(X_train_s, y_train, sample_weight=sw_train)
        fold_preds.append((y_test, model.predict(X_test_s)))
    y_true = np.concatenate([t for t, _ in fold_preds])
    y_pred = np.concatenate([p for _, p in fold_preds])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# LightGBM with Optuna tuning
# ---------------------------------------------------------------------------

def _lgbm_make_objective(X_outer_train, y_outer_train, inner_kf, sample_weight=None):
    """Return an Optuna objective that does inner-CV R² for LightGBM."""
    def objective(trial):
        params = {
            'n_estimators':      1000,
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'max_depth':         trial.suggest_int('max_depth', 1, 10),
            'num_leaves':        trial.suggest_int('num_leaves', 5, 60),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 500, log=True),
            'random_state':      RANDOM_SEED,
            'verbose':           -1,
        }
        inner_scores = []
        for tr, va in inner_kf.split(X_outer_train):
            model = lgb.LGBMRegressor(**params)
            sw_tr = sample_weight[tr] if sample_weight is not None else None
            model.fit(
                X_outer_train[tr], y_outer_train[tr],
                sample_weight=sw_tr,
                eval_set=[(X_outer_train[va], y_outer_train[va])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            inner_scores.append(r2_score(y_outer_train[va], model.predict(X_outer_train[va])))
        return float(np.mean(inner_scores))
    return objective


def _cv_lgbm(X, y, outer_kf, inner_kf, n_trials=LGBM_N_TRIALS, sample_weight=None, fixed_params=None):
    """Nested CV for LightGBM with Optuna inner tuning; returns (y_true, y_pred, params_per_fold).

    If fixed_params is provided, skips Optuna and uses those params in every fold.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    y_arr = y if isinstance(y, np.ndarray) else np.asarray(y)
    fold_preds = []
    params_per_fold = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_arr)):
        print(f'{_pt()} lgbm outer fold {outer_fold}')
        X_outer_train, X_outer_test = X_arr[train_idx], X_arr[test_idx]
        y_outer_train, y_outer_test = y_arr[train_idx], y_arr[test_idx]
        sw_outer_train = sample_weight[train_idx] if sample_weight is not None else None
        # Drop columns that are entirely NaN in the training split
        keep = np.where(~np.all(np.isnan(X_outer_train), axis=0))[0]
        X_outer_train, X_outer_test = X_outer_train[:, keep], X_outer_test[:, keep]
        if fixed_params is not None:
            best = fixed_params
        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            )
            study.optimize(_lgbm_make_objective(X_outer_train, y_outer_train, inner_kf, sw_outer_train), n_trials=n_trials)
            best = study.best_params
        params_per_fold.append(best)
        final_model = lgb.LGBMRegressor(n_estimators=1000, random_state=RANDOM_SEED, verbose=-1, **best)
        final_model.fit(X_outer_train, y_outer_train, sample_weight=sw_outer_train)
        fold_preds.append((y_outer_test, final_model.predict(X_outer_test)))
    y_true = np.concatenate([t for t, _ in fold_preds])
    y_pred = np.concatenate([p for _, p in fold_preds])
    return y_true, y_pred, params_per_fold


def _lgbm_holdout(X_train, y_train, X_holdout, y_holdout, inner_kf, n_trials=LGBM_N_TRIALS, sample_weight=None, fixed_params=None):
    """Optuna-tuned LightGBM trained on non-holdout, evaluated on holdout. Returns (y_ho, preds, best_params).

    If fixed_params is provided, skips Optuna and uses those params directly.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_tr = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
    y_tr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    X_ho = X_holdout.values if hasattr(X_holdout, 'values') else np.asarray(X_holdout)
    y_ho = y_holdout.values if hasattr(y_holdout, 'values') else np.asarray(y_holdout)
    keep = np.where(~np.all(np.isnan(X_tr), axis=0))[0]
    X_tr, X_ho = X_tr[:, keep], X_ho[:, keep]
    if fixed_params is not None:
        best = fixed_params
    else:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(_lgbm_make_objective(X_tr, y_tr, inner_kf, sample_weight), n_trials=n_trials)
        best = study.best_params
    final_model = lgb.LGBMRegressor(n_estimators=1000, random_state=RANDOM_SEED, verbose=-1, **best)
    final_model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return y_ho, final_model.predict(X_ho), best


# ---------------------------------------------------------------------------
# Hyperparameter logging
# ---------------------------------------------------------------------------

def _rf_boundary_warnings(params):
    """Return list of boundary-warning strings for a single RF best_params_ dict."""
    msgs = []
    for key, values in RF_PARAM_GRID.items():
        v = params.get(key)
        numeric = [x for x in values if isinstance(x, (int, float))]
        # Need at least two numeric values to have a meaningful boundary
        if len(numeric) < 2 or v is None or not isinstance(v, (int, float)):
            continue
        if v == min(numeric):
            msgs.append(f"{key.replace('model__', '')}={v} (at low boundary)")
        elif v == max(numeric):
            msgs.append(f"{key.replace('model__', '')}={v} (at high boundary)")
    return msgs


def _lgbm_boundary_warnings(params):
    """Return list of boundary-warning strings for a single Optuna best_params dict."""
    msgs = []
    for k, (lo, hi) in LGBM_SEARCH_BOUNDS.items():
        v = params.get(k)
        if v is None:
            continue
        at_lo = (v == lo) if isinstance(v, int) else (v <= lo * 1.01)
        at_hi = (v == hi) if isinstance(v, int) else (v >= hi * 0.99)
        if at_lo:
            msgs.append(f"{k}={v:.4g} (near low boundary of {lo})")
        elif at_hi:
            msgs.append(f"{k}={v:.4g} (near high boundary of {hi})")
    return msgs


def _format_hp_log(label, rf_params_per_fold, lgbm_params_per_fold):
    """Return a human-readable string of HP details for one evaluation context, or '' if none."""
    if not rf_params_per_fold and not lgbm_params_per_fold:
        return ''
    lines = [f'[{label}]']

    if rf_params_per_fold:
        lines.append('  Random Forest:')
        fold_warnings = []
        for i, params in enumerate(rf_params_per_fold):
            nice = {k.replace('model__', ''): v for k, v in params.items()}
            lines.append('    Fold {}: {}'.format(i + 1, ', '.join(f'{k}={v}' for k, v in nice.items())))
            w = _rf_boundary_warnings(params)
            if w:
                fold_warnings.append('    Fold {}: {}'.format(i + 1, '; '.join(w)))
        if fold_warnings:
            lines.append('    Boundary warnings:')
            lines.extend(fold_warnings)

    if lgbm_params_per_fold:
        lines.append('  LightGBM:')
        fold_warnings = []
        for i, params in enumerate(lgbm_params_per_fold):
            parts = [f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in params.items()]
            lines.append('    Fold {}: {}'.format(i + 1, ', '.join(parts)))
            w = _lgbm_boundary_warnings(params)
            if w:
                fold_warnings.append('    Fold {}: {}'.format(i + 1, '; '.join(w)))
        if fold_warnings:
            lines.append('    Boundary warnings:')
            lines.extend(fold_warnings)

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _compute_feature_importance(merged_features, consumption, cider_feature_cols, best_model_name, best_params=None):
    """
    Retrain best model type on full merged data using CV-selected hyperparameters,
    compute permutation lift and model importance for user-contributed (non-CIDER) features.

    Returns (importance_df, fig) where importance_df has columns
    [feature, lift, importance], sorted by lift descending.
    Returns (None, None) if no custom features exist or on failure.
    """
    if best_params is None:
        best_params = {}

    cider_cols = set(cider_feature_cols)

    data = merged_features.join(consumption, how='inner').dropna(subset=[consumption.name])
    all_feature_cols = [c for c in data.columns if c != consumption.name]

    # Match the training pipeline: drop entirely-NaN columns. Their coefficient
    # / importance is undefined, so they're omitted from the report rather than
    # reported as zero.
    feature_cols = DropAllNaNColumns().fit(data[all_feature_cols]).cols_to_keep_
    non_cider_cols = [c for c in feature_cols if c not in cider_cols]

    if not non_cider_cols:
        return None, None

    X_full = data[feature_cols].values
    y_full = data[consumption.name].values

    X_processed = SimpleImputer().fit_transform(X_full)
    X_scaled = StandardScaler().fit_transform(X_processed)
    X_scaled = np.nan_to_num(X_scaled)

    model_lower = best_model_name.lower()
    importance_dict = {}

    _RF_KEYS = {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'}

    if 'lasso' in model_lower:
        alpha = best_params.get('model__alpha', 0.01)
        model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=10000)
        model.fit(X_scaled, y_full)
        for i, col in enumerate(feature_cols):
            importance_dict[col] = abs(model.coef_[i])
    elif 'ridge' in model_lower:
        alpha = best_params.get('model__alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED, max_iter=10000)
        model.fit(X_scaled, y_full)
        for i, col in enumerate(feature_cols):
            importance_dict[col] = abs(model.coef_[i])
    elif 'forest' in model_lower or 'rf' in model_lower:
        rf_kwargs = {k.replace('model__', ''): v for k, v in best_params.items() if k.replace('model__', '') in _RF_KEYS}
        model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **rf_kwargs)
        model.fit(X_scaled, y_full)
        for i, col in enumerate(feature_cols):
            importance_dict[col] = model.feature_importances_[i]
    elif 'lgbm' in model_lower or 'lightgbm' in model_lower:
        model = lgb.LGBMRegressor(n_estimators=1000, random_state=RANDOM_SEED, verbose=-1, **best_params)
        model.fit(X_scaled, y_full)
        for i, col in enumerate(feature_cols):
            importance_dict[col] = model.feature_importances_[i]
    else:
        # Neural Net — permutation importance only, no model importance
        model = _TorchRegressorWrapper(X_scaled.shape[1])
        model.fit(X_scaled, y_full)
        for col in feature_cols:
            importance_dict[col] = np.nan

    def _score(est, X, y):
        return r2_score(y, est.predict(X))

    perm = _permutation_importance(
        model, X_scaled, y_full, n_repeats=10, random_state=RANDOM_SEED, scoring=_score
    )
    lift_dict = {feature_cols[i]: perm.importances_mean[i] for i in range(len(feature_cols))}

    table_data = [
        {
            'feature': col,
            'lift': lift_dict.get(col, np.nan),
            'importance': importance_dict.get(col, np.nan),
        }
        for col in non_cider_cols
    ]
    importance_df = pd.DataFrame(table_data).sort_values('lift', ascending=False)
    importance_df['importance'] = importance_df['importance'].astype(float)
    importance_df['lift'] = importance_df['lift'].astype(float)

    is_nn = 'neural' in model_lower or 'net' in model_lower
    has_importance = not is_nn and importance_df['importance'].notna().any()
    n_plots = 2 if has_importance else 1
    n_feat = len(non_cider_cols)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, max(4, n_feat * 0.4)))
    axes = np.atleast_1d(axes)
    y_pos = np.arange(n_feat)

    axes[0].barh(y_pos, importance_df['lift'], align='center', alpha=0.8, color='steelblue')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(importance_df['feature'])
    axes[0].set_xlabel('Lift (R² decrease when shuffled)')
    axes[0].set_title('Lift (permutation importance)')

    if has_importance:
        axes[1].barh(y_pos, importance_df['importance'], align='center', alpha=0.8, color='coral')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(importance_df['feature'])
        axes[1].set_xlabel('Feature importance')
        axes[1].set_title(f'Model importance ({best_model_name})')

    fig.suptitle(f'Features — best model: {best_model_name}', y=1.02)
    plt.tight_layout()

    return importance_df, fig


# ---------------------------------------------------------------------------
# Feature correlations
# ---------------------------------------------------------------------------

def _compute_feature_correlations(user_features, consumption, user, featurizer_name, impute_missing=True):
    """
    Compute per-feature Pearson correlation with log_consumption.

    n_obs and coverage reflect raw non-NaN counts before any imputation.
    If impute_missing=True, NaN values are filled with the column median
    before computing Pearson (all n_total rows are then used).
    If impute_missing=False, only rows with a non-NaN value are used.

    Returns a DataFrame with columns:
        [user, featurizer_name, feature, n_obs, coverage, pearson, pearson_pvalue]
    sorted by |pearson| descending. Returns None if fewer than 3 observations.
    """
    data = user_features.join(consumption, how='inner').dropna(subset=[consumption.name])
    n_total = len(data)
    if n_total < 3:
        return None

    y_all = data[consumption.name].values
    rows = []
    for col in user_features.columns:
        x = data[col]
        valid = x.notna()
        n_obs = int(valid.sum())
        coverage = n_obs / n_total
        if impute_missing:
            x_vals = x.fillna(x.median()).values
            y_vals = y_all
        else:
            x_vals = x[valid].values
            y_vals = y_all[valid.values]
        if len(x_vals) < 3 or x_vals.std() == 0:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(x_vals, y_vals)
        rows.append({
            'user': user,
            'featurizer_name': featurizer_name,
            'feature': col,
            'n_obs': n_obs,
            'coverage': float(round(coverage, 4)),
            'pearson': float(round(r, 4)) if not np.isnan(r) else np.nan,
            'pearson_pvalue': float(round(p, 6)) if not np.isnan(p) else np.nan,
        })

    return (
        pd.DataFrame(rows)
        .assign(pearson_abs=lambda d: d['pearson'].abs())
        .sort_values('pearson_abs', ascending=False)
        .drop(columns='pearson_abs')
        .reset_index(drop=True)
    )


def _compute_feature_mutual_info(user_features, consumption, user, featurizer_name, impute_missing=True):
    """
    Compute per-feature mutual information with log_consumption in three variants:
      - unnormalized_mutual_information: raw features (after optional imputation)
      - normalized_mutual_information:   z-score normalized features (zero mean, unit variance)
      - rank_mutual_information:         rank-normalized features (uniform [1/n, ..., 1])

    n_obs and coverage reflect raw non-NaN counts before any imputation.
    If impute_missing=True, NaN values are filled with the column median
    before computing MI. If impute_missing=False, any NaN in any feature
    column will raise (mutual_info_regression does not handle NaNs).

    Returns a DataFrame with columns:
        [user, featurizer_name, feature, n_obs, coverage, unnormalized_mutual_information, normalized_mutual_information, rank_mutual_information]
    sorted by unnormalized_mutual_information descending. Returns None if fewer than 3 observations.
    """
    data = user_features.join(consumption, how='inner').dropna(subset=[consumption.name])
    n_total = len(data)
    if n_total < 3:
        return None

    feature_cols = list(user_features.columns)
    X = data[feature_cols]
    y = data[consumption.name].values

    # Compute n_obs / coverage on raw data before imputation
    n_obs_map = {col: int(X[col].notna().sum()) for col in feature_cols}
    coverage_map = {col: round(n_obs_map[col] / n_total, 4) for col in feature_cols}

    if impute_missing:
        X = X.apply(lambda col: col.fillna(col.median()))

    X_raw = X.values

    X_norm = StandardScaler().fit_transform(X_raw)

    X_rank = np.apply_along_axis(lambda col: rankdata(col) / len(col), 0, X_raw)

    mi_raw                        = mutual_info_regression(X_raw,  y, random_state=RANDOM_SEED)
    normalized_mutual_information = mutual_info_regression(X_norm, y, random_state=RANDOM_SEED)
    rank_mutual_information       = mutual_info_regression(X_rank, y, random_state=RANDOM_SEED)

    rows = [
        {
            'user': user,
            'featurizer_name': featurizer_name,
            'feature': col,
            'n_obs': n_obs_map[col],
            'coverage': float(coverage_map[col]),
            'unnormalized_mutual_information': float(round(mi_raw[i],                        6)),
            'normalized_mutual_information':   float(round(normalized_mutual_information[i], 6)),
            'rank_mutual_information':         float(round(rank_mutual_information[i],       6)),
        }
        for i, col in enumerate(feature_cols)
    ]

    return (
        pd.DataFrame(rows)
        .sort_values('unnormalized_mutual_information', ascending=False)
        .reset_index(drop=True)
    )


def _merge_feature_stats(corr_df, mi_df):
    """Join Pearson and MI DataFrames on feature identity, keeping shared metadata once."""
    if corr_df is None or mi_df is None:
        return corr_df  # return whatever we have
    merged = corr_df.merge(
        mi_df[['user', 'featurizer_name', 'feature', 'unnormalized_mutual_information', 'normalized_mutual_information', 'rank_mutual_information']],
        on=['user', 'featurizer_name', 'feature'],
        how='left',
    )
    return merged.sort_values('pearson', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def _fmt_left_table(df, float_fmt='.4f'):
    """Format a DataFrame with string columns left-aligned and numeric columns right-aligned."""
    cols = list(df.columns)
    str_cols = {c for c in cols if df[c].dtype == object}

    formatted = {}
    for col in cols:
        if col in str_cols:
            formatted[col] = df[col].astype(str).tolist()
        else:
            formatted[col] = [f'{v:{float_fmt}}' if pd.notna(v) else 'N/A' for v in df[col]]

    widths = {col: max(len(col), max((len(s) for s in formatted[col]), default=0)) for col in cols}

    def _row(vals):
        parts = []
        for col, val in zip(cols, vals):
            parts.append(val.ljust(widths[col]) if col in str_cols else val.rjust(widths[col]))
        return '  '.join(parts)

    header = '  '.join(c.ljust(widths[c]) if c in str_cols else c.rjust(widths[c]) for c in cols)
    rows = [_row([formatted[col][i] for col in cols]) for i in range(len(df))]
    return '\n'.join([header] + rows)


def _format_email(result, final_evaluation=False):
    """Return (subject, body) for a human-readable result email."""
    if not result.get('success'):
        subject = 'Featurization Run — Run Error'
        lines = ['Your featurization run encountered an error.', '']
        lines.append(f"Error: {result.get('error', 'unknown')}")
        if result.get('error_type'):
            lines.append(f"Error type: {result['error_type']}")
        if result.get('traceback'):
            lines.append('')
            lines.append('Traceback:')
            lines.append(result['traceback'])
        return subject, '\n'.join(lines)

    if final_evaluation:
        return 'Featurization Run — Final Evaluation Complete', 'Your final evaluation run has completed successfully.'

    def fmt_metric(m):
        return f"{m['mean']:.4f}  (95% CI: {m['ci_low']:.4f} – {m['ci_high']:.4f})"

    use_holdout = result.get('use_holdout', False)

    def fmt_model_results(results):
        lines = []
        for model_name, metrics in results.items():
            lines.append(f"  {model_name}")
            label_r2 = "Holdout R²" if use_holdout else "R²"
            label_sp = "Holdout Spearman" if use_holdout else "Spearman"
            lines.append(f"    {label_r2}:       {fmt_metric(metrics['r2'])}")
            lines.append(f"    Pearson:  {fmt_metric(metrics['pearson'])}")
            lines.append(f"    {label_sp}: {fmt_metric(metrics['spearman'])}")
        return lines

    subject = 'Featurization Run — Results Ready'
    lines = ['Your featurization run has completed.', '']
    lines.append('=== Your Features Only ===')
    lines.extend(fmt_model_results(result.get('results_user_features_only', {})))
    lines.append('')
    lines.append('=== Merged with Existing Features ===')
    lines.extend(fmt_model_results(result.get('results_merged_with_existing', {})))

    importance_df = result.get('importance_df')
    if importance_df is not None and not importance_df.empty:
        lines.append('')
        lines.append('=== Feature Importance ===')
        best_alone = result.get('best_alone_model', 'best model')
        lines.append(f'Showing lift and model importance for your features when evaluated alone (without Cider features) (model: {best_alone}).')
        lines.append('Lift = drop in R² when that feature is randomly shuffled (higher = more important).')
        lines.append('')
        lines.append(_fmt_left_table(importance_df))
        lines.append('')

    correlation_df = result.get('correlation_df')
    if correlation_df is not None and not correlation_df.empty:
        lines.append('')
        lines.append('=== Per-Feature Correlation & Mutual Information ===')
        display_cols = [c for c in ['feature', 'pearson', 'pearson_pvalue', 'normalized_mutual_information', 'rank_mutual_information', 'coverage'] if c in correlation_df.columns]
        lines.append(_fmt_left_table(correlation_df[display_cols]))

    hp_alone = result.get('hp_log_user_only', '')
    hp_merged = result.get('hp_log_merged', '')
    if hp_alone or hp_merged:
        lines.append('')
        lines.append('=== Hyperparameter Details: Generally fine to ignore ===')
        if hp_alone:
            lines.append(hp_alone)
        if hp_merged:
            lines.append(hp_merged)

    return subject, '\n'.join(lines)


def _send_email(to_address, result, importance_fig=None, final_evaluation=False):
    """Send run results via Gmail SMTP. Silently logs on any failure."""
    password = os.environ.get('GMAIL_APP_PASSWORD', 'iaoq hrkt zamw glhy')
    if not password:
        print(f'{_pt()} GMAIL_APP_PASSWORD not set, skipping email')
        return
    subject, body = _format_email(result, final_evaluation=final_evaluation)

    if importance_fig is not None and False:
        msg = MIMEMultipart('mixed')
        msg.attach(MIMEText(body))
        buf = io.BytesIO()
        importance_fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(importance_fig)
        buf.seek(0)
        img = MIMEImage(buf.read(), name='feature_importance.png')
        img.add_header('Content-Disposition', 'attachment', filename='feature_importance.png')
        msg.attach(img)
    else:
        msg = MIMEMultipart('mixed')
        msg.attach(MIMEText(body))

    msg['Subject'] = subject
    msg['From'] = GMAIL_FROM
    msg['To'] = to_address
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(GMAIL_FROM, password)
            smtp.sendmail(GMAIL_FROM, [to_address], msg.as_string())
        print(json.dumps({
            'severity': 'INFO',
            'message': f'{_pt()} Email sent to {to_address}: {subject}',
            'email_to': to_address,
            'email_subject': subject,
            'email_body': body,
        }))
    except Exception as e:
        print(f'{_pt()} Failed to send email to {to_address}: {e}')


# ---------------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------------

def _log_to_sheet(name, user, timestamp, r2, spearman, feat_rt, model_rt, result_type, model_type, holdout_r2=None, holdout_spearman=None, sheet_id=None):
    """Append one row to the Google Sheet. Silently logs and returns on any failure."""
    print(f'{_pt()} logging to sheet')
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)
    row = [name, user, timestamp, r2, spearman, holdout_r2, holdout_spearman, feat_rt, model_rt, result_type, model_type]
    service.spreadsheets().values().append(
        spreadsheetId=sheet_id or SHEET_ID,
        range=f"'{SHEET_TAB}'!A:K",
        valueInputOption='USER_ENTERED',
        insertDataOption='INSERT_ROWS',
        body={'values': [row]},
    ).execute()


def _update_leaderboard(sheet_tab, name, user, timestamp, r2, model_type, sheet_id=None):
    """Update a leaderboard sheet if this run belongs in the top LEADERBOARD_SIZE by R2."""
    print(f'{_pt()} logging to leaderboard')
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)

    active_sheet_id = sheet_id or SHEET_ID
    range_name = f"'{sheet_tab}'!A:E"
    result = service.spreadsheets().values().get(
        spreadsheetId=active_sheet_id,
        range=range_name,
    ).execute()

    rows = result.get('values', [])
    header = rows[0]

    entries = []
    for row in rows[1:]:
        if len(row) >= 5:
            try:
                entries.append([row[0], row[1], row[2], float(row[3]), row[4]])
            except ValueError:
                pass

    duplicate = any(
        e[0] == name and e[1] == user and round(e[3], 6) == round(r2, 6)
        for e in entries
    )

    if not duplicate and (len(entries) < LEADERBOARD_SIZE or r2 > min(e[3] for e in entries)):
        entries.append([name, user, timestamp, r2, model_type])
        entries.sort(key=lambda e: e[3], reverse=True)
        entries = entries[:LEADERBOARD_SIZE]

        service.spreadsheets().values().update(
            spreadsheetId=active_sheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body={'values': [header] + entries},
        ).execute()


def _log_individual_features(name, user, timestamp, correlation_df, sheet_id=None):
    """Append one row per feature to the Individual Features leaderboard tab."""
    if correlation_df is None or correlation_df.empty:
        return
    print(f'{_pt()} logging individual features to sheet')
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)
    def _clean(v):
        try:
            if v != v:  # NaN check
                return ''
        except TypeError:
            pass
        return v

    rows = [
        [
            name, user, row.get('feature', ''), timestamp,
            _clean(row.get('pearson', '')),
            _clean(row.get('unnormalized_mutual_information', '')),
            _clean(row.get('normalized_mutual_information', '')),
            _clean(row.get('rank_mutual_information', '')),
        ]
        for _, row in correlation_df.iterrows()
    ]
    service.spreadsheets().values().append(
        spreadsheetId=sheet_id or SHEET_ID,
        range=f"'{INDIVIDUAL_FEATURES_TAB}'!A:H",
        valueInputOption='USER_ENTERED',
        insertDataOption='INSERT_ROWS',
        body={'values': rows},
    ).execute()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir, final_evaluation=False):
    dated_folder = 'togo_2019_apr_jun' if final_evaluation else 'togo_2018_oct_dec'
    print(f'geetting dated data from {dated_folder}')
    dated_data_path = os.path.join(data_dir, dated_folder)
    dateless_data_path = os.path.join(data_dir, 'togo_dateless')
    togo_dfs = {}

    # Survey outcomes: phone_number, weight, consumption
    survey = pd.read_csv(os.path.join(dateless_data_path, 'survey2018outcomes_no_duplicates.csv'))
    survey['log_consumption'] = np.log1p(survey['consumption'])
    survey.drop_duplicates('phone_number', keep='first', inplace=True)
    survey.index = survey['phone_number']
    survey.drop(columns=['phone_number'], inplace=True)
    togo_dfs['combined_real_consumption'] = survey  # columns: weight, consumption, log_consumption

    # CDR (partitioned parquet directory)
    togo_dfs['combined_real_cdr'] = pd.read_parquet(os.path.join(dated_data_path, 'real_data', 'cdr'))

    # Mobile money (partitioned parquet directory)
    togo_dfs['combined_real_mobile_money'] = pd.read_parquet(os.path.join(dated_data_path, 'real_data', 'mobile_money'))

    # Mobile data (partitioned parquet directory)
    togo_dfs['combined_real_mobile_data'] = pd.read_parquet(os.path.join(dated_data_path, 'real_data', 'mobile_data'))

    # Antennas
    togo_dfs['combined_real_antennas'] = pd.read_csv(os.path.join(dateless_data_path, 'antennas.csv'))

    # Existing CIDER features
    if LOAD_EXISTING_FEATURES:
        df = pd.read_parquet(os.path.join(dated_data_path, 'features'))
        df = df.rename(columns={"name": "phone_number"})
        df.index = df["phone_number"]
        df.drop(columns=["phone_number"], inplace=True)
        togo_dfs['cider_features'] = df
    else:
        togo_dfs['cider_features'] = pd.DataFrame(index=togo_dfs['combined_real_consumption'].index)

    # Shapefiles (GeoJSON)
    geo_path = os.path.join(dateless_data_path, 'geo')
    togo_dfs['shapefiles'] = {}
    for shapefile in [f for f in os.listdir(geo_path) if f.endswith('.geojson')]:
        name = os.path.splitext(shapefile)[0]
        togo_dfs['shapefiles'][name] = gpd.read_file(os.path.join(geo_path, shapefile))

    return togo_dfs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_cv_evaluation(features, consumption, full_run, sample_weight=None, name=None, toy_param_grids=False):
    """
    Run nested CV with Lasso, RF, and NN. Returns bootstrapped metrics per model.

    features:        pd.DataFrame indexed by phone_number
    consumption:     pd.Series of log_consumption indexed by phone_number
    sample_weight:   pd.Series of survey weights indexed by phone_number (optional)
    toy_param_grids: if True, use single-point grids for RF, LGBM, and NN (much faster)
    """

    if name is None:
        name = ''
    merged = features.join(consumption, how='inner')
    if sample_weight is not None:
        merged = merged.join(sample_weight.rename('__weight__'), how='left')
    merged = merged.dropna(subset='log_consumption').reset_index(drop=True)

    X = merged[features.columns]
    y = merged[consumption.name]
    sw = merged['__weight__'].values if sample_weight is not None else None

    lasso_pipeline = Pipeline([
        ('drop_all_nan', DropAllNaNColumns()),
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('model', Lasso(random_state=RANDOM_SEED, max_iter=10000)),
    ])
    ridge_pipeline = Pipeline([
        ('drop_all_nan', DropAllNaNColumns()),
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('model', Ridge(random_state=RANDOM_SEED, max_iter=10000)),
    ])
    rf_pipeline = Pipeline([
        ('drop_all_nan', DropAllNaNColumns()),
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('model', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
    ])

    outer_kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    inner_kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_results = {}

    print(f'{_pt()} starting lasso: {name}')
    lasso_grid = TOY_LASSO_PARAM_GRID if toy_param_grids else LASSO_PARAM_GRID
    y_true, y_pred, lasso_params_per_fold = _cv_sklearn(X, y, lasso_pipeline, lasso_grid, outer_kf, inner_kf, 'lasso', sample_weight=sw)
    all_results['Lasso'] = _bootstrap_metrics(y_true, y_pred)

    print(f'{_pt()} starting ridge: {name}')
    y_true, y_pred, ridge_params_per_fold = _cv_sklearn(X, y, ridge_pipeline, RIDGE_PARAM_GRID, outer_kf, inner_kf, 'ridge', sample_weight=sw)

    all_results['Ridge'] = _bootstrap_metrics(y_true, y_pred)

    rf_params_per_fold = []
    lgbm_params_per_fold = []

    if full_run:
        print(f'{_pt()} starting rf: {name}')
        if toy_param_grids:
            y_true, y_pred, rf_params_per_fold = _cv_sklearn(X, y, rf_pipeline, TOY_RF_PARAM_GRID, outer_kf, inner_kf, 'random forest', sample_weight=sw)
        else:
            y_true, y_pred, rf_params_per_fold = _cv_sklearn(X, y, rf_pipeline, RF_PARAM_GRID, outer_kf, inner_kf, 'random forest', sample_weight=sw, n_iter=RF_N_ITER)
        all_results['Random Forest'] = _bootstrap_metrics(y_true, y_pred)

        print(f'{_pt()} starting NN: {name}')
        torch_grid = TOY_TORCH_PARAMS if toy_param_grids else TORCH_GRID
        y_true, y_pred = _cv_torch(X, y.values, torch_grid, outer_kf, inner_kf, light=NN_LIGHT_MODE, sample_weight=sw)
        all_results['Neural Net'] = _bootstrap_metrics(y_true, y_pred)

        print(f'{_pt()} starting lgbm: {name}')
        lgbm_fixed = TOY_LGBM_PARAMS if toy_param_grids else None
        y_true, y_pred, lgbm_params_per_fold = _cv_lgbm(X, y.values, outer_kf, inner_kf, sample_weight=sw, fixed_params=lgbm_fixed)
        all_results['LightGBM'] = _bootstrap_metrics(y_true, y_pred)

    print(f'{_pt()} done: {name}')
    hp_log = _format_hp_log(name or 'CV', rf_params_per_fold, lgbm_params_per_fold)
    best_params_per_model = {
        'Lasso':         lasso_params_per_fold[0] if lasso_params_per_fold else {},
        'Ridge':         ridge_params_per_fold[0] if ridge_params_per_fold else {},
        'Random Forest': rf_params_per_fold[0]    if rf_params_per_fold    else {},
        'LightGBM':      lgbm_params_per_fold[0]  if lgbm_params_per_fold  else {},
    }
    return all_results, hp_log, best_params_per_model


def _run_holdout_evaluation(features, consumption, full_run, holdout_ids, sample_weight=None, name=None, toy_param_grids=False):
    """
    Tune hyperparameters via inner CV on non-holdout subscribers, train on all
    non-holdout, evaluate on holdout. Returns bootstrapped metrics per model.
    """
    if name is None:
        name = ''
    merged = features.join(consumption, how='inner')
    if sample_weight is not None:
        merged = merged.join(sample_weight.rename('__weight__'), how='left')
    merged = merged.dropna(subset='log_consumption')

    holdout_mask = merged.index.isin(holdout_ids)
    X_train = merged.loc[~holdout_mask, features.columns]
    y_train = merged.loc[~holdout_mask, consumption.name]
    X_holdout = merged.loc[holdout_mask, features.columns]
    y_holdout = merged.loc[holdout_mask, consumption.name]
    sw_train = merged.loc[~holdout_mask, '__weight__'].values if sample_weight is not None else None

    lasso_pipeline = Pipeline([
        ('drop_all_nan', DropAllNaNColumns()),
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('model', Lasso(random_state=RANDOM_SEED, max_iter=10000)),
    ])
    ridge_pipeline = Pipeline([
        ('drop_all_nan', DropAllNaNColumns()),
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('model', Ridge(random_state=RANDOM_SEED, max_iter=10000)),
    ])
    rf_pipeline = Pipeline([
        ('drop_all_nan', DropAllNaNColumns()),
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer()),
        ('model', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
    ])

    inner_kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_results = {}

    print(f'{_pt()} starting holdout lasso: {name}')
    lasso_grid = TOY_LASSO_PARAM_GRID if toy_param_grids else LASSO_PARAM_GRID
    gs = GridSearchCV(lasso_pipeline, lasso_grid, cv=inner_kf, n_jobs=-1, verbose=0)
    fit_params = {'model__sample_weight': sw_train} if sw_train is not None else {}
    gs.fit(X_train, y_train, **fit_params)
    all_results['Lasso'] = _bootstrap_metrics(y_holdout.values, gs.best_estimator_.predict(X_holdout))
    lasso_best_params = gs.best_params_

    print(f'{_pt()} starting holdout ridge: {name}')
    gs = GridSearchCV(ridge_pipeline, RIDGE_PARAM_GRID, cv=inner_kf, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train, **fit_params)
    all_results['Ridge'] = _bootstrap_metrics(y_holdout.values, gs.best_estimator_.predict(X_holdout))
    ridge_best_params = gs.best_params_

    rf_params_per_fold = []
    lgbm_params_per_fold = []

    if full_run:
        print(f'{_pt()} starting holdout rf: {name}')
        if toy_param_grids:
            gs = GridSearchCV(rf_pipeline, TOY_RF_PARAM_GRID, cv=inner_kf, n_jobs=-1, verbose=0)
        else:
            gs = RandomizedSearchCV(rf_pipeline, RF_PARAM_GRID, n_iter=RF_N_ITER, cv=inner_kf,
                                    n_jobs=-1, verbose=0, random_state=RANDOM_SEED)
        gs.fit(X_train, y_train, **fit_params)
        all_results['Random Forest'] = _bootstrap_metrics(y_holdout.values, gs.best_estimator_.predict(X_holdout))
        rf_params_per_fold = [gs.best_params_]

        print(f'{_pt()} starting holdout NN: {name}')
        scaler = StandardScaler().fit(X_train.values)
        X_train_s = np.nan_to_num(scaler.transform(X_train.values))
        X_holdout_s = np.nan_to_num(scaler.transform(X_holdout.values))
        y_train_arr = y_train.values
        if NN_LIGHT_MODE or toy_param_grids:
            best_params = TOY_TORCH_PARAMS[0] if toy_param_grids else TORCH_LIGHT_PARAMS
        else:
            best_params = _torch_inner_cv(X_train_s, y_train_arr, TORCH_GRID, inner_kf, sample_weight=sw_train)
        model = _TorchRegressorWrapper(X_train_s.shape[1], **best_params)
        model.fit(X_train_s, y_train_arr, sample_weight=sw_train)
        all_results['Neural Net'] = _bootstrap_metrics(y_holdout.values, model.predict(X_holdout_s))

        print(f'{_pt()} starting holdout lgbm: {name}')
        lgbm_fixed = TOY_LGBM_PARAMS if toy_param_grids else None
        y_ho, lgbm_preds, lgbm_best = _lgbm_holdout(X_train, y_train, X_holdout, y_holdout, inner_kf, sample_weight=sw_train, fixed_params=lgbm_fixed)
        all_results['LightGBM'] = _bootstrap_metrics(y_ho, lgbm_preds)
        lgbm_params_per_fold = [lgbm_best]

    print(f'{_pt()} done holdout: {name}')
    hp_log = _format_hp_log(name or 'Holdout', rf_params_per_fold, lgbm_params_per_fold)
    best_params_per_model = {
        'Lasso':         lasso_best_params,
        'Ridge':         ridge_best_params,
        'Random Forest': rf_params_per_fold[0]   if rf_params_per_fold   else {},
        'LightGBM':      lgbm_params_per_fold[0] if lgbm_params_per_fold else {},
    }
    return all_results, hp_log, best_params_per_model


def _execute_code(user_code):
    """Execute user-submitted code and return the Featurizer class it defines."""
    namespace = {}
    try:
        exec(user_code, namespace)
        if 'Featurizer' not in namespace:
            return None, {'success': False, 'error': 'User code must define a class named "Featurizer"'}
        return namespace['Featurizer'], None
    except Exception as e:
        return None, {
            'success': False,
            'error': str(e),
            'traceback': ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
            'error_type': type(e).__name__,
        }


def run_job(user_code, user, data_dir, full_run, use_holdout=False, toy_param_grids=False, final_evaluation=False, log_txt_path=None):
    """
    Run the full evaluation pipeline for a submitted featurizer.

    Executes user_code, evaluates the featurizer, and emails the results.
    This is the main testable entry point: call it directly with a code string.

    Returns the result dict (same shape as evaluate_featurizer).
    """
    if final_evaluation:
        full_run = True

    FeaturizerClass, error = _execute_code(user_code)
    if error is not None:
        print(f'{_pt()} ERROR executing code: {error}')
        _send_email(user, error, final_evaluation=final_evaluation)
        return error

    result = evaluate_featurizer(FeaturizerClass, data_dir, user=user, full_run=full_run, use_holdout=use_holdout, toy_param_grids=toy_param_grids, final_evaluation=final_evaluation)
    importance_fig = result.pop('_importance_fig', None)
    _send_email(user, result, importance_fig=importance_fig, final_evaluation=final_evaluation)

    if final_evaluation and result.get('success'):
        if log_txt_path:
            try:
                _GCS_BUCKET = 'featurization-test-bucket'
                _DATA_PREFIX = '/data/'
                client = _gcs.Client()
                bucket = client.bucket(_GCS_BUCKET)
                src_blob_name = log_txt_path.removeprefix(_DATA_PREFIX)
                dest_blob_name = 'successful_final_runs/' + os.path.basename(log_txt_path)
                bucket.copy_blob(bucket.blob(src_blob_name), bucket, dest_blob_name)
                print(f'{_pt()} Copied final evaluation log to gs://{_GCS_BUCKET}/{dest_blob_name}')
            except Exception as e:
                print(f'Warning: Unable to copy log to GCS: {e}')
        else:
            print(f'Warning: Unable to copy log. Log path not provided.')

    return result


def evaluate_featurizer(FeaturizerClass, data_dir, user, full_run, use_holdout=False, toy_param_grids=False, final_evaluation=False):
    """
    Load data from data_dir, instantiate FeaturizerClass, run featurize(),
    validate the output, and evaluate features against consumption.

    Args:
        FeaturizerClass: A class with a featurize() method.
        data_dir:        Top-level data directory. Expected structure:
                           {data_dir}/togo_2018_oct_dec/real_data/  — CSVs and parquet subdirs
                           {data_dir}/togo_2018_oct_dec/features/   — CIDER feature parquets
                           {data_dir}/togo_2018_oct_dec/real_data/geo/ — GeoJSONs

    Returns:
        dict with keys: success, results_user_features_only,
                        results_merged_with_existing  (or success, error on failure).

    For local testing:
        from evaluate import evaluate_featurizer

        class MyFeaturizer:
            def featurize(self, cdr, mobile_money, mobile_data, recharges, antennas, shapefiles):
                return my_features_df

        result = evaluate_featurizer(MyFeaturizer, data_dir='/local/path/to/data')
    """
    try:
        active_sheet_id = FINAL_EVAL_SHEET_ID if final_evaluation else SHEET_ID
        togo_dfs = load_data(data_dir, final_evaluation=final_evaluation)

        featurizer = FeaturizerClass()
        if not hasattr(featurizer, 'featurize'):
            return {'success': False, 'error': 'Featurizer class must have a "featurize" method'}

        name = ''
        if hasattr(featurizer, 'name'):
            if isinstance(featurizer.name, str):
                name = featurizer.name
            elif callable(featurizer.name):
                try:
                    name = FeaturizerClass.name(featurizer)
                except TypeError:
                    name = featurizer.name()
            if not isinstance(name, str):
                name = ''

        consumption = togo_dfs['combined_real_consumption']['log_consumption']
        weights = togo_dfs['combined_real_consumption']['weight']
        cider_features = togo_dfs['cider_features']

        featurize_kwargs = dict(
            cdr=togo_dfs['combined_real_cdr'].copy(),
            mobile_money=togo_dfs['combined_real_mobile_money'].copy(),
            mobile_data=togo_dfs['combined_real_mobile_data'].copy(),
            recharges=None,
            antennas=togo_dfs['combined_real_antennas'].copy(),
            shapefiles=togo_dfs['shapefiles'].copy(),
        )
        togo_dfs['cider_features']
        if 'existing_features' in inspect.signature(featurizer.featurize).parameters:
            featurize_kwargs['existing_features'] = togo_dfs['cider_features'].copy()
        if use_holdout:
            holdout_path = os.path.join(data_dir, 'togo_dateless', 'hold_out_subscribers.csv')
            holdout_ids = set(pd.read_csv(holdout_path)['phone_number'])
            train_consumption = consumption[~consumption.index.isin(holdout_ids)]
            if 'consumption' in inspect.signature(featurizer.featurize).parameters:
                featurize_kwargs['consumption'] = train_consumption.copy()

        feat_start = time.time()
        user_features = featurizer.featurize(**featurize_kwargs)
        feat_rt = round(time.time() - feat_start, 2)

        if not isinstance(user_features, pd.DataFrame):
            return {'success': False, 'error': 'featurize must return a pandas DataFrame'}

        merged_features = user_features.join(cider_features, how='outer')

        timestamp = datetime.datetime.now(PACIFIC).isoformat(timespec='seconds')

        if use_holdout:
            cv_start = time.time()
            results_alone, hp_log_alone, best_params_alone = _run_holdout_evaluation(
                user_features, consumption, full_run, holdout_ids,
                sample_weight=weights, name='user only', toy_param_grids=toy_param_grids,
            )
            model_rt_alone = round(time.time() - cv_start, 2)

            cv_start = time.time()
            results_merged, hp_log_merged, best_params_merged = _run_holdout_evaluation(
                merged_features, consumption, full_run, holdout_ids,
                sample_weight=weights, name='merged with existing', toy_param_grids=toy_param_grids,
            )
            model_rt_merged = round(time.time() - cv_start, 2)

            best_alone_model = max(results_alone, key=lambda m: results_alone[m]['r2']['mean'])
            best_merged_model = max(results_merged, key=lambda m: results_merged[m]['r2']['mean'])

            _log_to_sheet(name, user, timestamp,
                          None, None,
                          feat_rt, model_rt_alone, 'user only', best_alone_model,
                          holdout_r2=results_alone[best_alone_model]['r2']['mean'],
                          holdout_spearman=results_alone[best_alone_model]['spearman']['mean'],
                          sheet_id=active_sheet_id)
            _log_to_sheet(name, user, timestamp,
                          None, None,
                          feat_rt, model_rt_merged, 'merged with existing', best_merged_model,
                          holdout_r2=results_merged[best_merged_model]['r2']['mean'],
                          holdout_spearman=results_merged[best_merged_model]['spearman']['mean'],
                          sheet_id=active_sheet_id)

            importance_df, importance_fig = None, None
            if not final_evaluation:
                importance_df, importance_fig = _compute_feature_importance(
                    user_features, consumption, [], best_alone_model,
                    best_params=best_params_alone.get(best_alone_model, {}),
                )

            correlation_df = _merge_feature_stats(
                _compute_feature_correlations(user_features, consumption, user, name),
                _compute_feature_mutual_info(user_features, consumption, user, name),
            )

            if final_evaluation:
                _log_individual_features(name, user, timestamp, correlation_df, sheet_id=active_sheet_id)

            return {
                'success': True,
                'use_holdout': True,
                'results_user_features_only': results_alone,
                'results_merged_with_existing': results_merged,
                'hp_log_user_only': hp_log_alone,
                'hp_log_merged': hp_log_merged,
                'importance_df': importance_df,
                'best_alone_model': best_alone_model,
                '_importance_fig': importance_fig,
                'correlation_df': correlation_df,
            }

        cv_start = time.time()
        results_alone, hp_log_alone, best_params_alone = _run_cv_evaluation(
            user_features, consumption, full_run, sample_weight=weights, name='user only', toy_param_grids=toy_param_grids,
        )
        model_rt_alone = round(time.time() - cv_start, 2)

        cv_start = time.time()
        results_merged, hp_log_merged, best_params_merged = _run_cv_evaluation(
            merged_features, consumption, full_run, sample_weight=weights, name='merged with existing', toy_param_grids=toy_param_grids,
        )
        model_rt_merged = round(time.time() - cv_start, 2)

        best_alone_model = max(results_alone, key=lambda m: results_alone[m]['r2']['mean'])
        best_merged_model = max(results_merged, key=lambda m: results_merged[m]['r2']['mean'])

        _log_to_sheet(name, user, timestamp,
                      results_alone[best_alone_model]['r2']['mean'],
                      results_alone[best_alone_model]['spearman']['mean'],
                      feat_rt, model_rt_alone, 'user only', best_alone_model,
                      sheet_id=active_sheet_id)
        _log_to_sheet(name, user, timestamp,
                      results_merged[best_merged_model]['r2']['mean'],
                      results_merged[best_merged_model]['spearman']['mean'],
                      feat_rt, model_rt_merged, 'merged with existing', best_merged_model,
                      sheet_id=active_sheet_id)

        _update_leaderboard(LEADERBOARD_STANDALONE_TAB, name, user, timestamp,
                            results_alone[best_alone_model]['r2']['mean'], best_alone_model,
                            sheet_id=active_sheet_id)
        _update_leaderboard(LEADERBOARD_MERGED_TAB, name, user, timestamp,
                            results_merged[best_merged_model]['r2']['mean'], best_merged_model,
                            sheet_id=active_sheet_id)

        importance_df, importance_fig = None, None
        if not final_evaluation:
            importance_df, importance_fig = _compute_feature_importance(
                user_features, consumption, [], best_alone_model,
                best_params=best_params_alone.get(best_alone_model, {}),
            )

        correlation_df = _merge_feature_stats(
            _compute_feature_correlations(user_features, consumption, user, name),
            _compute_feature_mutual_info(user_features, consumption, user, name),
        )

        if final_evaluation:
            _log_individual_features(name, user, timestamp, correlation_df, sheet_id=active_sheet_id)

        return {
            'success': True,
            'results_user_features_only': results_alone,
            'results_merged_with_existing': results_merged,
            'hp_log_user_only': hp_log_alone,
            'hp_log_merged': hp_log_merged,
            'importance_df': importance_df,
            'best_alone_model': best_alone_model,
            '_importance_fig': importance_fig,
            'correlation_df': correlation_df,
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
            'error_type': type(e).__name__,
        }
