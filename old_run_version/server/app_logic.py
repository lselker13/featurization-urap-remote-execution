import datetime
import os
import smtplib
import time
import traceback
import uuid
import warnings
from email.mime.text import MIMEText

import threading
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import google.auth
from googleapiclient.discovery import build

USE_NN = False
USE_RF = False

PACIFIC = ZoneInfo('America/Los_Angeles')

GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'iaoq hrkt zamw glhy')

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
    "model__max_depth": [None, 10, 20, 40],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", "log2", 0.5],
}
TORCH_GRID = list(ParameterGrid({
    "lr": [1e-2, 1e-3, 3e-4],
    "patience": [10, 20, 30],
}))

LOAD_EXISTING_FEATURES = True

SHEET_ID = '13vZkBNoI1TNKEuWJhtFospr_XNR23DpZTobjaKAJneA'
SHEET_TAB = 'Results log'
LEADERBOARD_STANDALONE_TAB = 'Leaderboard: Stand-alone'
LEADERBOARD_MERGED_TAB = 'Leaderboard: Along with existing features'
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
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class _TorchRegressorWrapper:
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3, epochs=500, patience=20, device='cpu'):
        self.device = device
        self.model = _TorchMLP(input_dim, hidden_dim).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y, batch_size=64):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
        )
        best_loss, patience_left, best_state = float('inf'), self.patience, None

        for _ in range(self.epochs):
            self.model.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                self.opt.zero_grad()
                self.loss_fn(self.model(X_b), y_b).backward()
                self.opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = self.loss_fn(
                    self.model(X_t.to(self.device)),
                    y_t.to(self.device)
                ).item()
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


def _cv_sklearn(X, y, pipeline, param_grid, outer_kf, inner_kf, model_type_name='unspecified'):
    """Nested CV for sklearn pipelines; returns pooled (y_true, y_pred) arrays."""
    fold_preds = []
    for i, (train_idx, test_idx )in enumerate(outer_kf.split(X)):
        print(f'running grid search fold {i} for model type {model_type_name}')
        gs = GridSearchCV(pipeline, param_grid, cv=inner_kf, n_jobs=-1, verbose=0)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_preds.append((y.iloc[test_idx].values, gs.best_estimator_.predict(X.iloc[test_idx])))
    y_true = np.concatenate([t for t, _ in fold_preds])
    y_pred = np.concatenate([p for _, p in fold_preds])
    return y_true, y_pred


def _torch_inner_cv(X, y, param_grid, inner_kf):
    """Inner CV for torch hyperparameter selection."""
    best_params, best_score = None, -np.inf
    for params in param_grid:
        scores = []
        for tr, va in inner_kf.split(X):
            m = _TorchRegressorWrapper(X.shape[1], **params)
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[va], m.predict(X[va])))
        if np.mean(scores) > best_score:
            best_score, best_params = np.mean(scores), params
    return best_params


def _cv_torch(X, y, param_grid, outer_kf, inner_kf):
    """Nested CV for the torch MLP; scales inside each outer fold."""
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y if isinstance(y, np.ndarray) else np.asarray(y)
    fold_preds = []
    for train_idx, test_idx in outer_kf.split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        # Replace NaN/inf that may remain after scaling
        X_train_s = np.nan_to_num(X_train_s)
        X_test_s = np.nan_to_num(X_test_s)
        best_params = _torch_inner_cv(X_train_s, y_train, param_grid, inner_kf)
        model = _TorchRegressorWrapper(X_train_s.shape[1], **best_params)
        model.fit(X_train_s, y_train)
        fold_preds.append((y_test, model.predict(X_test_s)))
    y_true = np.concatenate([t for t, _ in fold_preds])
    y_pred = np.concatenate([p for _, p in fold_preds])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def _format_email(result):
    """Return (subject, body) for a human-readable result email."""
    if not result.get('success'):
        subject = 'Featurization Competition — Run Error'
        lines = ['Your featurization run encountered an error.', '']
        lines.append(f"Error: {result.get('error', 'unknown')}")
        if result.get('error_type'):
            lines.append(f"Error type: {result['error_type']}")
        if result.get('traceback'):
            lines.append('')
            lines.append('Traceback:')
            lines.append(result['traceback'])
        return subject, '\n'.join(lines)

    def fmt_metric(m):
        return f"{m['mean']:.4f}  (95% CI: {m['ci_low']:.4f} – {m['ci_high']:.4f})"

    def fmt_model_results(results):
        lines = []
        for model_name, metrics in results.items():
            lines.append(f"  {model_name}")
            lines.append(f"    R²:       {fmt_metric(metrics['r2'])}")
            lines.append(f"    Pearson:  {fmt_metric(metrics['pearson'])}")
            lines.append(f"    Spearman: {fmt_metric(metrics['spearman'])}")
        return lines

    subject = 'Featurization Competition — Results Ready'
    lines = ['Your featurization run has completed.', '']
    lines.append('=== Your Features Only ===')
    lines.extend(fmt_model_results(result.get('results_user_features_only', {})))
    lines.append('')
    lines.append('=== Merged with Existing Features ===')
    lines.extend(fmt_model_results(result.get('results_merged_with_existing', {})))
    return subject, '\n'.join(lines)


def _send_email(to_address, result):
    """Send run results via Gmail SMTP. Silently logs on any failure."""
    if not GMAIL_APP_PASSWORD:
        print('GMAIL_APP_PASSWORD not set, skipping email')
        return
    subject, body = _format_email(result)
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = GMAIL_FROM
    msg['To'] = to_address
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
            smtp.sendmail(GMAIL_FROM, [to_address], msg.as_string())
        print(f'Email sent to {to_address}')
    except Exception as e:
        print(f'Failed to send email to {to_address}: {e}')


# ---------------------------------------------------------------------------
# Submission logging
# ---------------------------------------------------------------------------

def log_submission(user_code, user, log_dir):
    print('logging user code')
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now(PACIFIC)
    timestamp_display = now.strftime('%Y-%m-%d %H:%M:%S %Z')
    filename_ts = now.strftime('%Y-%m-%dT%H:%M:%S')
    safe_user = ''.join(c if c.isalnum() or c in '.@-' else '_' for c in (user or 'anonymous'))
    submission_id = uuid.uuid4().hex[:8]
    filename = f"{filename_ts}_{safe_user}_{submission_id}.txt"
    content = (
        f"User: {user or 'anonymous'}\n"
        f"Timestamp: {timestamp_display}\n"
        f"\n"
        f"--- Submitted Code ---\n"
        f"\n"
        f"{user_code}\n"
    )
    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write(content)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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


def _run_and_email(user_code, user, data_dir):
    """Run full evaluation and email results. Executes in a background thread."""
    FeaturizerClass, error = _execute_code(user_code)
    if error is not None:
        print('ERROR: ', error)
        raise error
    assert error is None
    result = evaluate_featurizer(FeaturizerClass, data_dir, user=user)
    _send_email(user, result)


def run_submission(user_code, user, data_dir, log_dir):
    """
    Top-level entry point to this file, called from app.py.
    Logs submission, validates code and featurizer synchronously, then returns
    202 immediately. Evaluation and emailing run in a background thread.
    Raises on logging failure.
    """
    log_submission(user_code, user, log_dir)

    FeaturizerClass, error = _execute_code(user_code)
    if error is not None:
        return error, 400
    try:
        featurizer = FeaturizerClass()
        if not hasattr(featurizer, 'featurize'):
            return {'success': False, 'error': 'Featurizer class must have a "featurize" method'}, 400
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
            'error_type': type(e).__name__,
        }, 400
    
    thread = threading.Thread(target=_run_and_email, args=(user_code, user, data_dir), daemon=True)
    thread.start()

    return {'success': True, 'message': 'Submission accepted. Results will be emailed to you when the run is finished.'}, 202


# ---------------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------------

def _log_to_sheet(name, user, timestamp, r2, spearman, feat_rt, model_rt, result_type, model_type):
    """Append one row to the Google Sheet. Silently logs and returns on any failure."""
    print('logging to sheet')
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)
    row = [name, user, timestamp, r2, spearman, feat_rt, model_rt, result_type, model_type]
    service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=f"'{SHEET_TAB}'!A:H",
        valueInputOption='USER_ENTERED',
        insertDataOption='INSERT_ROWS',
        body={'values': [row]},
    ).execute()


def _update_leaderboard(sheet_tab, name, user, timestamp, r2, model_type):
    """Update a leaderboard sheet if this run belongs in the top LEADERBOARD_SIZE by R2."""
    print('logging to leaderboard')
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)

    range_name = f"'{sheet_tab}'!A:E"
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
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
    duplicate=False

    if not duplicate and (len(entries) < LEADERBOARD_SIZE or r2 > min(e[3] for e in entries)):
        entries.append([name, user, timestamp, r2, model_type])
        entries.sort(key=lambda e: e[3], reverse=True)
        entries = entries[:LEADERBOARD_SIZE]

        service.spreadsheets().values().update(
            spreadsheetId=SHEET_ID,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body={'values': [header] + entries},
        ).execute()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir):
    folder_path = os.path.join(data_dir, 'togo_full_2018_10')
    togo_dfs = {}

    for csv_file in [f for f in os.listdir(folder_path) if f.endswith('.csv')]:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        if "consumption" in csv_file:
            df['log_consumption'] = np.log1p(df['consumption'])
            df.drop_duplicates('phone_number', keep='first', inplace=True)
            df.index = df["phone_number"]
            df.drop(columns=["phone_number"], inplace=True)
            togo_dfs['consumption'] = df
        togo_dfs[os.path.splitext(csv_file)[0]] = df

    if LOAD_EXISTING_FEATURES:
        df = pd.read_parquet(os.path.join(data_dir, 'togo_features_2018_10/features'))
        df = df.rename(columns={"name": "phone_number"})
        df.index = df["phone_number"]
        df.drop(columns=["phone_number"], inplace=True)
        togo_dfs['cider_features'] = df
    else:
        togo_dfs['cider_features'] = pd.DataFrame(index=togo_dfs['consumption'].index)

    shapefiles_path = os.path.join(data_dir, 'togo_admin_boundaries')
    togo_dfs['shapefiles'] = {}
    for shapefile in [f for f in os.listdir(shapefiles_path) if f.endswith('.geojson')]:
        name = os.path.splitext(shapefile)[0]
        togo_dfs['shapefiles'][name] = gpd.read_file(os.path.join(shapefiles_path, shapefile))

    return togo_dfs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_cv_evaluation(features, consumption):
    """
    Run nested CV with Lasso, RF, and NN. Returns bootstrapped metrics per model.

    features:    pd.DataFrame indexed by phone_number
    consumption: pd.Series of log_consumption indexed by phone_number
    """
    merged = features.join(consumption, how='inner')
    merged = merged.dropna(subset='log_consumption').reset_index(drop=True)

    X = merged[features.columns]
    y = merged[consumption.name]

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

    print(f'starting lasso: {datetime.datetime.now(PACIFIC)}')
    y_true, y_pred = _cv_sklearn(X, y, lasso_pipeline, LASSO_PARAM_GRID, outer_kf, inner_kf, 'lasso')
    all_results['Lasso'] = _bootstrap_metrics(y_true, y_pred)

    print(f'starting ridge: {datetime.datetime.now(PACIFIC)}')
    y_true, y_pred = _cv_sklearn(X, y, ridge_pipeline, RIDGE_PARAM_GRID, outer_kf, inner_kf, 'ridge')
    all_results['Ridge'] = _bootstrap_metrics(y_true, y_pred)

    if USE_RF:
        print(f'starting rf: {datetime.datetime.now(PACIFIC)}')
        y_true, y_pred = _cv_sklearn(X, y, rf_pipeline, RF_PARAM_GRID, outer_kf, inner_kf, 'random forest')
        all_results['Random Forest'] = _bootstrap_metrics(y_true, y_pred)

    if USE_NN:
        print(f'starting NN: {datetime.datetime.now(PACIFIC)}')
        y_true, y_pred = _cv_torch(X, y.values, TORCH_GRID, outer_kf, inner_kf)
        all_results['Neural Net'] = _bootstrap_metrics(y_true, y_pred)
    print(f'done {datetime.datetime.now(PACIFIC)}')
    return all_results


def evaluate_featurizer(FeaturizerClass, data_dir, user):
    """
    Load data from data_dir, instantiate FeaturizerClass, run featurize(),
    validate the output, and evaluate features against consumption.

    Args:
        FeaturizerClass: A class with a featurize() method.
        data_dir:        Top-level data directory. Expected structure:
                           {data_dir}/togo/togo_data_2018_10/  — CSVs
                           {data_dir}/togo_admin_boundaries/   — GeoJSONs

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
        togo_dfs = load_data(data_dir)

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

        feat_start = time.time()
        user_features = featurizer.featurize(
            cdr=togo_dfs['combined_real_cdr'],
            mobile_money=togo_dfs['combined_real_mobile_money'],
            mobile_data=togo_dfs['combined_real_mobile_data'],
            recharges=None,
            antennas=togo_dfs['combined_real_antennas'],
            shapefiles=togo_dfs['shapefiles'],
        )
        feat_rt = round(time.time() - feat_start, 2)

        if not isinstance(user_features, pd.DataFrame):
            return {'success': False, 'error': 'featurize must return a pandas DataFrame'}

        consumption = togo_dfs['combined_real_consumption']['log_consumption']
        cider_features = togo_dfs['cider_features']

        cv_start = time.time()
        results_alone = _run_cv_evaluation(user_features, consumption)
        model_rt_alone = round(time.time() - cv_start, 2)

        merged_features = user_features.join(cider_features, how='outer')
        cv_start = time.time()
        results_merged = _run_cv_evaluation(merged_features, consumption)
        model_rt_merged = round(time.time() - cv_start, 2)

        timestamp = datetime.datetime.now(PACIFIC).isoformat(timespec='seconds')

        best_alone_r2, best_alone_model = -np.inf, None
        for model_type, results in results_alone.items():
            if results['r2']['mean'] > best_alone_r2:
                best_alone_r2 = results['r2']['mean']
                best_alone_model = model_type
        
        best_merged_r2, best_merged_model = -np.inf, None
        for model_type, results in results_merged.items():
            if results['r2']['mean'] > best_merged_r2:
                best_merged_r2 = results['r2']['mean']
                best_merged_model = model_type


        _log_to_sheet(name, user, timestamp,
                      results_alone[best_alone_model]['r2']['mean'],
                      results_alone[best_alone_model]['spearman']['mean'],
                      feat_rt, model_rt_alone, 'user only', best_alone_model)
        _log_to_sheet(name, user, timestamp,
                      results_merged[best_merged_model]['r2']['mean'],
                      results_merged[best_merged_model]['spearman']['mean'],
                      feat_rt, model_rt_merged, 'merged with existing', best_merged_model)
        
        _update_leaderboard(LEADERBOARD_STANDALONE_TAB, name, user, timestamp,
                            results_alone[best_alone_model]['r2']['mean'])
        _update_leaderboard(LEADERBOARD_MERGED_TAB, name, user, timestamp,
                            results_merged[best_merged_model]['r2']['mean'])

        return {
            'success': True,
            'results_user_features_only': results_alone,
            'results_merged_with_existing': results_merged,
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
            'error_type': type(e).__name__,
        }
