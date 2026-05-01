import datetime
import json
import os
import traceback
import uuid
from zoneinfo import ZoneInfo

import google.auth
import google.auth.transport.requests
import google.cloud.storage
from flask import Flask, request

PACIFIC = ZoneInfo('America/Los_Angeles')
ALLOW_FULL_RUN = True

LOG_DIR = os.environ.get('LOG_DIR', '/data/submission_logs')
SPEC_DIR = os.environ.get('SPEC_DIR', '/data/workspace/run_specs')
RATE_LIMIT_BUCKET = os.environ.get('RATE_LIMIT_BUCKET', 'featurization-test-bucket')
FULL_RUN_WEEKLY_LIMIT = 5
FINAL_EVAL_LIMIT = 1

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_user(user):
    return ''.join(c if c.isalnum() or c in '.@-' else '_' for c in (user or 'anonymous'))


def _current_iso_week():
    iso = datetime.datetime.now(PACIFIC).isocalendar()
    return iso.year, iso.week


def _rate_limit_blob_name(safe_user, year, week):
    return f'rate_limits/{safe_user}/{year}_W{week:02d}.json'


# ---------------------------------------------------------------------------
# Rate limiting (GCS-backed)
# ---------------------------------------------------------------------------

def _gcs_client():
    return google.cloud.storage.Client()


def _get_full_run_count(safe_user, year, week):
    """Return the current full_run count for this user+week, or 0 if not found."""
    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    blob = bucket.blob(_rate_limit_blob_name(safe_user, year, week))
    if not blob.exists():
        return 0
    data = json.loads(blob.download_as_text())
    return data.get('count', 0)


def _increment_full_run_count(safe_user, year, week):
    """Increment the full_run counter for this user+week in GCS. Returns new count."""
    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    blob = bucket.blob(_rate_limit_blob_name(safe_user, year, week))
    if blob.exists():
        data = json.loads(blob.download_as_text())
    else:
        data = {'user': safe_user, 'year': year, 'week': week, 'count': 0}
    data['count'] = data.get('count', 0) + 1
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    return data['count']


def _reset_full_run_count(safe_user, year, week):
    """Reset the full_run counter for this user+week to 0 in GCS."""
    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    blob = bucket.blob(_rate_limit_blob_name(safe_user, year, week))
    data = {'user': safe_user, 'year': year, 'week': week, 'count': 0}
    blob.upload_from_string(json.dumps(data), content_type='application/json')


def _get_final_eval_count(safe_user):
    """Return the total final_evaluation submission count for this user, or 0 if not found."""
    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    blob = bucket.blob(f'final_evaluation_runs/{safe_user}.json')
    if not blob.exists():
        return 0
    data = json.loads(blob.download_as_text())
    return data.get('count', 0)


def _increment_final_eval_count(safe_user):
    """Increment the final_evaluation counter for this user in GCS. Returns new count."""
    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    blob = bucket.blob(f'final_evaluation_runs/{safe_user}.json')
    if blob.exists():
        data = json.loads(blob.download_as_text())
    else:
        data = {'user': safe_user, 'count': 0}
    data['count'] = data.get('count', 0) + 1
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    return data['count']


# ---------------------------------------------------------------------------
# Submission logging
# ---------------------------------------------------------------------------

def log_submission(user_code, user, log_dir, full_run, use_holdout=False, toy_param_grids=False, final_evaluation=False):
    """Write a human-readable log to submission_logs/ and a JSON run spec to workspace/run_specs/. Returns the spec path."""
    print('logging user code')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(SPEC_DIR, exist_ok=True)
    now = datetime.datetime.now(PACIFIC)
    timestamp_display = now.strftime('%Y-%m-%d %H:%M:%S %Z')
    filename_ts = now.strftime('%Y-%m-%dT%H:%M:%S')
    safe = _safe_user(user)
    submission_id = uuid.uuid4().hex[:8]
    base = f"{filename_ts}_{safe}_{submission_id}"

    # Human-readable log
    txt_content = (
        f"User: {user or 'anonymous'}\n"
        f"Timestamp: {timestamp_display}\n"
        f"\n"
        f"--- Submitted Code ---\n"
        f"\n"
        f"{user_code}\n"
    )
    with open(os.path.join(log_dir, f"{base}.txt"), 'w') as f:
        f.write(txt_content)

    # JSON run spec for the job to read
    txt_path = os.path.join(log_dir, f"{base}.txt")
    spec_path = os.path.join(SPEC_DIR, f"{base}.json")
    with open(spec_path, 'w') as f:
        json.dump({'user': user, 'code': user_code, 'full_run': full_run, 'use_holdout': use_holdout, 'toy_param_grids': toy_param_grids, 'final_evaluation': final_evaluation, 'log_txt_path': txt_path}, f)

    return spec_path, submission_id


# ---------------------------------------------------------------------------
# Job trigger
# ---------------------------------------------------------------------------

def _trigger_cloud_run_job(json_file_path):
    """Trigger the Cloud Run Job execution, passing the submission JSON path via env var."""
    job_name = os.environ['JOB_NAME']
    region = os.environ.get('JOB_REGION', 'us-central1')
    print('Fetching credentials')

    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    gcp_project = os.environ.get('GCP_PROJECT', project)

    print('Starting authenticated session')

    authed_session = google.auth.transport.requests.AuthorizedSession(creds)
    url = (
        f'https://run.googleapis.com/v2/projects/{gcp_project}'
        f'/locations/{region}/jobs/{job_name}:run'
    )
    body = {
        'overrides': {
            'containerOverrides': [{
                'env': [{'name': 'SUBMISSION_JSON_PATH', 'value': json_file_path}],
            }]
        }
    }
    print('Posting request to Cloud Run Job')

    resp = authed_session.post(url, json=body, timeout=30)

    resp.raise_for_status()
    print(f'Job triggered: {resp.json().get("name")}')


def _trigger_vertex_job(json_file_path, user=None, submission_id=None, full_run=None):
    """Trigger a Vertex AI Custom Job, passing the submission JSON path via env var."""
    region = os.environ.get('JOB_REGION', 'us-central1')
    image_uri = os.environ.get('IMAGE_URI', 'us-central1-docker.pkg.dev/gol-cdr-featurization-comp/featurization-jobs/featurization-evaluator-vertex:latest')
    machine_type = os.environ.get('MACHINE_TYPE', 'n1-highmem-32')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD', '')
    print('Fetching credentials')

    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    gcp_project = os.environ.get('GCP_PROJECT', project)

    print('Starting authenticated session')

    authed_session = google.auth.transport.requests.AuthorizedSession(creds)
    url = (
        f'https://{region}-aiplatform.googleapis.com/v1'
        f'/projects/{gcp_project}/locations/{region}/customJobs'
    )
    display_name = f'featurization-evaluator-{datetime.datetime.now(PACIFIC).strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'
    def _label_safe(s):
        return ''.join(c if c.isalnum() or c == '-' else '_' for c in (s or 'anonymous').lower())[:63]

    body = {
        'displayName': display_name,
        'labels': {
            'user': _label_safe(user),
            'full_run': str(full_run)
            # 'submission-id': _label_safe(submission_id) if submission_id else 'unknown',
        },
        'jobSpec': {
            'workerPoolSpecs': [{
                'machineSpec': {'machineType': machine_type},
                'replicaCount': 1,
                'containerSpec': {
                    'imageUri': image_uri,
                    'env': [
                        {'name': 'SUBMISSION_JSON_PATH', 'value': json_file_path},
                        {'name': 'GMAIL_APP_PASSWORD', 'value': gmail_password},
                    ],
                },
            }],
        },
    }
    print('Posting request to Vertex AI')

    resp = authed_session.post(url, json=body, timeout=30)

    resp.raise_for_status()
    print(f'Job triggered: {resp.json().get("name")}')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_submission(user_code, user, log_dir, full_run, use_holdout=False, toy_param_grids=False, final_evaluation=False):
    """
    Top-level entry point called from app.py.
    Logs submission, validates code synchronously, triggers the job,
    then returns 202 immediately.
    Raises on logging failure.
    """

    # this is probably overkill, todo simplify after unit testing it
    message = 'Submission successful; starting evaluation. Results will be emailed to you from featurizationtestserver@gmail.com.'

    if isinstance(full_run, bool):
        pass
    elif full_run is None or not isinstance(full_run, str):
        message += ' Warning: specify full_run=True/False. Defaulting to False.'
        full_run = False
    elif full_run.lower() == 'true':
        full_run = True
    elif full_run.lower() == 'false':
        full_run = False
    else:
        message += f' Warning: full_run expected to be True/False; instead got {full_run}. Defaulting to False.'
        full_run = False

    if isinstance(use_holdout, bool):
        pass
    elif use_holdout is None or not isinstance(use_holdout, str):
        use_holdout = False
    elif use_holdout.lower() == 'true':
        use_holdout = True
    elif use_holdout.lower() == 'false':
        use_holdout = False
    else:
        use_holdout = False

    if full_run and (not ALLOW_FULL_RUN):
        message += ' Warning: full_run not yet permitted; overriding to False.'
        full_run = False

    print('checking counter')
    safe = _safe_user(user)
    # Rate limit check: only applies to full_run jobs
    if full_run:
        year, week = _current_iso_week()
        count = _get_full_run_count(safe, year, week)
        if count >= FULL_RUN_WEEKLY_LIMIT:
            return {
                'success': False,
                'error': (
                    f'Rate limit exceeded: you have used {count}/{FULL_RUN_WEEKLY_LIMIT} '
                    f'full_run submissions for ISO week {year}-W{week:02d}. '
                    f'Limit resets at the start of the next calendar week.'
                ),
            }, 429
    if final_evaluation:
        fe_count = _get_final_eval_count(safe)
        if fe_count >= FINAL_EVAL_LIMIT:
            return {
                'success': False,
                'error': (
                    f'Final evaluation limit reached: you have already used your '
                    f'{FINAL_EVAL_LIMIT} final evaluation submission.'
                ),
            }, 429
    print('logging submission')
    json_file_path, submission_id = log_submission(user_code, user, log_dir, full_run, use_holdout, toy_param_grids, final_evaluation)
    print('logged submission')
    _trigger_vertex_job(json_file_path, user=user, submission_id=submission_id, full_run=full_run)
    print('triggered job')
    # Increment counters only after successful job trigger
    if full_run:
        new_count = _increment_full_run_count(safe, year, week)
        print(f'Rate limit counter for {safe} week {year}-W{week:02d}: {new_count}/{FULL_RUN_WEEKLY_LIMIT}')
    if final_evaluation:
        new_fe_count = _increment_final_eval_count(safe)
        print(f'Final evaluation counter for {safe}: {new_fe_count}/{FINAL_EVAL_LIMIT}')

    return {'success': True, 'message': message.strip() + f" Full run: {full_run}."}, 202


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return {'status': 'running', 'message': 'Featurization evaluation service is alive'}

@app.route('/execute', methods=['POST'])
def execute():
    data = request.get_json()
    if not data or 'code' not in data:
        return {'success': False, 'error': 'No code provided. Send JSON with "code" field.'}, 400
    if not data.get('user'):
        return {'success': False, 'error': 'No user provided. Send JSON with "user" field.'}, 400
    print(f"Received submission: user={data.get('user')} full_run={data.get('full_run')} toy_param_grids={data.get('toy_param_grids')} final_evaluation={data.get('final_evaluation')} code_len={len(data.get('code', ''))}")
    return run_submission(
        data['code'],
        data['user'],
        LOG_DIR,
        data.get('full_run', False),
        data.get('use_holdout', False),
        data.get('toy_param_grids', False),
        data.get('final_evaluation', False),
    )

@app.route('/get_counters', methods=['GET'])
def get_counters():
    """
    Return the full_run counts for all users for a given week.
    Query params: year, week (optional, default = current week)
    Response: {"year": ..., "week": ..., "counters": {"user": count, ...}}
    """
    year, week = _current_iso_week()
    if request.args.get('year'):
        year = int(request.args['year'])
    if request.args.get('week'):
        week = int(request.args['week'])

    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    prefix = 'rate_limits/'
    week_suffix = f'{year}_W{week:02d}.json'

    counters = {}
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(week_suffix):
            try:
                data = json.loads(blob.download_as_text())
                counters[data.get('user', blob.name)] = data.get('count', 0)
            except Exception:
                pass

    return {
        'success': True,
        'year': year,
        'week': week,
        'counters': counters,
    }, 200


@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    """
    Manually reset a user's full_run counter for a given week.
    Body: {"user": "...", "year": 2025, "week": 12}  (year/week optional, default = current week)
    """
    data = request.get_json()
    if not data or not data.get('user'):
        return {'success': False, 'error': 'No user provided. Send JSON with "user" field.'}, 400
    safe = _safe_user(data['user'])
    year, week = _current_iso_week()
    if 'year' in data:
        year = int(data['year'])
    if 'week' in data:
        week = int(data['week'])
    _reset_full_run_count(safe, year, week)
    return {
        'success': True,
        'message': f'Counter reset to 0 for user "{data["user"]}" (week {year}-W{week:02d}).',
    }, 200

@app.route('/get_final_eval_counter', methods=['GET'])
def get_final_eval_counter():
    """
    Return the final_evaluation submission count for a given user.
    Query param: user (required)
    """
    user = request.args.get('user')
    if not user:
        return {'success': False, 'error': 'No user provided. Pass ?user=... as a query param.'}, 400
    safe = _safe_user(user)
    count = _get_final_eval_count(safe)
    return {'success': True, 'user': user, 'count': count, 'limit': FINAL_EVAL_LIMIT}, 200


@app.route('/reset_final_eval_counter', methods=['POST'])
def reset_final_eval_counter():
    """
    Reset a user's final_evaluation counter to 0.
    Body: {"user": "..."}
    """
    data = request.get_json()
    if not data or not data.get('user'):
        return {'success': False, 'error': 'No user provided. Send JSON with "user" field.'}, 400
    safe = _safe_user(data['user'])
    client = _gcs_client()
    bucket = client.bucket(RATE_LIMIT_BUCKET)
    blob = bucket.blob(f'final_evaluation_runs/{safe}.json')
    blob.upload_from_string(
        json.dumps({'user': safe, 'count': 0}),
        content_type='application/json',
    )
    return {
        'success': True,
        'message': f'Final evaluation counter reset to 0 for user "{data["user"]}".',
    }, 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
