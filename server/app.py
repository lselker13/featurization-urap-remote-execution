import datetime
import json
import os
import traceback
import uuid
from zoneinfo import ZoneInfo

import google.auth
import google.auth.transport.requests
from flask import Flask, request

PACIFIC = ZoneInfo('America/Los_Angeles')
ALLOW_FULL_RUN = False

LOG_DIR = os.environ.get('LOG_DIR', '/data/submission_logs')

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Submission logging
# ---------------------------------------------------------------------------

def log_submission(user_code, user, log_dir, full_run):
    """Write a human-readable log and a JSON dispatch file. Returns the JSON path."""
    print('logging user code')
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now(PACIFIC)
    timestamp_display = now.strftime('%Y-%m-%d %H:%M:%S %Z')
    filename_ts = now.strftime('%Y-%m-%dT%H:%M:%S')
    safe_user = ''.join(c if c.isalnum() or c in '.@-' else '_' for c in (user or 'anonymous'))
    submission_id = uuid.uuid4().hex[:8]
    base = f"{filename_ts}_{safe_user}_{submission_id}"

    # Human-readable log (same format as original)
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

    # JSON dispatch file for the job to read
    json_path = os.path.join(log_dir, f"{base}.json")
    with open(json_path, 'w') as f:
        json.dump({'user': user, 'code': user_code, 'full_run': full_run}, f)

    return json_path


# ---------------------------------------------------------------------------
# Job trigger
# ---------------------------------------------------------------------------

def _trigger_job(json_file_path):
    """Trigger the Cloud Run Job execution, passing the submission JSON path via env var."""
    job_name = os.environ['JOB_NAME']
    region = os.environ.get('JOB_REGION', 'us-central1')

    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    gcp_project = os.environ.get('GCP_PROJECT', project)

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
    resp = authed_session.post(url, json=body, timeout=30)
    resp.raise_for_status()
    print(f'Job triggered: {resp.json().get("name")}')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_submission(user_code, user, log_dir, full_run):
    """
    Top-level entry point called from app.py.
    Logs submission, validates code synchronously, triggers the Cloud Run Job,
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

    # for now
    if full_run and (not ALLOW_FULL_RUN):
        message += ' Warning: full_run not yet permitted; overriding to False.'
        full_run = False
    json_file_path = log_submission(user_code, user, log_dir, full_run)

    _trigger_job(json_file_path)

    return {'success': True, 'message': message.strip()}, 202


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
    try:
        return run_submission(data['code'], data['user'], LOG_DIR, data.get('full_run', False))
    except Exception as e:
        return {'success': False, 'error': f'Failed to log submission: {e}'}, 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
