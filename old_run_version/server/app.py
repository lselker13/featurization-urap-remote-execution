from flask import Flask, request
import os

from app_logic import run_submission

app = Flask(__name__)

DATA_DIR = os.environ.get('DATA_DIR', '/data/togo')
LOG_DIR = os.environ.get('LOG_DIR', '/data/submission_logs')


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
        return run_submission(data['code'], data['user'], DATA_DIR, LOG_DIR)
    except Exception as e:
        return {'success': False, 'error': f'Failed to log submission: {e}'}, 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
