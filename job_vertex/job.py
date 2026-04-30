"""
Cloud Run Job entry point.

Reads the submission JSON written by the service and delegates to run_job()
in job_logic. To unit test, import run_job from job_logic and call it directly
with a code string.
"""
import json
import os
import sys

from job_logic import run_job

DATA_DIR = os.environ.get('DATA_DIR', '/data')


def main():
    json_path = os.environ.get('SUBMISSION_JSON_PATH')
    if not json_path:
        print('ERROR: SUBMISSION_JSON_PATH environment variable is not set', file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        payload = json.load(f)
    print(
        f"Running job: user={payload['user']} full_run={payload['full_run']}"
        f" use_holdout={payload.get('use_holdout', False)}"
        f" toy_param_grids={payload.get('toy_param_grids', False)}"
        f" final_evaluation={payload.get('final_evaluation', False)}"
    )
    result = run_job(
        payload['code'],
        payload['user'],
        DATA_DIR,
        payload['full_run'],
        payload.get('use_holdout', False),
        payload.get('toy_param_grids', False),
        payload.get('final_evaluation', False),
    )

    if not result.get('success'):
        sys.exit(1)


if __name__ == '__main__':
    main()
