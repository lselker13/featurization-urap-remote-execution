#!/usr/bin/env python3
import subprocess
import requests


SERVICE_URL = 'https://featurization-test-server-uvs6s73ula-uc.a.run.app'


# Get the active gcloud account as the user identity sent to the sheet
user_result = subprocess.run(
    ['gcloud', 'config', 'get-value', 'account'],
    capture_output=True, text=True, check=True
)
USER_EMAIL = user_result.stdout.strip()

# Fetch an OIDC token via gcloud; Cloud Run requires this when --no-allow-unauthenticated is set.
# fetch_id_token() doesn't work with user credentials, but gcloud does.
token_result = subprocess.run(
    ['gcloud', 'auth', 'print-identity-token'],
    capture_output=True, text=True, check=True
)
id_token = token_result.stdout.strip()

# Read the featurizer
with open('test_featurizer.py', 'r') as f:
    code = f.read()

# Send authenticated request; include "user" so the server can log it to the sheet
response = requests.post(
    f"{SERVICE_URL}/execute",
    headers={"Authorization": f"Bearer {id_token}"},
    json={"code": code, "user": USER_EMAIL},
)
print(f"SERVICE_URL: '{SERVICE_URL}'")
print(f"User: '{USER_EMAIL}'")
print(f"Status code: {response.status_code}")
print(response)
print(response.json())