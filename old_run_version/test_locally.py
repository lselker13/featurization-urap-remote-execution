#!/usr/bin/env python3
import json
import requests
import sys

# Read the Python file
with open('test_featurizer.py', 'r') as f:
    code = f.read()

# Send to your endpoint
response = requests.post(
    'http://localhost:8080/execute',
    json={'code': code}
)

print(response)
print(response.json())