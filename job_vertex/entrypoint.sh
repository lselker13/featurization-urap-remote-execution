#!/bin/bash
set -e

echo "Fetching service account key from GCS..."
python - <<'EOF'
from google.cloud import storage
storage.Client().bucket("featurization-test-bucket").blob("sa-key.json").download_to_filename("/tmp/sa-key.json")
EOF
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa-key.json

# SUBMISSION_JSON_PATH is /data/workspace/run_specs/{base}.json (GCS mount convention).
# Strip the /data/ prefix to get the GCS blob name, then download to a local path.
GCS_SPEC_BLOB="${SUBMISSION_JSON_PATH#/data/}"
export GCS_SPEC_BLOB

echo "Downloading run spec from GCS..."
python - <<'EOF'
from google.cloud import storage
import os
storage.Client().bucket("featurization-test-bucket").blob(os.environ['GCS_SPEC_BLOB']).download_to_filename("/tmp/spec.json")
EOF
export SUBMISSION_JSON_PATH=/tmp/spec.json

echo "Syncing togo data from GCS..."
mkdir -p /data/input_data/togo
python - <<'EOF'
from google.cloud import storage
import os

client = storage.Client()
bucket = client.bucket("featurization-test-bucket")

for blob in bucket.list_blobs(prefix="input_data/"):
    if blob.name.endswith("/"):
        continue
    dest = os.path.join("/data", blob.name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    blob.download_to_filename(dest)
    print(f"Downloaded {blob.name}")
EOF
export DATA_DIR=/data/input_data/togo
echo "Data ready."

python job.py

echo "Deleting run spec from GCS..."
python - <<'EOF'
from google.cloud import storage
import os
blob_name = os.environ['GCS_SPEC_BLOB']
storage.Client().bucket("featurization-test-bucket").blob(blob_name).delete()
print(f"Deleted spec: {blob_name}")
EOF
