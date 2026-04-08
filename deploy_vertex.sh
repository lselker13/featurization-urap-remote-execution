#!/bin/bash

# Builds and pushes the Vertex AI job image.
# The server triggers actual runs via the Vertex AI API (see server/app.py _trigger_job).
set -e

PROJECT="gol-cdr-featurization-comp"
REGION="us-central1"
REPO="featurization-jobs"
IMAGE_NAME="featurization-evaluator-vertex"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:latest"

gcloud config set project "$PROJECT"

# Ensure Artifact Registry repo exists
gcloud artifacts repositories describe "$REPO" \
  --location="$REGION" --project="$PROJECT" &>/dev/null \
  || gcloud artifacts repositories create "$REPO" \
       --repository-format=docker \
       --location="$REGION" \
       --project="$PROJECT"

CLOUDBUILD_CONFIG=$(mktemp /tmp/cloudbuild-XXXXXX.yaml)
cat > "$CLOUDBUILD_CONFIG" <<YAML
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - build
      - -t
      - ${IMAGE_URI}
      - .
images:
  - ${IMAGE_URI}
YAML

echo "Uploading service account key to GCS..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud iam service-accounts keys create /tmp/sa-key.json --iam-account="$SA"
gsutil cp /tmp/sa-key.json gs://featurization-test-bucket/sa-key.json
rm /tmp/sa-key.json

echo "Building and pushing image..."
gcloud builds submit job_vertex/ \
  --config="$CLOUDBUILD_CONFIG" \
  --project="$PROJECT"
rm "$CLOUDBUILD_CONFIG"

echo "Image pushed: ${IMAGE_URI}"
echo "Update IMAGE_URI in server/app.py and redeploy the server to start using this image."
