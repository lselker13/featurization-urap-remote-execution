#!/bin/bash

PROJECT="gol-cdr-featurization-comp"
REGION="us-central1"
JOB_NAME="featurization-evaluator-job"

gcloud config set project "$PROJECT"
gcloud config set run/region "$REGION"

# Deploy the Cloud Run Job (runs the long evaluation; full CPU, no HTTP lifecycle throttling)
gcloud run jobs deploy "$JOB_NAME" \
  --source job \
  --region "$REGION" \
  --execution-environment gen2 \
  --memory 16Gi \
  --cpu 8 \
  --task-timeout 4h \
  --max-retries 0 \
  --add-volume name=data,type=cloud-storage,bucket=featurization-test-bucket \
  --add-volume-mount volume=data,mount-path=/data \
  --set-env-vars GMAIL_APP_PASSWORD="iaoq hrkt zamw glhy"

# Grant the default Compute service account permission to trigger the job.
# Cloud Run services run as PROJECT_NUMBER-compute@developer.gserviceaccount.com by default.
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/run.developer"
