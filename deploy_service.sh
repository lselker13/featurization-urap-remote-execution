#!/bin/bash

PROJECT="gol-cdr-featurization-comp"
REGION="us-central1"
JOB_NAME="featurization-evaluator-job"
SERVICE_NAME="featurization-test-server"

gcloud config set project "$PROJECT"
gcloud config set run/region "$REGION"

# Deploy the Cloud Run Service (thin HTTP dispatcher: validates code, triggers the job)
gcloud run deploy "$SERVICE_NAME" \
  --source server \
  --platform managed \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --execution-environment gen2 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60s \
  --max-instances 10 \
  --add-volume name=data,type=cloud-storage,bucket=featurization-test-bucket \
  --add-volume-mount volume=data,mount-path=/data \
  --set-env-vars "JOB_NAME=${JOB_NAME},JOB_REGION=${REGION},GCP_PROJECT=${PROJECT},GMAIL_APP_PASSWORD=iaoq hrkt zamw glhy"

gcloud run services set-iam-policy "$SERVICE_NAME" \
  --region "$REGION" \
  --quiet \
  "$(dirname "$0")/iam_policy.yaml"
