#!/bin/bash

# app password: iaoq hrkt zamw glhy 

gcloud config set project gol-cdr-featurization-comp
gcloud config set run/region us-central1

gcloud run deploy featurization-test-server \
  --source server \
  --platform managed \
  --region us-central1 \
  --no-allow-unauthenticated \
  --execution-environment gen2 \
  --memory 4Gi \
  --add-volume name=data,type=cloud-storage,bucket=featurization-test-bucket \
  --add-volume-mount volume=data,mount-path=/data \
  --cpu 1 \
  --timeout 3h \
  --max-instances 10 \
  --set-env-vars GMAIL_APP_PASSWORD="iaoq hrkt zamw glhy"

gcloud run services set-iam-policy featurization-test-server \
  --region us-central1 \
  --quiet \
  "$(dirname "$0")/iam_policy.yaml"