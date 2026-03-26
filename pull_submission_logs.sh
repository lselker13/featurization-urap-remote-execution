#!/bin/bash

DEST="submission_logs_from_bucket/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST"
gcloud storage cp --recursive gs://featurization-test-bucket/submission_logs "$DEST"