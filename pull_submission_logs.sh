#!/bin/bash

DEST="submission_logs_from_bucket/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST"
gsutil -m cp -r gs://featurization-test-bucket/submission_logs "$DEST"