#!/bin/bash

# Bucket layout:
#   input_data/togo/     - input data for the job (uploaded here; mounted at /data/input_data/togo/)
#   submission_logs/     - human-readable .txt logs (written by service at runtime)
#   workspace/run_specs/ - JSON run specs (written by service, deleted by job after completion)
#   rate_limits/         - GCS-backed rate limit counters (written by service at runtime)

gcloud config set project gol-cdr-featurization-comp
gcloud config set run/region us-central1

# Create the bucket
gcloud storage buckets create gs://featurization-test-bucket

# Transfer data
gcloud storage cp --recursive data_for_bucket/* gs://featurization-test-bucket/data/