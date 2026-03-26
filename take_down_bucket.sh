#!/bin/bash

# pull the logs; only take the bucket down if that worked
./pull_submission_logs.sh && gcloud storage rm --recursive gs://featurization-test-bucket