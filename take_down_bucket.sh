#!/bin/bash

# pull the logs; only take the bucket down if that worked
./pull_submission_logs.sh && gsutil rm -r gs://featurization-test-bucket