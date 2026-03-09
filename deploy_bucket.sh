#!/bin/bash

gcloud config set project gol-cdr-featurization-comp
gcloud config set run/region us-central1

# Create the bucket
gsutil mb gs://featurization-test-bucket

# Transfer data
gsutil -m cp -r data_for_bucket/* gs://featurization-test-bucket