#!/bin/bash

gcloud run services delete featurization-test-server --region us-central1
gcloud artifacts docker images delete us-central1-docker.pkg.dev/gol-cdr-featurization-comp/featurization-jobs/featurization-evaluator-vertex --quiet || true
