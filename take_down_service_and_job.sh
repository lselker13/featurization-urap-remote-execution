#!/bin/bash

gcloud run services delete featurization-test-server --region us-central1
gcloud run jobs delete featurization-evaluator-job --region us-central1
