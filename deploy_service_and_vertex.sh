#!/bin/bash

# app password: iaoq hrkt zamw glhy

SCRIPT_DIR="$(dirname "$0")"

bash "$SCRIPT_DIR/deploy_service.sh"
bash "$SCRIPT_DIR/deploy_vertex.sh"
