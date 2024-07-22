#!/usr/bin/env bash

# Starter from https://stackoverflow.com/questions/40216311/reading-in-environment-variables-from-an-environment-file
set -a
source .env
set +a

python3 instruction-synth-instructlab.py