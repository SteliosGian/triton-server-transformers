#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'
trap "exit 1" HUP INT QUIT ABRT SEGV

docker build -t ner-model:local .

docker run -it --rm -p8080:8080 -p8081:8001 -p8002:8002 ner-model:local tritonserver --model-repository=./models --http-port=8080
