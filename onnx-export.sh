#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'
trap "exit 1" HUP INT QUIT ABRT SEGV

optimum-cli export onnx --model dslim/bert-base-NER models/ner-model-onnx/1/ --task token-classification
