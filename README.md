# Triton inference server with Python backend and transformers

Nvidia Triton inference server using the Python backend with HuggingFace Transformers library.

Example request using curl

```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/v2/models/ner-model/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    {
      "name": "prompt",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["My name is Wolfgang and I live in Berlin"]
    }
  ]
}'

```
