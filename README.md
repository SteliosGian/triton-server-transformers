# Triton inference server with Python backend and transformers

Nvidia Triton inference server using the Python backend with HuggingFace Transformers library.

Example request using curl:

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

For the Triton ensemble model (transformers), transform the model into an ONNX format. For that, use the ```onnx-export.sh``` script that uses the optimum library.

The following libraries need to be installed via pip.
- [optimum[exporters]](https://pypi.org/project/optimum/)
- [accelerate](https://pypi.org/project/accelerate/)

Then run ```bash onnx-export.sh```.

Example request of the transformers ensemble:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/v2/models/transformers/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    {
      "name": "text",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["My name is Wolfgang and I live in Berlin"]
    }
  ]
}'
```
