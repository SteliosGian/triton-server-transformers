import logging
import json
import os
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')


class TritonPythonModel:
    def initialize(self, _):
        CONFIG_PATH = "/".join(os.path.abspath(__file__).split('/')[:-1])
        MODEL_ID = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        with open(f"{CONFIG_PATH}/config.json") as json_file:
            self.config = json.load(json_file)

    def execute(self, requests):
        responses = []

        for request in requests:

            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()[0]

            predicted_labels_classes = np.argmax(logits, axis=-1)[0]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist(), skip_special_tokens=True)

            labels = np.vectorize(self.config["id2label"].get)(predicted_labels_classes.astype(str))

            output = []
            for token, label in zip(tokens, labels[1:-1]):
                output.append({"token": token, "label": label})

            output = np.array(output, dtype=object)

            output_text = pb_utils.Tensor("output_text", output)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_text
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self, args=None):
        self.tokenizer = None
