import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification


class TritonPythonModel:
    def initialize(self, args):
        MODEL_ID = "dslim/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

        self.generator = pipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            # Decode the Byte Tensor into Text
            input = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_string = input.as_numpy().astype(str).tolist()

            # Call the Model pipeline
            pipeline_output = self.generator(input_string)
            output = json.dumps(str(pipeline_output))
            # Encode the text to byte tensor to send back
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_text",
                        np.array([output.encode()]),
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self, args=None):
        self.generator = None
