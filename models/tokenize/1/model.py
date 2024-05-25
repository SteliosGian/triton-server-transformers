import logging
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, TensorType

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')


class TritonPythonModel:
    def initialize(self, _):
        MODEL_ID = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Decode the Byte Tensor into Text
            query = pb_utils.get_input_tensor_by_name(request, "text")
            query = query.as_numpy().astype(str).tolist()

            tokens = self.tokenizer(text=query, return_token_type_ids=True, return_tensors=TensorType.NUMPY)

            input_ids = pb_utils.Tensor("input_ids", tokens["input_ids"])
            attention = pb_utils.Tensor("attention", tokens["attention_mask"])
            token_type_ids = pb_utils.Tensor("token_type_ids", tokens["token_type_ids"])

            inference_response = pb_utils.InferenceResponse(output_tensors=[input_ids, attention, token_type_ids])
            responses.append(inference_response)

        return responses

    def finalize(self, args=None):
        self.generator = None
