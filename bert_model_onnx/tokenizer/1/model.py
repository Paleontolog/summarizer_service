import os
import triton_python_backend_utils as pb_utils

from transformers import BertTokenizerFast


class TritonPythonModel:
    def initialize(self, _):
        self._model_name = os.environ.get("TOKENIZER", "bert-base-multilingual-cased")
        self._tokenizer = BertTokenizerFast.from_pretrained(self._model_name)
        self._separator = os.environ.get("SEPARATOR", "<sep>")
        self._max_length = int(os.environ.get("MAX_SENT_LENGTH", 128))

    def execute(self, requests):
        responses = []
        for request in requests:
            text_sentences = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()
            text_sentences = text_sentences[0].decode("UTF-8")
            text_sentences = text_sentences.split(self._separator)

            tokens = self._tokenizer(text_sentences,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="np",
                                     max_length=self._max_length)

            input_ids = pb_utils.Tensor("input_ids", tokens["input_ids"])
            token_type_ids = pb_utils.Tensor("token_type_ids", tokens["token_type_ids"])
            attention_mask = pb_utils.Tensor("attention_mask", tokens["attention_mask"])
            inference_response = pb_utils.InferenceResponse(output_tensors=[input_ids, token_type_ids, attention_mask])
            responses.append(inference_response)

        return responses
