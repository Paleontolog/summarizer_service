from abc import ABC
from typing import Tuple, List

import nltk
import numpy as np
import requests
import tritonclient.http as httpclient

from client_type import ClientType
from request import GenerationRequest, Response, SummarizationRequest


class Client(ABC):
    def __init__(self, host: str, port: str):
        self._host = host
        self._port = port
        self._base_url = f"http://{self._host}:{self._port}"

    @staticmethod
    def create(client_type: ClientType, host: str, port: str, **kwargs):
        if client_type == ClientType.BASE:
            return BaseHttpClient(host, port)
        elif client_type == ClientType.TRITON:
            return TritonClient(host, port)


class BaseHttpClient(Client):
    def __init__(self, host: str, port: str, **kwargs):
        super().__init__(host, port)

        self._method = kwargs.get("method", "generate")
        self._url = f'{self._base_url}/{self._method}'

    def _generate_http(self, request: GenerationRequest) -> Response:
        try:
            result = requests.post(self._url, data=request.to_json())
            if result.ok:
                return Response.from_json(result.text)
            else:
                result.raise_for_status()
        except Exception as e:
            raise Exception(f"Incorrect method call {self._url}", e)

    def process(self, request: GenerationRequest):
        return self._generate_http(request)


class TritonClient(Client):

    def __init__(self, host: str, port: str, **kwargs):
        super().__init__(host, port)
        self._triton_client = httpclient.InferenceServerClient(url=self._base_url)
        self._model_version = kwargs.get("model_version", "1")
        self._model_name = kwargs.get("model_name", "transformers")

    def _get_sample_tokenized_text_binary(self, result_text: str) \
            -> Tuple[List[httpclient.InferInput], List[httpclient.InferRequestedOutput]]:

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput(name="TEXT", shape=[1, ], datatype="BYTES"))

        inputs[0].set_data_from_numpy(np.asarray([result_text], dtype=object))

        outputs.append(httpclient.InferRequestedOutput("logits", binary_data=False))

        return inputs, outputs

    def _generate_triton(self, request: GenerationRequest) -> Response:
        try:
            tokenized_text = nltk.sent_tokenize(request.input_text)
            result_text = "<sep>".join(tokenized_text)

            inputs, outputs = self._get_sample_tokenized_text_binary(result_text)

            response = self._triton_client.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=inputs,
                outputs=outputs
            )

            classes = np.argmax(response.as_numpy("logits"), axis=1)

            result_text = [text for ind, text in zip(classes, tokenized_text) if ind == 1]

            return Response(" ".join(result_text))
        except Exception as e:
            raise Exception("Incorrect method call", e)

    def _summarize_triton(self, request: SummarizationRequest) -> Response:
        raise NotImplemented()

    def process(self, request: GenerationRequest) -> Response:
        if request is GenerationRequest:
            return self._generate_triton(request)
        elif request is SummarizationRequest:
            return self._summarize_triton(request)
