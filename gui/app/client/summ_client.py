from typing import Optional

from app.client.request import GenerationRequest, Response

import requests


class Client:
    def __init__(self, url: str, method='api/generate'):
        self._method = method
        self._url = url + method

    def generate(self, request: GenerationRequest) -> Response:
        try:
            result = requests.post(self._url, data=request.to_json())
            if result.ok:
                return Response.from_json(result.text)
            else:
                result.raise_for_status()
        except Exception as e:
            raise Exception(f"Incorrect method call {self._url}", e)
