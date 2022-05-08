import logging
from typing import Optional, Any

from request import GenerationRequest, Response

import requests


class Client:
    def __init__(self, host: str, port: str, method='generate'):
        self._host = host
        self._port = port
        self._method = method
        self._url = f'http://{self._host}:{self._port}/{self._method}'

    def generate(self, request: GenerationRequest) -> Optional[Response]:
        try:
            result = requests.post(self._url, data=request.to_json())
            if result.ok:
                return Response.from_json(result.text)
            else:
                result.raise_for_status()
        except Exception as e:
            raise Exception("Incorrect method call", e)
