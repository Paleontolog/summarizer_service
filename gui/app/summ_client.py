from app.request import GenerationRequest


class Client:
    def __int__(self, host, port, mode):
        self._host = host
        self._port = port
        self._mode = mode

    def generate(self, request: GenerationRequest) -> str:
        pass
