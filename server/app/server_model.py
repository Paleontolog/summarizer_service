import os
#
from app.model_type import ModelType
from app.request import GenerationRequest, SummarizationRequest
from app.models.bart_summarizer import BartSummarizer
from app.models.binary_classifier import BinaryClassifier


class ServerModel:
    def __init__(self):
        args = os.environ
        self.model_type = ModelType[args.get('BACKEND_TYPE')]

        if self.model_type == ModelType.BART:
            self._model = BartSummarizer(**args)
        elif self.model_type == ModelType.BINARY_CLASSIFIER:
            self._model = BinaryClassifier(**args)

    def process(self, request) -> str:

        if self.model_type == ModelType.BART:
            request = SummarizationRequest.schema().loads(request)
        elif self.model_type == ModelType.BINARY_CLASSIFIER:
            request = GenerationRequest.schema().loads(request)

        return self._model.predict(request)
