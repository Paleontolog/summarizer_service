import nltk
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from app.request import GenerationRequest

nltk.download('punkt')


class BinaryClassifier:
    def __init__(self, **args):
        self._num_labels = 2
        self._max_length = 128

        self._device = args.get("DEVICE", "cpu")
        self._model_name = args.get("MODEL_NAME")
        self._batch_size = int(args.get("BATCH_SIZE", 32))

        self.tokenizer = BertTokenizer.from_pretrained(self._model_name)
        model = BertForSequenceClassification.from_pretrained(self._model_name,
                                                              num_labels=self._num_labels)
        model = model.to(self._device)
        self.model = model.eval()

    def predict(self, request: GenerationRequest):
        text_sentences = nltk.sent_tokenize(request.input_text)

        n = self._batch_size
        text_sentences = [text_sentences[i: i + n] for i in range(0, len(text_sentences), n)]

        result_text = []
        for sentences in text_sentences:
            input_ids = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt",
                                       max_length=self._max_length)

            input_ids = {
                "input_ids": input_ids["input_ids"].to(self._device),
                "token_type_ids": input_ids["token_type_ids"].to(self._device),
                "attention_mask": input_ids["attention_mask"].to(self._device)
            }

            with torch.no_grad():
                predictions = self.model(**input_ids)

            indices = torch.max(predictions[0], dim=1).indices

            result_text += [text for ind, text in zip(indices, sentences) if ind == 1]

        return " ".join(result_text)
