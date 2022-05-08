import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration

from app.request import SummarizationRequest


class BartSummarizer:

    def __init__(self, **args):
        self._device = args.get("DEVICE", "cpu")
        self._model_name = args.get("MODEL_NAME")
        self._max_source_tokens_count = int(args.get("MAX_CONTEXT_WINDOW", 600))

        self.tokenizer = MBartTokenizer.from_pretrained(self._model_name, src_lang="ru_RU")
        model = MBartForConditionalGeneration.from_pretrained(self._model_name)
        model = model.to(self._device)
        self.model = model.eval()

    def predict(self, summarization_request: SummarizationRequest):
        input_ids = self.tokenizer(
            summarization_request.input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._max_source_tokens_count
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids["input_ids"].to(self._device),
                attention_mask=input_ids["attention_mask"].to(self._device),
                max_length=summarization_request.max_length,
                no_repeat_ngram_size=summarization_request.no_repeat_ngram_size,
                num_beams=summarization_request.num_beams,
                repetition_penalty=summarization_request.repetition_penalty,
                temperature=summarization_request.temperature,
                top_k=summarization_request.top_k,
                top_p=summarization_request.top_p
            )

        summaries = self.tokenizer.batch_decode(output_ids,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        return summaries[0]
