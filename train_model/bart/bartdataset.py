from h5py import Dataset


class MBartSummarizationDataset(Dataset):
    def __init__(
            self,
            input_file,
            tokenizer,
            max_source_tokens_count,
            max_target_tokens_count,
            src_lang="ru_RU",
            tgt_lang="ru_RU"
    ):
        self.pairs = []
        for sample in input_file:
            self.pairs.append((sample["text"], sample["summary"]))
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        model_inputs = self.tokenizer(
            source, return_tensors="pt",
            max_length=self.max_source_tokens_count, padding="max_length", truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target, max_length=self.max_target_tokens_count,
                                    return_tensors="pt", padding="max_length", truncation=True)

        labels["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return {
            "input_ids": model_inputs["input_ids"][0],
            "attention_mask": model_inputs["attention_mask"][0],
            "labels": model_inputs["labels"][0]
        }
