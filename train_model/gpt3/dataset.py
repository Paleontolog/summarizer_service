import torch
from torch.utils.data import Dataset

from gpt3.utils import add_special_tokens
from data_utils import *

tokenizer = add_special_tokens()

class GPT21024Dataset(Dataset):

    def __init__(self, records, max_len=2048):
        self.data = records
        self.max_len = max_len
        self.tokenizer = add_special_tokens()
        self.sep_token = self.tokenizer.encode(self.tokenizer.sep_token)
        self.pad_token = self.tokenizer.encode(self.tokenizer.pad_token)

    def __len__(self):
        return len(self.data)

    def _truncate(self, article, abstract, addit_symbols_count = 1):
        len_abstract = len(abstract)
        max_len_article = self.max_len - len_abstract - addit_symbols_count
        return article[:max_len_article], abstract

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = self.pad_token * self.max_len
        article, abstract = self._truncate(sample['text'], sample['summary'])
        content = article + self.sep_token + abstract
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'article': text, 'sum_idx': len(article)}
        return sample


if __name__ == "__main__":
    train_records = read_gazeta_records("gazeta_train.jsonl")
    val_records = read_gazeta_records("gazeta_val.jsonl")
    test_records = read_gazeta_records("gazeta_test.jsonl")

    train_records = tokenize_gazeta(train_records)
    val_records = tokenize_gazeta(val_records)
    test_records = tokenize_gazeta(test_records)
