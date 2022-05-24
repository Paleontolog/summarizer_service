import json

import nltk
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import BertTokenizer


def split_data_positive_negative(dataset):
    negative, positive = [], []
    for sample in dataset:
        abstract = nltk.sent_tokenize(sample["abstract"])
        abstract_sentences = sample["abstract_sentences"].items()
        abstract_sentences = sorted(abstract_sentences, key=lambda x: x[1])
        if len(abstract_sentences) > 6 * len(abstract):
            positive.extend(abstract_sentences[:(3 * len(abstract))])
            negative.extend(abstract_sentences[-(3 * len(abstract)):])
        else:
            mid = len(abstract_sentences) // 2
            positive.extend(abstract_sentences[:mid])
            negative.extend(abstract_sentences[mid:])
    return positive, negative


with open("/train_data.json", "r", encoding="utf-8") as r:
    train_dataset = [json.loads(i) for i in r.readlines()]

with open("/eval_data.json", "r", encoding="utf-8") as r:
    train_dataset += [json.loads(i) for i in r.readlines()]

positive_samples, negative_samples = split_data_positive_negative(train_dataset)
positive_samples = [sample for sample, perplexity in positive_samples]
negative_samples = [sample for sample, perplexity in negative_samples]

positive_labels, negative_labels = [1] * len(positive_samples), [0] * len(negative_samples)

train_dataset = positive_samples + negative_samples
train_labels = positive_labels + negative_labels

tokenizer = BertTokenizer.from_pretrained(args.model_name)

train_dataset, test_dataset, train_labels, test_labels = train_test_split(train_dataset, train_labels, test_size=0.33,
                                                                          random_state=42)

train_encodings = tokenizer(train_dataset, truncation=True, padding=True, max_length=128, return_tensors="pt")
test_encodings = tokenizer(test_dataset, truncation=True, padding=True, max_length=128, return_tensors="pt")

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)


class BinaryClassificationDataset(Dataset):

    def __init__(self, args, data, labels):
        self.data = data
        self.args = args
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx].to(self.args.device),
            'token_type_ids': self.data['token_type_ids'][idx].to(self.args.device),
            'attention_mask': self.data['attention_mask'][idx].to(self.args.device),
            'labels': self.labels[idx].to(self.args.device)
        }

