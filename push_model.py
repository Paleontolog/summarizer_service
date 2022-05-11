from huggingface_hub import login
from transformers import BertForSequenceClassification, BertTokenizerFast, MBartTokenizer, MBartForConditionalGeneration

access_token = ""
name = ""
password = ""
login(name, password)


model_name = "bert_model"

tokenizer = BertTokenizerFast.from_pretrained(model_name)
model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer.push_to_hub("bert_sentence_classifier", use_auth_token=access_token)
model.push_to_hub("bert_sentence_classifier", use_auth_token=access_token)


model_name = "bart_model"

tokenizer = MBartTokenizer.from_pretrained(model_name, src_lang="ru_RU")
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer.push_to_hub("bart_rus_summarizer", use_auth_token=access_token)
model.push_to_hub("bart_rus_summarizer", use_auth_token=access_token)