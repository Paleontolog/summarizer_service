# import time
#
import nltk as nltk
import onnxruntime.transformers.convert_beam_search
import onnxruntime.transformers.optimizer as opt
import psutil
import torch
from onnxruntime.transformers.fusion_options import FusionOptions
from transformers import BertForSequenceClassification, BertTokenizerFast

# nltk.download('punkt')
#
# data = r"C:\Users\lezgy\OneDrive\Рабочий стол\Data_summ\data.txt"
# # r"/mnt/c/Users/lezgy/OneDrive/Рабочий стол/Data_summ/data.txt"
#
# with open(data, "r", encoding="UTF-8") as r:
#     text = r.read()
#
# device = 'cpu'
model_name = "bert_model"
# export_model_path = r"server_triton/bert/1/model.onnx"
# export_opt_model_path = r"server_triton/bert_opt/1/model.onnx"
# max_length = 128
# enable_overwrite = True
# opset_version = 11
#
# # "bert-base-multilingual-cased"
# tokenizer = BertTokenizerFast.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
print(model.num_parameters())
# model = model.to(device)
# model = model.eval()

# text_sentences = nltk.sent_tokenize(text)
#
# input_ids = tokenizer(text_sentences,
#                       padding=True,
#                       truncation=True,
#                       return_tensors="pt",
#                       max_length=max_length)
#
# input_ids = {
#     "input_ids": input_ids["input_ids"].to(device),
#     "token_type_ids": input_ids["token_type_ids"].to(device),
#     "attention_mask": input_ids["attention_mask"].to(device)
# }
#
# start = time.time()
# with torch.no_grad():
#     predictions = model(**input_ids)
#     print(predictions)
# print(time.time() - start)
#
# with torch.no_grad():
#     symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
#     torch.onnx.export(model,  # model being run
#                       args=tuple(input_ids.values()),  # model input (or a tuple for multiple inputs)
#                       f=export_model_path,  # where to save the model (can be a file or file-like object)
#                       opset_version=opset_version,  # the ONNX version to export the model to
#                       do_constant_folding=False,  # whether to execute constant folding for optimization
#                       input_names=['input_ids',  # the model's input names
#                                    'token_type_ids',
#                                    'attention_mask'],
#                       output_names=['output'],  # the model's output names
#                       dynamic_axes={'input_ids': symbolic_names,  # variable length axes
#                                     'token_type_ids': symbolic_names,
#                                     'attention_mask': symbolic_names,
#                                     'output': {0: 'batch_size'}})
#     print("Model exported at ", export_model_path)
#
# options = FusionOptions("bert")
# options.enable_gelu_approximation = True
# opt_model = opt.optimize_model(
#     export_model_path,
#     use_gpu=True,
#     model_type="bert",
#     opt_level=99,
#     num_heads=0,
#     hidden_size=0,
#     optimization_options=options
# )
# opt_model.convert_float_to_float16()
# opt_model.save_model_to_file(export_opt_model_path)
#
# input_ids = {
#     "input_ids": input_ids["input_ids"].cpu().numpy(),
#     "token_type_ids": input_ids["token_type_ids"].cpu().numpy(),
#     "attention_mask": input_ids["attention_mask"].cpu().numpy()
# }
#
# sess_options = onnxruntime.SessionOptions()
#
# sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
#
# sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#
# session = onnxruntime.InferenceSession(export_model_path, sess_options,
#                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#
# start = time.time()
# predictions = session.run(None, input_ids)
# print(time.time() - start)
# print(predictions)
import enum
import time

import nltk as nltk
import onnxruntime.transformers.convert_beam_search
import onnxruntime.transformers.optimizer as opt
import psutil
import torch
from onnxruntime.transformers.fusion_options import FusionOptions
from transformers import BertForSequenceClassification, BertTokenizerFast, MBartTokenizer, MBartForConditionalGeneration
#
# opt_model = opt.optimize_model(
#     input=export_model_path,
#     use_gpu=True,
#     model_type="bert",
#     opt_level=99,
#     num_heads=0,
#     hidden_size=0,
#     optimization_options=options
# )
# nltk.download('punkt')
#
# data = r"C:\Users\lezgy\OneDrive\Рабочий стол\Data_summ\data.txt"
#
# with open(data, "r", encoding="UTF-8") as r:
#     text = r.read()
#
# device = 'cuda'
# model_name = "bert_model"
# export_model_path = r"server_triton/bert/1/model.onnx"
# export_opt_model_path = r"server_triton/bert_opt/1/model.onnx"
# max_length = 128
# enable_overwrite = True
# opset_version = 11
#
#
# import onnxruntime as rt
#
#
# tokenizer = BertTokenizerFast.from_pretrained(model_name)
#
# text_sentences = nltk.sent_tokenize(text)
#
# input_ids = tokenizer(text_sentences,
#                       padding=True,
#                       truncation=True,
#                       return_tensors="pt",
#                       max_length=max_length)
#
#
# sess_options = rt.SessionOptions()
#
# sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
#
# sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
#
# session = rt.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])
#
#
#
# input_ids = {
#     "input_ids": input_ids["input_ids"].cpu().numpy(),
#     "token_type_ids": input_ids["token_type_ids"].cpu().numpy(),
#     "attention_mask": input_ids["attention_mask"].cpu().numpy()
# }
#
# result = 0
# with torch.no_grad():
#     for i in range(20):
#         start = time.time()
#         session.run(None, input_ids)
#         end = time.time() - start
#         print(end)
#         result += end
#
# print(result / 20)
