#
import torch.nn as nn

# class Net(nn.Module):
#     def __init__(self, model):
#         super(Net, self).__init__()
#         self.model = model
#
#     def forward(self, data):
#         result = self.model(**data)
#         return result.logits
#

with open(r"C:\Users\lezgy\OneDrive\Рабочий стол\Data_summ\data.txt", "r", encoding="UTF-8") as r:
    text = r.read()

#
#
# bart_model = "bart_model"
from transformers import MBartTokenizerFast, MBartTokenizer

device = "cuda"
#
bart_model = "bart_model"
tokenizer = MBartTokenizer.from_pretrained(bart_model, src_lang="ru_RU")


# model = model.eval()
#
# model_new = Net(model)
# model_new = model_new.eval()
input_ids = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=600
)

# input_ids_new = {"input_ids": input_ids["input_ids"].to(device),
#                  "attention_mask": input_ids["attention_mask"].to(device)}
#
# # with torch.no_grad():
# #     start = time.time()
# #     print(model(**input_ids))
# #     print(time.time() - start)
#
# with torch.no_grad():
#     traced_mlm_model = torch.jit.trace(model_new, {**input_ids})
#     # frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model_new.eval()))
#     #
#     # start = time.time()
#     # frozen_mod({**input_ids_new})
#     # print(time.time() - start)
#
#     # torch.onnx.export(model,  # model being run
#     #                   [input_ids["input_ids"], input_ids["attention_mask"]],  # model input (or a tuple for multiple inputs)
#     #                   "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
#     #                   export_params=True,  # store the trained parameter weights inside the model file
#     #                   opset_version=10,  # the ONNX version to export the model to
#     #                   do_constant_folding=True,  # whether to execute constant folding for optimization
#     #                   input_names=['input_ids', 'attention_mask'],  # the model's input names
#     #                   output_names=['output'],  # the model's output names
#     #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#     #                                 'output': {0: 'batch_size'}})

# from transformers import AutoTokenizer, pipeline
# from optimum.onnxruntime import ORTSeq2SeqTrainer
import onnxruntime

sess = onnxruntime.InferenceSession(r'D:\PycharmProjects\Summarizer_Library\server\bart\onnx\model.onnx',
                                    providers=['CUDAExecutionProvider'])


import numpy as np

inputs_onnx = {k: np.atleast_2d(v) for k, v in input_ids.items()}

result = sess.run(None, inputs_onnx)
print(result)
