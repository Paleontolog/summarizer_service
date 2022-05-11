FROM nvcr.io/nvidia/tritonserver:22.04-py3

WORKDIR /backend

COPY bert ./models/bert
COPY tokenizer ./models/tokenizer
COPY transformers ./models/transformers
COPY requirements.txt .

ENV TOKENIZER ./models/tokenizer/pretrained

RUN pip install --upgrade pip -r requirements.txt

EXPOSE 8000 8001 8002

ENTRYPOINT ["bash", "-c"]
CMD ["tritonserver --model-repository=./models", "bash"]