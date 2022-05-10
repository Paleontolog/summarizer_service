FROM nvcr.io/nvidia/tritonserver:22.04-py3

WORKDIR /backend

COPY . /backend

RUN pip install --upgrade pip -r requirements.txt

EXPOSE 8000 8001 8002