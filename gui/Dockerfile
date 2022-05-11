FROM python:3.9

WORKDIR /backend

COPY . /backend

RUN pip3 install --upgrade pip -r requirements.txt

RUN python3 -m nltk.downloader punkt

ENV BINARY_CLASSIFIER_TYPE BASE
ENV SUMMARIZER_TYPE BASE

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]