FROM anibali/pytorch:1.8.1-cuda11.1

WORKDIR /backend

COPY . /backend

RUN pip install --upgrade pip -r requirements.txt
RUN python3 -m nltk.downloader punkt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]