FROM anibali/pytorch:1.8.1-cuda11.1

WORKDIR /backend

COPY . /backend

RUN pip install --upgrade pip -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]