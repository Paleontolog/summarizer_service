FROM python:3.9

WORKDIR /backend

COPY . /backend

RUN pip3 install --upgrade pip -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]