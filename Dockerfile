FROM python:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN mkdir /.cache && chmod 777 /.cache

RUN mkdir /app/model && chmod 777 /app/model
RUN mkdir /app/tokenizer && chmod 777 /app/tokenizer

CMD ["python3", "train.py"]