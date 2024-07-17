FROM python:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /.cache && chmod 777 /.cache

RUN mkdir /app/mayo && chmod 777 /app/mayo

COPY app.py .

CMD ["python3", "app.py"]