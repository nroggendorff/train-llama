FROM python:latest

WORKDIR /app

ENV TRANSFORMERS_CACHE /app/cache

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/cache && chmod 777 /app/cache

COPY app.py .

CMD ["python", "app.py"]