FROM python:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /.cache && chmod 777 /.cache

RUN mkdir /app/mayo && chmod 777 /app/mayo


CMD ["python3", "app.py"]