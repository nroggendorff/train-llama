FROM python:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /.cache && chmod 777 /.cache

COPY app.py .

RUN python app.py

CMD curl -X POST "https://discord.com/api/webhooks/1245084721923358730/pVHUf2PR4Wst52KVNxVSeAHnSIKxx-PLdd90OHASegb30cNoGZe9N476LzCDVLQXDbT0" \
    -H "Content-Type: application/json" \
    -d '{"content": "that shit is finally done"}'