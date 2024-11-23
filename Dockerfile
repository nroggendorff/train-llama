FROM python:3.9

COPY --chown=user ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
pip install -U --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

ENV HF_TOKEN={$HF_TOKEN}

CMD ["python", "main.py"]