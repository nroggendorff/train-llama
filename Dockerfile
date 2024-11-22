FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install -U --no-cache-dir -r requirements.txt

COPY --chown=user . /app

RUN touch __init__.py
RUN mv configlib config.py && mv util util.py && mv config config.json

ENV HF_TOKEN $(cat /run/secrets/HF_TOKEN)

RUN python -u prep.py

CMD ["python", "train.py"]