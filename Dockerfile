FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

RUN touch __init__.py
RUN mv configlib config.py && mv util util.py

RUN python -u prep.py

CMD ["python", "train.py"]