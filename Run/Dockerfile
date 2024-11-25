FROM nroggendorff/train-llama:latest

USER root
RUN python -u prep.py

CMD ["python", "train.py"]