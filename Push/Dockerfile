FROM docker:dind

USER root

RUN apk add --no-cache git bash

ARG REPO_URL=https://github.com/nroggendorff/train-llama.git

WORKDIR /workspace
RUN git clone $REPO_URL .

RUN dockerd && \
    while ! docker info > /dev/null 2>&1; do\n\
        echo 'Waiting for Docker daemon...'\n\
        sleep 1\n\
    done

RUN --mount=type=secret,id=HF_TOKEN,mode=0444 \
    docker build -f Build/Dockerfile \
    --build-arg HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    -t nroggendorff/train-llama:latest .

RUN --mount=type=secret,id=DOCKER_PASSWORD,mode=0444 \
    echo $(cat /run/secrets/DOCKER_PASSWORD) | \
    docker login --username "nroggendorff" --password-stdin

CMD ["docker push", "nroggendorff/train-llama:latest"]