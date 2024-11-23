FROM docker

USER root

RUN apk add --no-cache git bash

ARG DOCKER_USERNAME='nroggendorff'

RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
export HF_TOKEN=$(cat /run/secrets/HF_TOKEN)

ARG REPO_URL=https://github.com/nroggendorff/train-llama.git

WORKDIR /workspace
RUN git clone $REPO_URL build

RUN docker build build \
-f Dockerfile \
--build-arg HF_TOKEN=${HF_TOKEN} \
-t nroggendorff/train-llama:latest

RUN --mount=type=secret,id=DOCKER_PASSWORD,mode=0444,required=true \
cat /run/secrets/DOCKER_PASSWORD | \
docker login --username "nroggendorff" --password-stdin

CMD ["docker push", "nroggendorff/train-llama:latest"]