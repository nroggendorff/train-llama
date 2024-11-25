docker buildx build -f Build/Dockerfile \
--build-arg HF_TOKEN=$(cat ~/run/secrets/HF_TOKEN) \
-t nroggendorff/train-llama:latest .