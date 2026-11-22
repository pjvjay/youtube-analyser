# One image for the HTTP API (yta serve, the default command) and the workers (yta worker ...),
# on one machine or on Azure Container Apps (see docs/cloud.md). Build from the repository root:
#   docker build -t youtube-analyser .
#   docker run -p 8000:8000 -e API_KEY=change-me youtube-analyser
#   docker run youtube-analyser yta worker --store azblob://acct/yta --queue azqueue://acct/yta-work --until-empty

# 1. the wheel (the package with its models)
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps -w /wheels .

# 2. the image: ffmpeg, the wheel with OpenVINO, the API, Azure storage and yt-dlp
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
# openvino-telemetry would send a usage event on every import; OpenVINO works without it.
RUN pip install --no-cache-dir "$(ls /wheels/*.whl)[openvino,service,azure,youtube]" \
 && pip uninstall -y openvino-telemetry \
 && rm -rf /wheels
# The bank of title frames (yta intro --build-bank) is built from your videos, so it is not in
# the repository; it is copied in when it is there (README.md keeps the line valid without it).
WORKDIR /app
COPY README.md intro_bank.np[z] /app/
ENV WORK_DIR=/tmp/yta \
    INTRO_BANK=/app/intro_bank.npz \
    PYTHONUNBUFFERED=1
USER nobody
EXPOSE 8000
CMD ["yta", "serve", "--host", "0.0.0.0", "--port", "8000"]
