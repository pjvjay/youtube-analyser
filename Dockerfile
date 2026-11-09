# The standing-detector API (service/app.py) as one container, for running locally or on
# Azure Container Apps. Build from the repository root:
#   docker build -t standing-api .
#   docker run -p 8000:8000 -e API_KEY=change-me standing-api
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY standing/requirements.txt standing/requirements.txt
COPY service/requirements.txt service/requirements.txt
# openvino-telemetry would send a usage event on every import; OpenVINO works without it.
RUN pip install --no-cache-dir -r service/requirements.txt \
 && pip uninstall -y openvino-telemetry

COPY standing/detect_standing.py standing/
COPY standing/models/yolo11s-pose.onnx standing/models/
COPY service/app.py service/

ENV WORK_DIR=/tmp/standing-jobs \
    IMGSZ=640 \
    PYTHONUNBUFFERED=1
USER nobody
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
