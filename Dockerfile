# The standing (and other actions) detector API (service/app.py) as one container, for running locally or on
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

COPY standing/detect_standing.py standing/intro.py standing/actions.py standing/audio.py \
     standing/people.py standing/search.py standing/people.toml standing/
# intro_bank.npz is built locally (standing/intro.py --build-bank) and copied when it exists;
# the [z] wildcard keeps the build working without it. The face models (2.7 MB) are for
# smiling, talking and silent; the voice models (7 MB) for talking and silent; the person mask
# (0.5 MB) and gender/age (49 MB, see standing/people.toml about its training data) for who=.
# To leave gender/age out, delete that file from the line below (who= then takes gender from
# voices only).
COPY standing/models/yolo11s-pose.onnx standing/models/face_landmark.onnx standing/models/face_expression.onnx \
     standing/models/voice_activity.onnx standing/models/voice_embedding.onnx \
     standing/models/person_mask.onnx standing/models/gender_age.onnx \
     standing/models/intro_bank.np[z] standing/models/
COPY service/app.py service/

ENV WORK_DIR=/tmp/standing-jobs \
    IMGSZ=640 \
    PYTHONUNBUFFERED=1
USER nobody
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
