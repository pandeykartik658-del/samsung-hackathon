# Dockerfile
# CPU-only image for the Theme 05 agent. Offline at run time: Whisper weights are baked in
# at build time (the grader may have no network), RapidOCR weights ship inside its wheel.
#
#   docker build -t theme5 .                                   # base.en weights (~145 MB)
#   docker build -t theme5 --build-arg WHISPER_MODEL=tiny.en .  # smaller, less accurate
#   docker build -t theme5 --build-arg FETCH_MODELS=0 .         # no Hugging Face access: use ./models or hints only
#   docker run --rm theme5                                     # tests + demo
#   docker run --rm -i theme5 python -m theme5 serve            # JSONL events in, JSONL actions out
FROM python:3.11-slim

ARG WHISPER_MODEL=base.en
ARG FETCH_MODELS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    THEME5_MODEL_DIR=/opt/theme5/models \
    THEME5_WHISPER_MODEL=${WHISPER_MODEL} \
    THEME5_OFFLINE=1

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
# rapidocr pins the GUI build of OpenCV, which needs libGL/libxcb from apt. Swapping in the
# headless wheel of the same version provides the same cv2 module, needs no system packages
# and saves ~100 MB, so the image needs no apt layer at all.
RUN pip install -r requirements.txt -r requirements-dev.txt \
 && pip uninstall -y opencv-python \
 && pip install opencv-python-headless==5.0.0.93

# Weights layer sits before the source copy so code edits do not re-download models.
# Pre-downloaded weights in ./models (make models) are used as-is.
COPY models/ ${THEME5_MODEL_DIR}/
COPY scripts/fetch_models.py scripts/fetch_models.py
RUN python scripts/fetch_models.py --skip-whisper \
 && if [ "$FETCH_MODELS" = "1" ]; then \
        THEME5_OFFLINE=0 python scripts/fetch_models.py --skip-ocr --model "$WHISPER_MODEL" --dir "$THEME5_MODEL_DIR"; \
    else \
        python scripts/fetch_models.py --skip-ocr --check --model "$WHISPER_MODEL" --dir "$THEME5_MODEL_DIR" \
        || echo "WARNING: no Whisper weights in image; audio falls back to kit transcripts + clarifying questions"; \
    fi

COPY . .
RUN useradd --create-home --uid 1000 agent && chown -R agent /app
USER agent

CMD ["bash", "run_demo.sh"]
