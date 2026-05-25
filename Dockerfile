# ──────────────────────────────────────────────────────────────────────────────
# het-ai training image
# Contains: Python 3.12, DVC CLI, MLflow, all framework extras (torch/tf/sklearn)
#
# Usage (one-shot training run):
#   docker build -t het-ai .
#   docker run --rm \
#     -e DVC_GITHUB_REPO=owner/data-repo \
#     -e DVC_GITHUB_TOKEN=$GITHUB_TOKEN \
#     -e MINIO_ENDPOINT=minio:9000 \
#     -e MINIO_ACCESS_KEY=minioadmin \
#     -e MINIO_SECRET_KEY=minioadmin \
#     -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#     -e MLFLOW_EXPERIMENT=my_experiment \
#     het-ai python -m my_trainer
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies required by DVC (git) and some ML libs (libgomp for sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ── Install het-ai with all production extras ─────────────────────────────────
COPY pyproject.toml setup.py README.rst NOTICE LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[platform]" \
 && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    scikit-learn \
    joblib \
    onnx \
    onnxruntime

# ── Copy application code last (cache-friendly) ───────────────────────────────
COPY . .

# Default entrypoint: run all dry-run smoke tests
CMD ["python", "-m", "pytest", "tests/test_simulate_all.py", "-v"]
