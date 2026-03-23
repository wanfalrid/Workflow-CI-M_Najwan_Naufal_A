# ──────────────────────────────────────────────────────────────────
#  Dockerfile — Wine Quality MLflow Project
#  Student: M_Najwan_Naufal_A
# ──────────────────────────────────────────────────────────────────
#
#  Build:  docker build -t wanfalrid/wine-quality-mlops:latest .
#  Push:   docker push wanfalrid/wine-quality-mlops:latest
#  Run:    docker run -it wanfalrid/wine-quality-mlops:latest
# ──────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="M_Najwan_Naufal_A <wanfalrid>"
LABEL description="Wine Quality MLflow Project — Binary Classification"
LABEL version="1.0"

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────
COPY MLProject/conda.yaml /app/conda.yaml

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pandas>=2.0.0 \
        numpy>=1.24.0 \
        scikit-learn>=1.3.0 \
        mlflow>=2.10.0 \
        dagshub>=0.3.0 \
        matplotlib>=3.7.0 \
        seaborn>=0.12.0 \
        joblib>=1.3.0 \
        optuna>=3.5.0

# ── Copy MLProject files ────────────────────────────────────────
COPY MLProject/ /app/

# ── Environment variables ────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import mlflow; print('OK')" || exit 1

# ── Entry point ──────────────────────────────────────────────────
# Default: run modelling.py with default parameters
# Override with: docker run <image> python modelling.py --n-estimators 200
ENTRYPOINT ["python", "modelling.py"]
CMD ["--data-dir", "winequality_preprocessing"]
