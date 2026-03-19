# Workflow-CI — Wine Quality ML Training Pipeline

![CI Pipeline](https://github.com/najwanopal/Workflow-CI/actions/workflows/ci.yml/badge.svg)
[![Docker Image](https://img.shields.io/docker/v/najwanopal/wine-quality-mlops?label=Docker%20Hub&color=blue)](https://hub.docker.com/r/najwanopal/wine-quality-mlops)

**Student:** M_Najwan_Naufal_A  
**Username Dicoding:** najwanopal  
**Course:** Membangun Sistem Machine Learning — Dicoding

---

## 📋 Deskripsi

Repository ini berisi **MLflow Project** untuk melatih model klasifikasi biner Wine Quality.  
Pipeline CI/CD menggunakan **GitHub Actions** untuk otomatisasi training dan Docker image build.

### Arsitektur CI/CD

```
Push ke main
    │
    ├── [Job 1] Lint & Validate
    │       └── flake8 + cek file
    │
    ├── [Job 2] Train Model
    │       ├── Install dependencies
    │       ├── Run modelling.py (manual MLflow logging)
    │       ├── Generate artifacts (5 plots + report)
    │       └── Upload artifacts
    │
    └── [Job 3] Build & Push Docker
            ├── docker build
            └── docker push → Docker Hub
```

---

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/workflows/
│   └── ci.yml                          ← GitHub Actions workflow
├── MLProject/
│   ├── MLProject                       ← MLflow Project config
│   ├── modelling.py                    ← Training script (manual logging)
│   ├── conda.yaml                      ← Conda environment spec
│   ├── DockerHub.txt                   ← Docker Hub link
│   └── winequality_preprocessing/      ← Data siap latih
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── Dockerfile                          ← Docker image untuk training
└── README.md                           ← Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

### 1. Jalankan MLProject Lokal (Standalone)

```bash
cd MLProject

# Default parameters
python modelling.py

# Custom parameters
python modelling.py \
    --data-dir winequality_preprocessing \
    --n-estimators 200 \
    --max-depth 15 \
    --random-state 42 \
    --experiment-name wine-quality-classification
```

### 2. Jalankan via MLflow CLI

```bash
# Dari root Workflow-CI/
mlflow run MLProject \
    -P n_estimators=200 \
    -P max_depth=15

# Dengan Docker
mlflow run MLProject \
    --env-manager=local \
    -P n_estimators=200
```

### 3. Jalankan via Docker

```bash
# Build image
docker build -t najwanopal/wine-quality-mlops:latest .

# Run training
docker run -it najwanopal/wine-quality-mlops:latest

# Custom parameters
docker run -it najwanopal/wine-quality-mlops:latest \
    --data-dir winequality_preprocessing \
    --n-estimators 200 \
    --max-depth 15
```

---

## 🔄 Trigger GitHub Actions

### Otomatis
- **Push** ke branch `main` → CI pipeline jalan otomatis
- **Pull Request** ke branch `main` → lint + training

### Manual (Workflow Dispatch)
1. Buka tab **Actions** di GitHub repo
2. Pilih workflow **ML Training CI Pipeline**
3. Klik **Run workflow**
4. (Opsional) Isi `n_estimators` dan `max_depth`
5. Klik **Run workflow**

---

## 🐳 Docker Hub

- **Image:** [`najwanopal/wine-quality-mlops`](https://hub.docker.com/r/najwanopal/wine-quality-mlops)
- **Tags:** `latest`, `<commit-sha>`

### Push ke Docker Hub (manual)

```bash
docker login
docker build -t najwanopal/wine-quality-mlops:latest .
docker push najwanopal/wine-quality-mlops:latest
```

### GitHub Secrets yang Diperlukan

| Secret | Deskripsi |
|--------|-----------|
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access Token Docker Hub |

---

## 📊 CI Pipeline Output

Setiap run menghasilkan:

| Artifact | Deskripsi |
|----------|-----------|
| `confusion_matrix.png` | Confusion matrix |
| `roc_curve.png` | ROC curve + AUC |
| `pr_curve.png` | Precision-Recall curve |
| `feature_importance.png` | Feature importance |
| `classification_report.txt` | Classification report |
| `model.pkl` | Trained model |
| `metadata.json` | Run metadata + metrics |

---

## 📝 MLflow Manual Logging

Script ini menggunakan **manual logging** (BUKAN autolog) sesuai syarat Advanced:

```python
# Parameters — logged manually
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)

# Metrics — logged manually
mlflow.log_metric("test_accuracy", 0.85)
mlflow.log_metric("test_f1", 0.67)

# Tags — logged manually
mlflow.set_tag("experiment_type", "mlproject-training")

# Model — logged manually
mlflow.sklearn.log_model(model, artifact_path="model")

# Artifacts — logged manually
mlflow.log_artifact("confusion_matrix.png", "plots")
```
