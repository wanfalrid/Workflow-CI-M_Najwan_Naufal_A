# Workflow-CI — Wine Quality ML Training Pipeline

![CI/CD Pipeline](https://github.com/wanfalrid/Workflow-CI/actions/workflows/ci.yml/badge.svg)
[![Docker Image](https://img.shields.io/docker/v/wanfalrid/wine-quality-mlflow?label=Docker%20Hub&color=blue)](https://hub.docker.com/r/wanfalrid/wine-quality-mlflow)

**Student:** M_Najwan_Naufal_A  
**Username Dicoding:** wanfalrid  
**Course:** Membangun Sistem Machine Learning — Dicoding

---

## 📋 Deskripsi

Repository ini berisi **MLflow Project** + **GitHub Actions CI/CD Pipeline** untuk melatih model klasifikasi biner Wine Quality secara otomatis.

---

## 🔄 Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRIGGERS                                          │
│  • push ke main  • PR ke main  • manual dispatch  • weekly schedule  │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ JOB 1: Lint & Test  │
               │  • flake8 lint      │
               │  • structure check  │
               │  • data validation  │
               └──────────┬──────────┘
                          │ ✅
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
   ┌──────────────────┐    ┌──────────────────┐
   │ JOB 2: Docker    │    │ JOB 3: Train     │
   │  • build image   │───▶│  • setup DagsHub │
   │  • push to Hub   │    │  • mlflow run    │
   │  • tag sha+latest│    │  • log metrics   │
   └──────────────────┘    └──────┬───────────┘
                                  │ ✅
                                  ▼
                       ┌──────────────────┐
                       │ JOB 4: Save      │
                       │  • commit model  │
                       │  • GitHub Release│
                       └──────────────────┘
                                  │
                           ┌──────┴──────┐
                           │ (on failure)│
                           ▼             │
                  ┌──────────────┐       │
                  │ JOB 5: Alert │       │
                  │ create issue │       │
                  └──────────────┘       │
                                         ▼
                                      ✅ DONE
```

---

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/workflows/
│   └── ci.yml                          ← CI/CD pipeline (5 jobs)
├── MLProject/
│   ├── MLProject                       ← MLflow Project config
│   ├── modelling.py                    ← Training (manual MLflow logging)
│   ├── conda.yaml                      ← Conda environment spec
│   ├── DockerHub.txt                   ← Docker Hub image info
│   └── winequality_preprocessing/      ← Data siap latih
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── Dockerfile                          ← Docker image
├── secrets_setup_guide.md              ← Panduan GitHub Secrets
└── README.md                           ← Dokumentasi ini
```

---

## 🔐 GitHub Secrets

| Secret | Deskripsi | Contoh |
|--------|-----------|--------|
| `DOCKERHUB_USERNAME` | Username Docker Hub | `wanfalrid` |
| `DOCKERHUB_TOKEN` | Access token Docker Hub | `dckr_pat_xxx` |
| `DAGSHUB_USERNAME` | Username DagsHub | `wanfalrid` |
| `DAGSHUB_TOKEN` | Access token DagsHub | `a1b2c3d4e5` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `https://dagshub.com/wanfalrid/...mlflow` |

> Lihat **[secrets_setup_guide.md](secrets_setup_guide.md)** untuk panduan lengkap.

---

## 🚀 Cara Menjalankan

### Trigger CI via GitHub Actions

**Otomatis:**
- Push ke `main` → pipeline jalan
- Setiap Minggu 02:00 UTC → scheduled run

**Manual Dispatch:**
1. Buka tab **Actions** di GitHub
2. Pilih **ML Training CI/CD Pipeline**
3. Klik **Run workflow**
4. Isi `n_estimators`, `max_depth` (opsional)
5. Klik **Run workflow**

### Jalankan Lokal

```bash
cd MLProject

# Default
python modelling.py

# Custom parameters
python modelling.py \
    --n-estimators 200 \
    --max-depth 15 \
    --random-state 42
```

### Docker

```bash
# Build
docker build -t wanfalrid/wine-quality-mlflow:latest .

# Run
docker run -it wanfalrid/wine-quality-mlflow:latest \
    --n-estimators 200 --max-depth 15
```

---

## 📊 Output Artifacts

| File | Deskripsi |
|------|-----------|
| `confusion_matrix.png` | Confusion matrix |
| `roc_curve.png` | ROC curve + AUC |
| `pr_curve.png` | Precision-Recall curve |
| `feature_importance.png` | Feature importance |
| `classification_report.txt` | Full classification report |
| `model.pkl` | Trained model pickle |
| `metadata.json` | Parameters, metrics, run info |

---

## 📝 Manual MLflow Logging

Script menggunakan **100% manual logging** (BUKAN `mlflow.sklearn.autolog()`):

```python
# Parameters
mlflow.log_param("n_estimators", 100)

# Metrics
mlflow.log_metric("test_accuracy", 0.85)
mlflow.log_metric("test_f1", 0.77)
mlflow.log_metric("training_time_seconds", 2.5)

# Tags
mlflow.set_tag("experiment_type", "mlproject-training")
mlflow.set_tag("mlflow.source.type", "LOCAL")

# Model
mlflow.sklearn.log_model(model, artifact_path="model")

# Custom artifacts
mlflow.log_artifact("roc_curve.png", "plots")
```
