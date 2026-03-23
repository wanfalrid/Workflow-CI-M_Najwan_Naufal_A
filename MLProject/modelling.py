"""
=============================================================================
  MLProject Modelling — Wine Quality Binary Classification
  Student : M_Najwan_Naufal_A
  Adapted from Kriteria 2 modelling for MLflow Project execution.
=============================================================================

Manual MLflow logging — BUKAN autolog.
Menerima parameter dari MLProject entry point via argparse.

Usage (standalone):
    python modelling.py --data-dir winequality_preprocessing
    python modelling.py --n-estimators 200 --max-depth 15

Usage (via MLflow):
    mlflow run . -P n_estimators=200 -P max_depth=15
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report,
    ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mlproject_modelling")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# ===================================================================== #
#  1. LOAD DATA                                                          #
# ===================================================================== #
def load_data(data_dir: str) -> tuple:
    """Load preprocessed train/test data."""
    log.info(f"Loading data: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    log.info(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    log.info(f"  y_train: 0={int((y_train==0).sum())} 1={int((y_train==1).sum())}")
    return X_train, X_test, y_train, y_test


# ===================================================================== #
#  2. CREATE MODEL                                                       #
# ===================================================================== #
def create_model(n_estimators, max_depth, random_state):
    """Create a RandomForestClassifier with given parameters."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth > 0 else None,
        "random_state": random_state,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    log.info(f"Model params: {params}")
    return RandomForestClassifier(**params), params


# ===================================================================== #
#  3. LOG METRICS — MANUAL                                               #
# ===================================================================== #
def log_metrics_manual(model, X_train, X_test, y_train, y_test, training_time):
    """Manual logging of ALL metrics to MLflow active run."""

    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        # Train
        "training_accuracy": accuracy_score(y_train, y_train_pred),
        "training_precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "training_recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "training_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "training_roc_auc": roc_auc_score(y_train, y_train_proba),
        # Test
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba),
        # Binary (class 1)
        "test_precision_binary": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall_binary": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1_binary": f1_score(y_test, y_test_pred, zero_division=0),
        # Time
        "training_time_seconds": training_time,
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    mlflow.log_metrics(metrics)
    log.info("  Metrics logged:")
    for k, v in sorted(metrics.items()):
        log.info(f"    {k:<25s}: {v:.4f}")

    return metrics, y_test_pred, y_test_proba


# ===================================================================== #
#  4. LOG ARTIFACTS                                                      #
# ===================================================================== #
def log_artifacts(model, X_test, y_test, y_pred, y_proba, feature_names, metrics):
    """Generate and log all plot artifacts to MLflow."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Confusion Matrix ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Low Quality", "High Quality"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")
    log.info("  [1/5] Confusion matrix ✓")

    # ── 2. ROC Curve ─────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2ecc71", linewidth=2.5,
            label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2ecc71")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")
    log.info("  [2/5] ROC curve ✓")

    # ── 3. Precision-Recall Curve ────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#e74c3c", linewidth=2.5,
            label=f"AP = {avg_prec:.4f}")
    ax.fill_between(rec, prec, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(ARTIFACTS_DIR, "pr_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")
    log.info("  [3/5] PR curve ✓")

    # ── 4. Feature Importance ────────────────────────────────────────
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_names)), importances[indices],
            color=[colors[i] for i in indices], edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for i, imp in enumerate(importances[indices]):
        ax.text(imp + 0.003, i, f"{imp:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")
    log.info("  [4/5] Feature importance ✓")

    # ── 5. Classification Report ─────────────────────────────────────
    report = classification_report(
        y_test, y_pred, target_names=["Low Quality", "High Quality"]
    )
    path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
    with open(path, "w") as f:
        f.write("Classification Report — Wine Quality\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
        f.write(f"\nROC AUC: {metrics.get('test_roc_auc', 0):.4f}\n")
        f.write(f"\nGenerated: {datetime.now().isoformat()}\n")
    mlflow.log_artifact(path, "reports")
    log.info("  [5/5] Classification report ✓")

    return report


# ===================================================================== #
#  5. MAIN                                                               #
# ===================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="MLProject Wine Quality Modelling — Manual MLflow Logging",
    )
    parser.add_argument("--data-dir", type=str,
                        default="winequality_preprocessing",
                        help="Path to preprocessing output")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=10,
                        help="Max tree depth; 0=None (default: 10)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--experiment-name", type=str,
                        default="wine-quality-classification",
                        help="MLflow experiment name")
    args = parser.parse_args()

    print()
    log.info("=" * 65)
    log.info("  MLPROJECT — WINE QUALITY MODELLING")
    log.info("  Manual MLflow Logging (BUKAN autolog)")
    log.info("  Student: M_Najwan_Naufal_A")
    log.info("=" * 65)
    log.info(f"  data_dir:        {args.data_dir}")
    log.info(f"  n_estimators:    {args.n_estimators}")
    log.info(f"  max_depth:       {args.max_depth}")
    log.info(f"  random_state:    {args.random_state}")
    log.info(f"  experiment_name: {args.experiment_name}")
    log.info("=" * 65)

    # ── Load data ────────────────────────────────────────────────────
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
    if not os.path.exists(data_dir):
        log.error(f"Data dir tidak ditemukan: {data_dir}")
        sys.exit(1)

    X_train, X_test, y_train, y_test = load_data(data_dir)
    feature_names = list(X_train.columns)

    # ── Set experiment ───────────────────────────────────────────────
    mlflow.set_experiment(args.experiment_name)

    # ── Start run (MANUAL LOGGING) ───────────────────────────────────
    with mlflow.start_run(run_name="mlproject-training") as run:
        run_id = run.info.run_id
        log.info(f"Run ID: {run_id}")

        # ── Log parameters MANUALLY ─────────────────────────────────
        model, params = create_model(
            args.n_estimators, args.max_depth, args.random_state
        )
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("data_dir", args.data_dir)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(feature_names))

        # ── Log tags ────────────────────────────────────────────────
        mlflow.set_tags({
            "mlflow.source.type": "LOCAL",
            "mlflow.source.name": "modelling.py",
            "dataset": "wine-quality-combined",
            "experiment_type": "mlproject-training",
            "student": "M_Najwan_Naufal_A",
        })

        # ── Train ───────────────────────────────────────────────────
        log.info("Training model ...")
        t_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - t_start
        log.info(f"Training time: {training_time:.2f}s ✓")

        # ── Log metrics MANUALLY ────────────────────────────────────
        metrics, y_pred, y_proba = log_metrics_manual(
            model, X_train, X_test, y_train, y_test, training_time
        )

        # ── Log model artifact ──────────────────────────────────────
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
            log.info("  Model logged to MLflow ✓")
        except Exception as e:
            log.warning(f"  Model log warning: {e}")

        # ── Log additional artifacts ────────────────────────────────
        log.info("Generating artifacts ...")
        report = log_artifacts(
            model, X_test, y_test, y_pred, y_proba, feature_names, metrics
        )
        log.info(f"Report generated: {len(report)} chars")

        # ── Save model locally ──────────────────────────────────────
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        model_path = os.path.join(BASE_DIR, "models", "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        metadata = {
            "model_name": "RandomForestClassifier",
            "run_id": run_id,
            "params": {k: str(v) for k, v in params.items()},
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "features": feature_names,
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = os.path.join(BASE_DIR, "models", "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow.log_artifact(meta_path, "metadata")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    log.info("=" * 65)
    log.info("  TRAINING SELESAI ✓")
    log.info("=" * 65)
    log.info(f"  Test F1 (weighted) : {metrics['test_f1']:.4f}")
    log.info(f"  Test ROC AUC       : {metrics['test_roc_auc']:.4f}")
    log.info(f"  CV F1 Mean         : {metrics['cv_f1_mean']:.4f}")
    log.info(f"  Run ID             : {run_id}")
    log.info("=" * 65)
    print()


if __name__ == "__main__":
    main()
