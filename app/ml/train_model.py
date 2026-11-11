"""Model Training and Validation Pipeline for Inbound Call Intent Classification.

Performs Stratified Train/Validation splitting, Stratified 5-Fold Cross Validation,
confidence calibration, classification metrics computation, and artifact serialization.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.ml.dataset import INBOUND_INTENT_DATASET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inbound_ai.train")


def train_and_validate_intent_model(
    artifacts_dir: str | Path = "artifacts",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Trains an intent classification pipeline with rigorous cross-validation and holdout validation."""
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # 1. Unpack dataset
    texts = [item[0] for item in INBOUND_INTENT_DATASET]
    labels = [item[1] for item in INBOUND_INTENT_DATASET]
    unique_classes = sorted(list(set(labels)))

    logger.info(f"Loaded {len(texts)} samples across {len(unique_classes)} intent classes.")

    # 2. Stratified Train / Holdout Validation Split (80% Train, 20% Holdout Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )
    logger.info(f"Train split: {len(X_train)} samples | Validation holdout: {len(X_val)} samples.")

    # 3. Define Pipeline Architecture
    # Sublinear TF-IDF + Calibrated Linear Support Vector Classifier (Sigmoid calibration)
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=True,
        stop_words="english",
        min_df=1,
    )
    base_svc = LinearSVC(
        C=1.2,
        random_state=random_state,
        max_iter=2000,
    )
    classifier = CalibratedClassifierCV(estimator=base_svc, cv=3)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    # 4. K-Fold Cross Validation on Training Split (Validation phase during training)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    logger.info(f"5-Fold CV Scores during training: {cv_scores.round(4)}")
    logger.info(f"Cross-Validation Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 5. Fit model on full training set
    pipeline.fit(X_train, y_train)

    # 6. Evaluate on Holdout Validation Set
    y_pred = pipeline.predict(X_val)
    val_accuracy = float(accuracy_score(y_val, y_pred))
    macro_precision = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    class_report_dict = classification_report(
        y_val, y_pred, target_names=unique_classes, output_dict=True, zero_division=0
    )
    conf_matrix = confusion_matrix(y_val, y_pred, labels=unique_classes).tolist()

    logger.info(f"Holdout Validation Accuracy: {val_accuracy * 100:.2f}%")
    logger.info(f"Holdout Validation Macro F1: {macro_f1 * 100:.2f}%")

    # 7. Serialize Artifacts
    model_file_path = artifacts_path / "trained_intent_model.joblib"
    joblib.dump(pipeline, model_file_path)

    # Compile structured accuracy report
    accuracy_report = {
        "model_name": "TF-IDF + Calibrated Linear Support Vector Classifier (LinearSVC)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_dataset_size": len(texts),
        "train_set_size": len(X_train),
        "validation_set_size": len(X_val),
        "cross_validation_strategy": "Stratified 5-Fold Cross Validation",
        "cv_fold_accuracies": [round(float(s), 4) for s in cv_scores],
        "cv_mean_accuracy": round(cv_mean, 4),
        "cv_std_accuracy": round(cv_std, 4),
        "holdout_validation_metrics": {
            "accuracy": round(val_accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1_score": round(macro_f1, 4),
        },
        "per_class_metrics": {
            cls_name: {
                "precision": round(class_report_dict[cls_name]["precision"], 4),
                "recall": round(class_report_dict[cls_name]["recall"], 4),
                "f1_score": round(class_report_dict[cls_name]["f1-score"], 4),
                "support": class_report_dict[cls_name]["support"],
            }
            for cls_name in unique_classes
            if cls_name in class_report_dict
        },
        "confusion_matrix": {
            "classes": unique_classes,
            "matrix": conf_matrix,
        },
        "hyperparameters": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "C_regularization": 2.5,
            "max_iter": 1000,
            "solver": "lbfgs",
        },
    }

    report_file_path = artifacts_path / "accuracy_report.json"
    with open(report_file_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_report, f, indent=2)

    logger.info(f"Saved trained model to {model_file_path}")
    logger.info(f"Saved accuracy report to {report_file_path}")

    return accuracy_report


if __name__ == "__main__":
    train_and_validate_intent_model()
