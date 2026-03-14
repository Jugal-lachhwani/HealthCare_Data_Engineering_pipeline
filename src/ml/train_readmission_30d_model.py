from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


TARGET_COL = "readmitted_under_30_days"

# Exclude outcome-leakage and technical columns.
EXCLUDE_COLS = {
    TARGET_COL,
    "readmitted",
    "days_until_readmission",
    "processed_at_utc",
}

ID_COLS = ["patientunitstayid", "uniquepid", "hospitalid", "wardid"]


@dataclass
class TrainConfig:
    input_csv: str
    model_path: str
    predictions_csv: str
    metrics_json: str
    feature_importance_csv: str
    test_size: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 30-day readmission risk model")
    parser.add_argument("--input-csv", default="Data/final/readmission_model_dataset.csv")
    parser.add_argument("--model-path", default="models/readmission_30d_model.joblib")
    parser.add_argument("--predictions-csv", default="Data/final/readmission_predictions.csv")
    parser.add_argument("--metrics-json", default="Data/final/readmission_model_metrics.json")
    parser.add_argument("--feature-importance-csv", default="Data/final/readmission_feature_importance.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET_COL}")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Parse object columns as numeric when possible to avoid unnecessary one-hot expansion.
    for col in X.columns:
        if col in numeric_cols:
            continue
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            X[col] = converted

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def build_candidate_pipelines(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {}

    candidates["logistic_regression"] = Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)),
        ]
    )

    candidates["random_forest"] = Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    candidates["extra_trees"] = Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=700,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return candidates


def find_best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0

    # precision/recall arrays are one element longer than thresholds.
    p = precision[:-1]
    r = recall[:-1]
    f1_scores = (2 * p * r) / np.clip(p + r, a_min=1e-12, a_max=None)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def export_feature_importance(model: Pipeline, output_csv: str) -> None:
    prep = model.named_steps["prep"]
    clf = model.named_steps["model"]

    feature_names = prep.get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        weights = np.asarray(clf.feature_importances_)
    else:
        weights = np.zeros(len(feature_names), dtype=float)

    importance = pd.DataFrame({"feature": feature_names, "importance": weights}).sort_values(
        "importance", ascending=False
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_path, index=False)


def export_predictions(
    model: Pipeline,
    source_df: pd.DataFrame,
    X: pd.DataFrame,
    threshold: float,
    output_csv: str,
) -> None:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    prediction_df = source_df.copy()
    prediction_df["predicted_readmission_risk"] = probs.round(6)
    prediction_df["predicted_readmission_30d"] = preds

    cols = [col for col in ID_COLS if col in prediction_df.columns]
    cols += [
        TARGET_COL,
        "predicted_readmission_30d",
        "predicted_readmission_risk",
    ]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df[cols].to_csv(out_path, index=False)


def save_metrics(metrics: dict[str, float], cfg: TrainConfig, y: pd.Series) -> None:
    payload = {
        "rows": int(len(y)),
        "target_positive_rate_pct": round(float(y.mean() * 100.0), 4),
        "metrics": metrics,
        "artifacts": {
            "model_path": cfg.model_path,
            "predictions_csv": cfg.predictions_csv,
            "feature_importance_csv": cfg.feature_importance_csv,
        },
    }

    out_path = Path(cfg.metrics_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        input_csv=args.input_csv,
        model_path=args.model_path,
        predictions_csv=args.predictions_csv,
        metrics_json=args.metrics_json,
        feature_importance_csv=args.feature_importance_csv,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    df = load_data(cfg.input_csv)
    X, y, numeric_cols, categorical_cols = select_features(df)

    if y.nunique() < 2:
        raise ValueError("Target has only one class; cannot train a classifier.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=cfg.random_state,
        stratify=y_train_val,
    )

    candidates = build_candidate_pipelines(numeric_cols, categorical_cols, cfg.random_state)
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_threshold = 0.5
    best_val_f1 = -1.0

    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        val_prob = candidate.predict_proba(X_val)[:, 1]
        threshold, val_f1 = find_best_f1_threshold(y_val, val_prob)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_name = name
            best_pipeline = candidate
            best_threshold = threshold

    if best_pipeline is None:
        raise ValueError("No model candidate was trained.")

    # Refit winner on full train+validation data.
    best_pipeline.fit(X_train_val, y_train_val)

    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_prob, best_threshold)
    metrics["threshold_selection"] = "best_f1_on_validation"
    metrics["validation_f1"] = float(best_val_f1)
    metrics["selected_model"] = best_name

    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_pipeline, model_path)

    export_feature_importance(best_pipeline, cfg.feature_importance_csv)
    export_predictions(best_pipeline, df, X, best_threshold, cfg.predictions_csv)
    save_metrics(metrics, cfg, y)

    print("Training complete")
    print(f"Rows: {len(df)}")
    print(f"Positive rate: {y.mean() * 100:.2f}%")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Selected Model: {metrics['selected_model']}")
    print(f"Validation F1 (selection): {metrics['validation_f1']:.4f}")
    print(f"F1: {metrics['f1']:.4f} @ threshold {metrics['threshold']:.4f}")
    print(
        "Confusion Matrix (tn, fp, fn, tp): "
        f"{metrics['confusion_matrix']['tn']}, {metrics['confusion_matrix']['fp']}, "
        f"{metrics['confusion_matrix']['fn']}, {metrics['confusion_matrix']['tp']}"
    )
    print(f"Model saved: {cfg.model_path}")
    print(f"Predictions saved: {cfg.predictions_csv}")
    print(f"Metrics saved: {cfg.metrics_json}")


if __name__ == "__main__":
    main()
