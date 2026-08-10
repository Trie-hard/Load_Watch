"""Train/evaluate logistic baseline + XGBoost; export 0–100 risk scores."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.feature_engineering import build_feature_matrix, get_model_matrix
from src.utils import (
    DATA_PROCESSED,
    FEATURE_COLUMNS,
    MODELS_DIR,
    PRIMARY_SEASON,
    ensure_dirs,
    save_parquet,
    setup_logging,
)

logger = setup_logging("loadwatch.model")


def time_based_split(df: pd.DataFrame, date_col: str = "GAME_DATE", train_frac: float = 0.75):
    """Split by chronological order — no future leakage."""
    df = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df) * train_frac)
    # Ensure cut falls on a date boundary
    cut_date = df.loc[cut, date_col]
    train = df[df[date_col] < cut_date].copy()
    test = df[df[date_col] >= cut_date].copy()
    if train.empty or test.empty:
        # Fallback pure index split
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()
    return train, test


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> XGBClassifier:
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int(len(y_train) - y_train.sum()), 1)
    spw = n_neg / n_pos
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(name: str, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "n_test": int(len(y_test)),
        "n_positive_test": int(y_test.sum()),
        "positive_rate_test": float(y_test.mean()),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        logger.warning("%s: test set has a single class — AUC undefined.", name)

    logger.info(
        "%s | ROC-AUC=%s | PR-AUC=%s | test+=%d/%d",
        name,
        f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "n/a",
        f"{metrics['pr_auc']:.3f}" if metrics["pr_auc"] is not None else "n/a",
        metrics["n_positive_test"],
        metrics["n_test"],
    )
    # Thresholded report at 0.5 for transparency only — not a primary metric
    preds = (proba >= 0.5).astype(int)
    logger.info("\n%s classification report @0.5 (interpret with caution):\n%s", name, classification_report(y_test, preds, zero_division=0))
    return metrics


def probability_to_risk_score(proba: np.ndarray, method: str = "percentile") -> np.ndarray:
    """Convert predicted probabilities into a 0–100 relative risk score."""
    proba = np.asarray(proba, dtype=float)
    if method == "linear":
        # Simple 0-1 → 0-100 (can cluster low due to rarity)
        return np.clip(proba * 100.0, 0, 100)
    # Percentile rank within the scored population (more interpretable for coaches)
    series = pd.Series(proba)
    return (series.rank(method="average", pct=True) * 100.0).to_numpy()


def risk_band(score: float) -> str:
    if score >= 85:
        return "High"
    if score >= 60:
        return "Moderate"
    return "Low"


def logistic_coefficients(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    clf: LogisticRegression = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns=["abs_coef"])
        .reset_index(drop=True)
    )


def train_and_score(season: str = PRIMARY_SEASON, rebuild_features: bool = False) -> dict:
    ensure_dirs()
    feat_path = DATA_PROCESSED / f"features_{season}.parquet"
    if rebuild_features or not feat_path.exists():
        df = build_feature_matrix(season)
    else:
        df = pd.read_parquet(feat_path)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if "distance_last_7d" in df.columns:
        feature_cols = feature_cols + ["distance_last_7d"]

    train_df, test_df = time_based_split(df)
    X_train, y_train = get_model_matrix(train_df, feature_cols)
    X_test, y_test = get_model_matrix(test_df, feature_cols)

    logger.info(
        "Time split: train=%d (+=%d) | test=%d (+=%d) | features=%s",
        len(train_df),
        int(y_train.sum()),
        len(test_df),
        int(y_test.sum()),
        feature_cols,
    )

    logit = train_logistic(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    metrics = [
        evaluate_classifier("logistic_regression", logit, X_test, y_test),
        evaluate_classifier("xgboost", xgb, X_test, y_test),
    ]

    coef_df = logistic_coefficients(logit, feature_cols)
    coef_df.to_csv(MODELS_DIR / "logistic_coefficients.csv", index=False)

    # Score full season with XGBoost (main model)
    X_all, _ = get_model_matrix(df, feature_cols)
    proba = xgb.predict_proba(X_all)[:, 1]
    risk = probability_to_risk_score(proba, method="percentile")

    scored = df.copy()
    scored["injury_proba"] = proba
    scored["risk_score"] = risk
    scored["risk_band"] = scored["risk_score"].apply(risk_band)

    scored_path = DATA_PROCESSED / f"scored_{season}.parquet"
    save_parquet(scored, scored_path)
    scored.to_csv(DATA_PROCESSED / f"scored_{season}.csv", index=False)

    joblib.dump(logit, MODELS_DIR / "logistic_pipeline.joblib")
    xgb.save_model(str(MODELS_DIR / "xgb_model.json"))
    joblib.dump(xgb, MODELS_DIR / "xgb_model.joblib")

    meta = {
        "season": season,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_end_date": str(train_df["GAME_DATE"].max().date()),
        "test_start_date": str(test_df["GAME_DATE"].min().date()),
        "notes": (
            "Validation uses a chronological train/test split. "
            "Primary metrics are ROC-AUC and PR-AUC; accuracy is not emphasized "
            "because soft-tissue labels are heavily imbalanced."
        ),
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved models + scored table (%d rows)", len(scored))
    return meta


def load_xgb_model():
    path = MODELS_DIR / "xgb_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Train the model first (scripts/run_pipeline.py).")
    return joblib.load(path)


if __name__ == "__main__":
    train_and_score(PRIMARY_SEASON, rebuild_features=False)
