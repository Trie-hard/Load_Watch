"""SHAP explainability for XGBoost risk scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_PROCESSED, FEATURE_COLUMNS, MODELS_DIR, PRIMARY_SEASON, ensure_dirs, save_parquet, setup_logging

logger = setup_logging("loadwatch.explain")

FRIENDLY_NAMES = {
    "minutes_last_7d": "Minutes over last 7 days",
    "minutes_last_14d": "Minutes over last 14 days",
    "games_last_7d": "Games in last 7 days",
    "is_back_to_back": "Back-to-back game",
    "rest_days": "Rest days before game",
    "season_minutes_cumulative": "Cumulative season minutes",
    "age": "Player age",
    "days_since_last_injury": "Days since last soft-tissue injury",
    "distance_last_7d": "Estimated distance last 7 days",
}


def _feature_cols(df: pd.DataFrame, meta_features: list[str] | None = None) -> list[str]:
    if meta_features:
        return [c for c in meta_features if c in df.columns]
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if "distance_last_7d" in df.columns:
        cols.append("distance_last_7d")
    return cols


def compute_shap_top_factors(
    season: str = PRIMARY_SEASON,
    top_k: int = 5,
    max_rows_for_explainer: int = 8000,
) -> pd.DataFrame:
    """
    Compute TreeSHAP values for scored player-games and persist top contributing factors.
    """
    import joblib
    import shap

    ensure_dirs()
    scored_path = DATA_PROCESSED / f"scored_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing {scored_path}; run model training first.")

    df = pd.read_parquet(scored_path)
    metrics_path = MODELS_DIR / "metrics.json"
    meta_features = None
    if metrics_path.exists():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta_features = meta.get("feature_columns")

    feature_cols = _feature_cols(df, meta_features)
    model = joblib.load(MODELS_DIR / "xgb_model.joblib")

    X = df[feature_cols].astype(float)

    # Background sample keeps SHAP tractable on full-season tables
    if len(X) > max_rows_for_explainer:
        bg = X.sample(max_rows_for_explainer, random_state=42)
    else:
        bg = X

    logger.info("Fitting TreeExplainer on %d background rows...", len(bg))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    # Convert SHAP log-odds contributions into a display-friendly "pts" scale
    # by ranking absolute impact within each row and reporting signed contribution * 10
    records = []
    for i in range(len(df)):
        row_shap = shap_df.iloc[i]
        top = row_shap.reindex(row_shap.abs().sort_values(ascending=False).index)[:top_k]
        factors = []
        for feat, val in top.items():
            direction = "Elevated" if val > 0 else "Reduced"
            factors.append(
                {
                    "feature": feat,
                    "label": FRIENDLY_NAMES.get(feat, feat),
                    "shap": float(val),
                    "display": f"{direction}: {FRIENDLY_NAMES.get(feat, feat)} ({val * 10:+.1f} pts)",
                    "feature_value": float(X.iloc[i][feat]),
                }
            )
        records.append(
            {
                "PLAYER_ID": int(df.iloc[i]["PLAYER_ID"]),
                "GAME_DATE": df.iloc[i]["GAME_DATE"],
                "top_factors_json": json.dumps(factors),
                "top_factor_1": factors[0]["display"] if factors else "",
                "top_factor_2": factors[1]["display"] if len(factors) > 1 else "",
                "top_factor_3": factors[2]["display"] if len(factors) > 2 else "",
            }
        )

    out = pd.DataFrame(records)
    # Also store dense SHAP matrix for optional deep dives
    shap_full = pd.concat(
        [
            df[["PLAYER_ID", "GAME_DATE", "PLAYER_NAME", "TEAM_ABBREVIATION", "risk_score"]].reset_index(drop=True),
            shap_df.add_prefix("shap_").reset_index(drop=True),
            out[["top_factors_json", "top_factor_1", "top_factor_2", "top_factor_3"]],
        ],
        axis=1,
    )

    save_parquet(shap_full, DATA_PROCESSED / f"shap_{season}.parquet")
    # Merge top factors onto scored table for the dashboard
    merged = df.merge(
        out[["PLAYER_ID", "GAME_DATE", "top_factors_json", "top_factor_1", "top_factor_2", "top_factor_3"]],
        on=["PLAYER_ID", "GAME_DATE"],
        how="left",
    )
    save_parquet(merged, DATA_PROCESSED / f"scored_{season}.parquet")
    merged.to_csv(DATA_PROCESSED / f"scored_{season}.csv", index=False)

    # Global mean |SHAP|
    importance = shap_df.abs().mean().sort_values(ascending=False)
    importance.to_csv(MODELS_DIR / "shap_global_importance.csv", header=["mean_abs_shap"])
    logger.info("Global SHAP importance:\n%s", importance.head(10).to_string())
    return shap_full


def factors_for_row(top_factors_json: str) -> list[dict]:
    if not top_factors_json or (isinstance(top_factors_json, float) and np.isnan(top_factors_json)):
        return []
    try:
        return json.loads(top_factors_json)
    except (TypeError, json.JSONDecodeError):
        return []


if __name__ == "__main__":
    compute_shap_top_factors(PRIMARY_SEASON)
