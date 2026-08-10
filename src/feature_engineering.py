"""Workload feature engineering and soft-tissue injury labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_collection import (
    collect_season_workload,
    get_player_ages,
    try_get_tracking_distance,
)
from src.injury_data import build_clean_injury_table
from src.utils import (
    DATA_PROCESSED,
    FEATURE_COLUMNS,
    PRIMARY_SEASON,
    ensure_dirs,
    save_parquet,
    setup_logging,
)

logger = setup_logging("loadwatch.features")

LABEL_HORIZON_DAYS = 7


def _add_rolling_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player rolling minutes / games using calendar-day windows."""
    parts = []
    for pid, g in df.groupby("PLAYER_ID", sort=False):
        g = g.sort_values("GAME_DATE").copy()
        g = g.set_index("GAME_DATE")

        minutes = g["MIN_FLOAT"]
        # Rolling on time index: trailing 7 / 14 calendar days (exclude current via closed='left' not available
        # on older pandas rolling — compute as sum including today then subtract today's minutes where needed.
        # For risk-before-game semantics we want features known entering the game:
        # use shift(1) then rolling.
        prior_min = minutes.shift(1)
        prior_played = (~prior_min.isna()).astype(float)
        # After shift, first game has NaN — fill 0 for rolling sums of prior workload
        prior_min = prior_min.fillna(0.0)

        roll7 = prior_min.rolling("7D").sum()
        roll14 = prior_min.rolling("14D").sum()
        games7 = prior_min.gt(0).astype(float).rolling("7D").sum()

        g["minutes_last_7d"] = roll7.values
        g["minutes_last_14d"] = roll14.values
        g["games_last_7d"] = games7.values
        g["season_minutes_cumulative"] = prior_min.cumsum().values

        prev_date = g.index.to_series().shift(1)
        rest = (g.index.to_series() - prev_date).dt.days - 1
        g["rest_days"] = rest.fillna(5).clip(lower=0).astype(float).values
        g["is_back_to_back"] = (g["rest_days"] == 0).astype(int).values

        parts.append(g.reset_index())

    return pd.concat(parts, ignore_index=True)


def _add_injury_recency(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """days_since_last_injury at each player-game (large number if never injured)."""
    if injuries is None or injuries.empty:
        df["days_since_last_injury"] = 365.0
        return df

    inj = injuries[["player_id", "injury_date"]].copy()
    inj["player_id"] = inj["player_id"].astype(int)
    inj["injury_date"] = pd.to_datetime(inj["injury_date"])
    inj = inj.sort_values(["player_id", "injury_date"])

    out_parts = []
    inj_by_player = {pid: grp["injury_date"].values.astype("datetime64[ns]") for pid, grp in inj.groupby("player_id")}

    for pid, g in df.groupby("PLAYER_ID", sort=False):
        g = g.sort_values("GAME_DATE").copy()
        dates = g["GAME_DATE"].values.astype("datetime64[ns]")
        hist = inj_by_player.get(int(pid))
        if hist is None or len(hist) == 0:
            g["days_since_last_injury"] = 365.0
        else:
            # For each game date, last injury strictly before game
            days = []
            for d in dates:
                prior = hist[hist < d]
                if len(prior) == 0:
                    days.append(365.0)
                else:
                    delta = (d - prior[-1]) / np.timedelta64(1, "D")
                    days.append(float(delta))
            g["days_since_last_injury"] = days
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def _add_labels(df: pd.DataFrame, injuries: pd.DataFrame, horizon_days: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """
    Label = 1 if a soft-tissue injury occurs within the next `horizon_days` days
    after the game date (exclusive of much older injuries).
    """
    df = df.copy()
    df["label"] = 0
    if injuries is None or injuries.empty:
        return df

    inj = injuries.copy()
    inj["player_id"] = inj["player_id"].astype(int)
    inj["injury_date"] = pd.to_datetime(inj["injury_date"])
    inj_by_player = {pid: grp["injury_date"].values.astype("datetime64[ns]") for pid, grp in inj.groupby("player_id")}

    labels = []
    for pid, g in df.groupby("PLAYER_ID", sort=False):
        dates = g["GAME_DATE"].values.astype("datetime64[ns]")
        hist = inj_by_player.get(int(pid))
        if hist is None or len(hist) == 0:
            labels.extend([0] * len(g))
            continue
        for d in dates:
            window_end = d + np.timedelta64(horizon_days, "D")
            # injury after game, within horizon
            hit = np.any((hist > d) & (hist <= window_end))
            labels.append(1 if hit else 0)

    df["label"] = labels
    return df


def _merge_age(df: pd.DataFrame, season: str) -> pd.DataFrame:
    ids = df["PLAYER_ID"].dropna().astype(int).unique().tolist()
    # Limit CommonPlayerInfo calls: only players with meaningful minutes
    minutes_by_player = df.groupby("PLAYER_ID")["MIN_FLOAT"].sum()
    qualifying = minutes_by_player[minutes_by_player >= 200].index.astype(int).tolist()
    # Always include labeled / high-minute players; sample rest if huge
    ages = get_player_ages(qualifying, season=season)
    age_map = dict(zip(ages["PLAYER_ID"], ages["AGE"]))
    df["age"] = df["PLAYER_ID"].map(age_map)
    # Fallback median age ~26
    df["age"] = df["age"].fillna(df["age"].median() if df["age"].notna().any() else 26.0)
    return df


def _maybe_add_distance(df: pd.DataFrame, season: str) -> tuple[pd.DataFrame, bool]:
    """
    Tracking endpoints usually give season aggregates, not per-game distance.
    We approximate distance_last_7d ≈ minutes_last_7d * (season DIST_MILES / season MIN).
    If tracking data is unavailable, skip the feature.
    """
    tracking = try_get_tracking_distance(season)
    if tracking is None or tracking.empty:
        return df, False

    t = tracking.copy()
    # Column names vary slightly across seasons
    id_col = "PLAYER_ID" if "PLAYER_ID" in t.columns else None
    dist_col = next((c for c in t.columns if "DIST" in c.upper() and "MILE" in c.upper()), None)
    min_col = next((c for c in ("MIN", "MINUTES") if c in t.columns), None)
    if id_col is None or dist_col is None:
        logger.warning("Tracking frame missing expected columns; skipping distance feature.")
        return df, False

    t["miles_per_min"] = t[dist_col] / t[min_col].replace(0, np.nan) if min_col else t[dist_col] / 30.0
    rate = t.set_index(id_col)["miles_per_min"].to_dict()
    df["distance_last_7d"] = df["PLAYER_ID"].map(rate).fillna(0) * df["minutes_last_7d"]
    return df, True


def build_feature_matrix(season: str = PRIMARY_SEASON, min_season_minutes: float = 150.0) -> pd.DataFrame:
    """Build the modeling table for one season and persist it."""
    ensure_dirs()
    logger.info("Collecting workload for %s...", season)
    logs = collect_season_workload(season)
    logs = logs.dropna(subset=["PLAYER_ID", "GAME_DATE"]).copy()
    logs["PLAYER_ID"] = logs["PLAYER_ID"].astype(int)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    # Filter to players with enough sample
    totals = logs.groupby("PLAYER_ID")["MIN_FLOAT"].sum()
    keep_ids = totals[totals >= min_season_minutes].index
    logs = logs[logs["PLAYER_ID"].isin(keep_ids)].copy()
    logger.info("Players with >= %.0f season minutes: %d", min_season_minutes, logs["PLAYER_ID"].nunique())

    logger.info("Engineering rolling workload features...")
    feats = _add_rolling_workload(logs)

    logger.info("Building injury labels...")
    injuries = build_clean_injury_table(season)
    feats = _add_injury_recency(feats, injuries)
    feats = _add_labels(feats, injuries)

    logger.info("Attaching ages (API calls; cached)...")
    feats = _merge_age(feats, season)

    feats, has_distance = _maybe_add_distance(feats, season)
    feature_cols = list(FEATURE_COLUMNS)
    if has_distance and "distance_last_7d" not in feature_cols:
        feature_cols.append("distance_last_7d")

    # Drop early-season rows with insufficient history noise if desired — keep all for dashboard
    feats["is_back_to_back"] = feats["is_back_to_back"].astype(int)
    for c in feature_cols:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0)

    out_path = DATA_PROCESSED / f"features_{season}.parquet"
    save_parquet(feats, out_path)
    feats.to_csv(DATA_PROCESSED / f"features_{season}.csv", index=False)

    pos_rate = feats["label"].mean() if len(feats) else 0
    logger.info(
        "Feature matrix: %s rows | positive label rate=%.3f%% | distance_feature=%s",
        f"{len(feats):,}",
        100 * pos_rate,
        has_distance,
    )

    meta = {
        "season": season,
        "n_rows": len(feats),
        "n_players": int(feats["PLAYER_ID"].nunique()),
        "positive_rate": float(pos_rate),
        "feature_columns": feature_cols,
        "has_distance": has_distance,
    }
    pd.Series(meta).to_json(DATA_PROCESSED / f"features_meta_{season}.json")
    return feats


def get_model_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_cols or [c for c in FEATURE_COLUMNS if c in df.columns]
    if "distance_last_7d" in df.columns and "distance_last_7d" not in cols:
        cols = cols + ["distance_last_7d"]
    X = df[cols].astype(float)
    y = df["label"].astype(int)
    return X, y


if __name__ == "__main__":
    build_feature_matrix(PRIMARY_SEASON)
