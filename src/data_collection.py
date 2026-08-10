"""Pull NBA workload / schedule data via nba_api with caching and retries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguegamelog,
    playergamelog,
    teamgamelog,
)
from nba_api.stats.static import players, teams

from src.utils import (
    DATA_RAW,
    PRIMARY_SEASON,
    ensure_dirs,
    load_parquet,
    retry_with_backoff,
    save_parquet,
    setup_logging,
)

logger = setup_logging("loadwatch.data")

# Gentle pacing between API calls
API_SLEEP = 0.7


@retry_with_backoff(max_attempts=5, base_delay=2.0)
def _fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    time.sleep(API_SLEEP)
    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    df = gl.get_data_frames()[0]
    return df


@retry_with_backoff(max_attempts=5, base_delay=2.0)
def _fetch_team_game_log(team_id: int, season: str) -> pd.DataFrame:
    time.sleep(API_SLEEP)
    gl = teamgamelog.TeamGameLog(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    return gl.get_data_frames()[0]


@retry_with_backoff(max_attempts=5, base_delay=2.0)
def _fetch_league_game_log(season: str) -> pd.DataFrame:
    time.sleep(API_SLEEP)
    gl = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
    )
    return gl.get_data_frames()[0]


@retry_with_backoff(max_attempts=4, base_delay=2.0)
def _fetch_player_info(player_id: int) -> pd.DataFrame:
    time.sleep(API_SLEEP)
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    return info.get_data_frames()[0]


def get_player_game_logs(player_id: int, season: str, use_cache: bool = True) -> pd.DataFrame:
    """Return regular-season game logs for a player, cached as parquet."""
    ensure_dirs()
    cache_path = DATA_RAW / f"player_gamelog_{player_id}_{season}.parquet"
    if use_cache and cache_path.exists():
        return load_parquet(cache_path)

    df = _fetch_player_game_log(player_id, season)
    if df.empty:
        save_parquet(df, cache_path)
        return df

    df = df.copy()
    df["PLAYER_ID"] = player_id
    df["SEASON"] = season
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # Minutes often come as "MM:SS"
    df["MIN_FLOAT"] = df["MIN"].apply(_parse_minutes)
    df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False)
    df["OPPONENT"] = df["MATCHUP"].astype(str).str.replace(r".*(vs\.|@)\s*", "", regex=True)
    save_parquet(df, cache_path)
    return df


def get_team_schedule(team_id: int, season: str, use_cache: bool = True) -> pd.DataFrame:
    """Team schedule / game log used for rest-day and B2B features."""
    ensure_dirs()
    cache_path = DATA_RAW / f"team_schedule_{team_id}_{season}.parquet"
    if use_cache and cache_path.exists():
        return load_parquet(cache_path)

    df = _fetch_team_game_log(team_id, season)
    if not df.empty:
        df = df.copy()
        df["TEAM_ID"] = team_id
        df["SEASON"] = season
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE")
        df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(3).astype(int) - 1
        df["IS_BACK_TO_BACK"] = df["REST_DAYS"] == 0
    save_parquet(df, cache_path)
    return df


def get_league_player_game_logs(season: str = PRIMARY_SEASON, use_cache: bool = True) -> pd.DataFrame:
    """
    Bulk pull of all player-game rows for a season (preferred over per-player loops).
    Falls back gracefully if the endpoint flakes.
    """
    ensure_dirs()
    cache_path = DATA_RAW / f"league_player_gamelog_{season}.parquet"
    if use_cache and cache_path.exists():
        logger.info("Loading cached league game log for %s", season)
        return load_parquet(cache_path)

    logger.info("Fetching league player game log for %s (this can take a minute)...", season)
    df = _fetch_league_game_log(season)
    if df.empty:
        save_parquet(df, cache_path)
        return df

    df = df.copy()
    df["SEASON"] = season
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN_FLOAT"] = df["MIN"].apply(_parse_minutes)
    df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False)
    df["OPPONENT"] = df["MATCHUP"].astype(str).str.replace(r".*(vs\.|@)\s*", "", regex=True)
    # TEAM_ABBREVIATION is present; resolve TEAM_ID via static map
    team_map = {t["abbreviation"]: t["id"] for t in teams.get_teams()}
    df["TEAM_ID"] = df["TEAM_ABBREVIATION"].map(team_map)
    save_parquet(df, cache_path)
    logger.info("Cached %s player-game rows for %s", f"{len(df):,}", season)
    return df


def get_active_players_lookup(use_cache: bool = True) -> pd.DataFrame:
    """Static NBA player directory with ids and names."""
    ensure_dirs()
    cache_path = DATA_RAW / "players_lookup.parquet"
    if use_cache and cache_path.exists():
        return load_parquet(cache_path)

    rows = players.get_players()
    df = pd.DataFrame(rows)
    df = df.rename(columns={"id": "PLAYER_ID", "full_name": "PLAYER_NAME"})
    save_parquet(df, cache_path)
    return df


def get_teams_lookup() -> pd.DataFrame:
    rows = teams.get_teams()
    df = pd.DataFrame(rows).rename(
        columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"}
    )
    return df


@retry_with_backoff(max_attempts=4, base_delay=2.0)
def _fetch_team_roster(team_id: int, season: str) -> pd.DataFrame:
    time.sleep(API_SLEEP)
    from nba_api.stats.endpoints import commonteamroster

    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    return roster.get_data_frames()[0]


def get_player_ages(player_ids: Iterable[int], season: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Player ages from CommonTeamRoster (30 team calls) with CommonPlayerInfo fallback.
    Cached per season to avoid hammering the API.
    """
    ensure_dirs()
    cache_path = DATA_RAW / f"player_ages_{season}.parquet"
    if use_cache and cache_path.exists():
        cached = load_parquet(cache_path)
        missing = set(int(x) for x in player_ids) - set(cached["PLAYER_ID"].astype(int).tolist())
        if not missing:
            return cached
        base = cached
    else:
        base = pd.DataFrame(columns=["PLAYER_ID", "BIRTHDATE", "AGE"])
        missing = set(int(x) for x in player_ids)

    rows: list[dict] = []
    # Prefer roster endpoint — includes AGE directly
    for team in teams.get_teams():
        try:
            roster = _fetch_team_roster(team["id"], season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Roster fetch failed for %s: %s", team["abbreviation"], exc)
            continue
        if roster.empty:
            continue
        for _, r in roster.iterrows():
            pid = int(r["PLAYER_ID"])
            age = pd.to_numeric(r.get("AGE"), errors="coerce")
            birth = pd.to_datetime(r.get("BIRTH_DATE"), errors="coerce")
            rows.append({"PLAYER_ID": pid, "BIRTHDATE": birth, "AGE": float(age) if pd.notna(age) else None})

    roster_df = pd.DataFrame(rows)
    out = pd.concat([base, roster_df], ignore_index=True).drop_duplicates("PLAYER_ID", keep="last")

    still_missing = missing - set(out["PLAYER_ID"].dropna().astype(int).tolist())
    season_mid = pd.Timestamp(f"{int(season.split('-')[0]) + 1}-01-15")
    for i, pid in enumerate(sorted(still_missing)):
        try:
            info = _fetch_player_info(int(pid))
            if info.empty:
                continue
            birth = pd.to_datetime(info.iloc[0].get("BIRTHDATE"), errors="coerce")
            age = (season_mid - birth).days / 365.25 if pd.notna(birth) else None
            out = pd.concat(
                [out, pd.DataFrame([{"PLAYER_ID": int(pid), "BIRTHDATE": birth, "AGE": age}])],
                ignore_index=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch age for player %s: %s", pid, exc)
        if (i + 1) % 25 == 0:
            logger.info("Fetched fallback ages for %d / %d players...", i + 1, len(still_missing))

    out = out.drop_duplicates("PLAYER_ID", keep="last")
    save_parquet(out, cache_path)
    return out


def try_get_tracking_distance(season: str, use_cache: bool = True) -> pd.DataFrame | None:
    """
    Attempt to pull player tracking distance. Many tracking endpoints are
    rate-limited or gated — return None and let feature engineering drop the column.
    """
    ensure_dirs()
    cache_path = DATA_RAW / f"tracking_distance_{season}.parquet"
    if use_cache and cache_path.exists():
        return load_parquet(cache_path)

    try:
        from nba_api.stats.endpoints import leaguedashptstats

        time.sleep(API_SLEEP)
        endpoint = leaguedashptstats.LeagueDashPtStats(
            season=season,
            season_type_all_star="Regular Season",
            pt_measure_type="SpeedDistance",
            player_or_team="Player",
            per_mode_simple="PerGame",
        )
        df = endpoint.get_data_frames()[0]
        if df.empty:
            return None
        save_parquet(df, cache_path)
        logger.info("Cached tracking distance for %s (%d rows)", season, len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Tracking distance unavailable for %s (%s). "
            "distance_last_7d will be omitted from the model.",
            season,
            exc,
        )
        return None


def collect_season_workload(season: str = PRIMARY_SEASON) -> pd.DataFrame:
    """High-level helper: league game logs + team metadata for one season."""
    logs = get_league_player_game_logs(season)
    team_lu = get_teams_lookup()
    player_lu = get_active_players_lookup()

    # Prefer PLAYER_NAME from league log; fill from static lookup when missing
    if "PLAYER_NAME" not in logs.columns:
        logs = logs.merge(
            player_lu[["PLAYER_ID", "PLAYER_NAME"]],
            on="PLAYER_ID",
            how="left",
        )
    else:
        missing = logs["PLAYER_NAME"].isna() | (logs["PLAYER_NAME"].astype(str).str.len() == 0)
        if missing.any():
            fill = player_lu[["PLAYER_ID", "PLAYER_NAME"]].rename(columns={"PLAYER_NAME": "_PN"})
            logs = logs.merge(fill, on="PLAYER_ID", how="left")
            logs.loc[missing, "PLAYER_NAME"] = logs.loc[missing, "_PN"]
            logs = logs.drop(columns=["_PN"], errors="ignore")

    logs = logs.merge(
        team_lu[["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]],
        on="TEAM_ID",
        how="left",
        suffixes=("", "_lu"),
    )
    if "TEAM_ABBREVIATION_lu" in logs.columns:
        logs["TEAM_ABBREVIATION"] = logs["TEAM_ABBREVIATION"].fillna(logs["TEAM_ABBREVIATION_lu"])
        logs = logs.drop(columns=["TEAM_ABBREVIATION_lu"], errors="ignore")

    return logs


def _parse_minutes(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text or text.upper() in {"NONE", "N/A", "NAN"}:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    ensure_dirs()
    df = collect_season_workload(PRIMARY_SEASON)
    print(df.head())
    print(f"rows={len(df)}")
