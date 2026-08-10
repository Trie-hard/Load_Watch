"""Load and clean NBA soft-tissue injury logs; fuzzy-match to nba_api player IDs."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import fuzz, process

from src.data_collection import get_active_players_lookup
from src.utils import (
    DATA_PROCESSED,
    DATA_RAW,
    PRIMARY_SEASON,
    ensure_dirs,
    is_soft_tissue,
    load_parquet,
    normalize_name,
    save_parquet,
    setup_logging,
)

logger = setup_logging("loadwatch.injury")

PROSPORTS_URL = (
    "https://www.prosportstransactions.com/basketball/Search/SearchResults.php"
)

# Extra soft-tissue-ish phrases common on official NBA injury reports
EXTRA_SOFT = (
    "soreness",
    "tightness",
    "inflammation",
    "impingement",
    "hyperextension",
    "sprain",
    "strain",
)


def load_injury_csv(path: Path) -> pd.DataFrame:
    """Load a user-supplied injury CSV with flexible column names."""
    df = pd.read_csv(path)
    colmap = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in {"player", "player_name", "name", "relinquished", "player name"}:
            colmap[c] = "player"
        elif cl in {"date", "injury_date", "datetime", "game date"}:
            colmap[c] = "injury_date"
        elif cl in {"injury", "injury_type", "notes", "note", "reason"}:
            colmap[c] = "injury_type"
        elif cl in {"games_missed", "games", "out_games"}:
            colmap[c] = "games_missed"
        elif cl in {"team", "team_name"}:
            colmap[c] = "team"
        elif cl in {"current status", "status"}:
            colmap[c] = "status"
    df = df.rename(columns=colmap)
    required = {"player", "injury_date", "injury_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Injury CSV missing columns: {missing}")
    if "games_missed" not in df.columns:
        df["games_missed"] = pd.NA
    return df


def _flip_last_first(name: str) -> str:
    """Convert 'Last, First' → 'First Last'."""
    name = re.sub(r"\s+", " ", str(name)).strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{first} {last}".strip()
    return name


def scrape_nba_injury_reports(
    begin_date: str = "2023-10-24",
    end_date: str = "2024-04-14",
    step_days: int = 2,
    report_hour: int = 14,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Sample official NBA pre-game injury reports via `nba-injury-report`.
    Much more reliable than third-party HTML scrapes.
    """
    ensure_dirs()
    cache_path = DATA_RAW / f"nba_injury_reports_{begin_date}_{end_date}_step{step_days}.parquet"
    if use_cache and cache_path.exists():
        logger.info("Loading cached NBA injury reports %s → %s", begin_date, end_date)
        return load_parquet(cache_path)

    try:
        from nba_injury_report import get_injury_report
    except ImportError as exc:
        raise ImportError(
            "Install nba-injury-report to fetch official injury reports: "
            "pip install nba-injury-report"
        ) from exc

    start = pd.Timestamp(begin_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq=f"{step_days}D")
    frames: list[pd.DataFrame] = []

    for i, day in enumerate(dates):
        ts = f"{day.strftime('%Y-%m-%d')}T{report_hour:02d}:00:00"
        try:
            report = get_injury_report(ts)
            df = report.to_dataframe()
            if df is None or df.empty:
                continue
            df = df.copy()
            df["report_timestamp"] = ts
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.debug("No report for %s (%s)", ts, exc)
        if (i + 1) % 20 == 0:
            logger.info("Fetched injury reports %d / %d days...", i + 1, len(dates))
        time.sleep(0.35)

    if not frames:
        logger.warning("No NBA injury reports retrieved for %s–%s", begin_date, end_date)
        empty = pd.DataFrame(
            columns=["injury_date", "team", "player", "injury_type", "status", "games_missed"]
        )
        save_parquet(empty, cache_path)
        return empty

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    out = pd.DataFrame(
        {
            "injury_date": pd.to_datetime(raw.get("Game Date"), errors="coerce"),
            "team": raw.get("Team"),
            "player": raw.get("Player Name").map(_flip_last_first),
            "injury_type": raw.get("Reason"),
            "status": raw.get("Current Status"),
            "games_missed": pd.NA,
            "report_timestamp": raw.get("report_timestamp"),
        }
    )
    out = out.dropna(subset=["injury_date", "player"])
    save_parquet(out, cache_path)
    logger.info("Cached %d NBA injury-report rows", len(out))
    return out


def scrape_prosports_injuries(
    begin_date: str = "2023-10-01",
    end_date: str = "2024-06-30",
    use_cache: bool = True,
    max_pages: int = 80,
) -> pd.DataFrame:
    """Optional Prosportstransactions scrape (often blocked with HTTP 403)."""
    ensure_dirs()
    cache_path = DATA_RAW / f"prosports_injuries_{begin_date}_{end_date}.parquet"
    if use_cache and cache_path.exists():
        logger.info("Loading cached Prosports injuries %s → %s", begin_date, end_date)
        return load_parquet(cache_path)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )

    rows: list[dict] = []
    start = 0
    for page in range(max_pages):
        params = {
            "Player": "",
            "Team": "",
            "BeginDate": begin_date,
            "EndDate": end_date,
            "InjuriesChkBx": "yes",
            "PersonalChkBx": "yes",
            "submit": "Search",
            "start": start,
        }
        try:
            resp = session.get(PROSPORTS_URL, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Prosports request failed at start=%s: %s", start, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", class_="datatable") or (
            soup.find_all("table")[0] if soup.find_all("table") else None
        )
        if table is None:
            break

        try:
            page_df = pd.read_html(StringIO(str(table)))[0]
        except ValueError:
            break

        if page_df.empty or len(page_df) <= 1:
            break

        page_df.columns = [str(c).strip() for c in page_df.columns]
        if "Date" in page_df.columns:
            page_df = page_df[page_df["Date"].astype(str).str.lower() != "date"]

        for _, r in page_df.iterrows():
            player = _clean_player_cell(str(r.get("Relinquished") or r.get("Acquired") or ""))
            notes = str(r.get("Notes", "") or "")
            if not player or player.lower() in {"nan", "none"}:
                continue
            rows.append(
                {
                    "injury_date": r.get("Date"),
                    "team": r.get("Team"),
                    "player": player,
                    "injury_type": notes,
                    "status": "Out",
                    "games_missed": pd.NA,
                }
            )

        if len(page_df) < 25:
            break
        start += 25
        time.sleep(1.0)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["injury_date"] = pd.to_datetime(df["injury_date"], errors="coerce")
        df = df.dropna(subset=["injury_date", "player"])
    else:
        df = pd.DataFrame(
            columns=["injury_date", "team", "player", "injury_type", "status", "games_missed"]
        )
    save_parquet(df, cache_path)
    logger.info("Cached %d Prosports injury rows", len(df))
    return df


def _clean_player_cell(text: str) -> str:
    text = re.sub(r"•\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_soft_tissue_extended(text: str) -> bool:
    if is_soft_tissue(text):
        return True
    tl = (text or "").lower()
    reason = tl.split("injury/illness", 1)[-1] if "injury/illness" in tl else tl
    if any(
        x in reason
        for x in (
            "fracture",
            "broken",
            "concussion",
            "contusion",
            "illness",
            "g league",
            "personal",
            "load management",
        )
    ):
        return False
    return any(k in reason for k in EXTRA_SOFT)


def filter_soft_tissue(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries is None or injuries.empty:
        return pd.DataFrame(
            columns=["injury_date", "team", "player", "injury_type", "status", "games_missed"]
        )

    df = injuries.copy()
    if "injury_type" not in df.columns:
        logger.warning("Injury frame missing injury_type; returning empty soft-tissue set.")
        return pd.DataFrame(
            columns=["injury_date", "team", "player", "injury_type", "status", "games_missed"]
        )

    df["injury_type"] = df["injury_type"].astype(str)
    notes_l = df["injury_type"].str.lower()
    is_return = notes_l.str.contains("returned|activated|recalled|cleared", regex=True, na=False)
    soft = df["injury_type"].apply(_is_soft_tissue_extended)

    # Prefer Out / Doubtful designations when status exists
    if "status" in df.columns:
        status_l = df["status"].astype(str).str.lower()
        serious = status_l.isin({"out", "doubtful"}) | status_l.str.contains("out|doubtful", na=False)
        # Keep soft-tissue Questionable only if clearly strain/sprain
        strong = notes_l.str.contains("strain|sprain|tear|rupture|hamstring|achilles|calf", na=False)
        keep_status = serious | strong
        out = df.loc[soft & ~is_return & keep_status].copy()
    else:
        out = df.loc[soft & ~is_return].copy()

    if "games_missed" not in out.columns:
        out["games_missed"] = pd.NA
    return out


def collapse_to_onset_events(injuries: pd.DataFrame, gap_days: int = 5) -> pd.DataFrame:
    """
    Injury reports repeat daily while a player is out. Collapse to onset dates:
    first report in a contiguous soft-tissue absence spell per player.
    """
    if injuries.empty:
        return injuries

    df = injuries.sort_values(["player", "injury_date"]).copy()
    events = []
    for player, grp in df.groupby("player", sort=False):
        grp = grp.sort_values("injury_date")
        last_date = None
        for _, row in grp.iterrows():
            d = row["injury_date"]
            if last_date is None or (d - last_date).days > gap_days:
                events.append(row)
            last_date = d
    return pd.DataFrame(events).reset_index(drop=True)


def match_players_to_ids(
    injuries: pd.DataFrame,
    players_df: Optional[pd.DataFrame] = None,
    score_cutoff: int = 84,
) -> pd.DataFrame:
    """Fuzzy-match injury player names to nba_api PLAYER_IDs."""
    if injuries.empty:
        out = injuries.copy()
        out["player_id"] = pd.Series(dtype="Int64")
        out["matched_name"] = pd.Series(dtype=object)
        out["match_score"] = pd.Series(dtype=int)
        return out

    if players_df is None:
        players_df = get_active_players_lookup()

    players_df = players_df.copy()
    players_df["norm"] = players_df["PLAYER_NAME"].astype(str).map(normalize_name)
    name_to_id = dict(zip(players_df["norm"], players_df["PLAYER_ID"]))
    # Also index by flipped forms
    choices = list(name_to_id.keys())

    matched_ids = []
    matched_names = []
    scores = []

    for raw in injuries["player"].astype(str):
        norm = normalize_name(raw)
        if norm in name_to_id:
            matched_ids.append(name_to_id[norm])
            matched_names.append(raw)
            scores.append(100)
            continue
        hit = process.extractOne(norm, choices, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= score_cutoff:
            matched_ids.append(name_to_id[hit[0]])
            matched_names.append(hit[0])
            scores.append(int(hit[1]))
        else:
            best = max(
                choices,
                key=lambda c: SequenceMatcher(None, norm, c).ratio(),
                default=None,
            )
            ratio = SequenceMatcher(None, norm, best).ratio() if best else 0
            if best and ratio >= score_cutoff / 100:
                matched_ids.append(name_to_id[best])
                matched_names.append(best)
                scores.append(int(ratio * 100))
            else:
                matched_ids.append(pd.NA)
                matched_names.append(None)
                scores.append(0)

    out = injuries.copy()
    out["player_id"] = matched_ids
    out["matched_name"] = matched_names
    out["match_score"] = scores
    matched = out.dropna(subset=["player_id"])
    logger.info(
        "Matched %d / %d injury rows to player IDs (cutoff=%d)",
        len(matched),
        len(out),
        score_cutoff,
    )
    return out


def estimate_games_missed(injuries: pd.DataFrame) -> pd.DataFrame:
    df = injuries.copy()
    if df.empty:
        df["games_missed"] = pd.Series(dtype=int)
        return df
    if "games_missed" not in df.columns:
        df["games_missed"] = pd.NA
    df["games_missed"] = pd.to_numeric(df["games_missed"], errors="coerce")
    df["games_missed"] = df["games_missed"].fillna(3).astype(int)
    return df


def _season_date_window(season: str) -> tuple[str, str]:
    year = int(season.split("-")[0])
    return f"{year}-10-24", f"{year + 1}-04-15"


def build_clean_injury_table(
    season: str = PRIMARY_SEASON,
    local_csv: Path | None = None,
) -> pd.DataFrame:
    """
    End-to-end injury cleaning.
    Output columns: player_id, injury_date, injury_type, games_missed (+ helpers)
    """
    ensure_dirs()
    out_path = DATA_PROCESSED / f"injuries_clean_{season}.parquet"

    if local_csv and Path(local_csv).exists():
        raw = load_injury_csv(Path(local_csv))
        raw["injury_date"] = pd.to_datetime(raw["injury_date"], errors="coerce")
    else:
        begin, end = _season_date_window(season)
        raw = scrape_nba_injury_reports(begin_date=begin, end_date=end, step_days=2)
        if raw.empty:
            logger.warning("NBA injury reports empty; trying Prosports fallback...")
            year = int(season.split("-")[0])
            raw = scrape_prosports_injuries(
                begin_date=f"{year}-10-01",
                end_date=f"{year + 1}-06-30",
            )

    soft = filter_soft_tissue(raw)
    soft = collapse_to_onset_events(soft)
    soft = estimate_games_missed(soft)
    matched = match_players_to_ids(soft)
    clean = matched.dropna(subset=["player_id"]).copy()

    if clean.empty:
        logger.warning("No matched soft-tissue injuries for %s — labels will be all zeros.", season)
        final = pd.DataFrame(
            columns=[
                "player_id",
                "injury_date",
                "injury_type",
                "games_missed",
                "player",
                "team",
                "match_score",
            ]
        )
    else:
        clean["player_id"] = clean["player_id"].astype(int)
        clean = clean.sort_values(["player_id", "injury_date"]).drop_duplicates(
            subset=["player_id", "injury_date"], keep="first"
        )
        final = clean[
            [
                "player_id",
                "injury_date",
                "injury_type",
                "games_missed",
                "player",
                "team",
                "match_score",
            ]
        ].reset_index(drop=True)

    save_parquet(final, out_path)
    final.to_csv(DATA_PROCESSED / f"injuries_clean_{season}.csv", index=False)
    logger.info("Wrote clean injury table: %d rows → %s", len(final), out_path)
    return final


if __name__ == "__main__":
    df = build_clean_injury_table(PRIMARY_SEASON)
    print(df.head(20))
    print(len(df))
