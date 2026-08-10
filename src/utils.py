"""Shared paths, constants, and helpers for LoadWatch."""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Default seasons: completed seasons relative to Aug 2026 context
DEFAULT_SEASONS = ("2023-24", "2024-25")
PRIMARY_SEASON = "2023-24"

SOFT_TISSUE_KEYWORDS = (
    "strain",
    "sprain",
    "tendin",
    "tendonitis",
    "tendinitis",
    "hamstring",
    "quad",
    "groin",
    "calf",
    "achilles",
    "plantar",
    "fasciitis",
    "adductor",
    "hip flexor",
    "oblique",
    "glute",
    "soleus",
    "gastroc",
    "acl",
    "mcl",
    "lcl",
    "pcl",
    "meniscus",
    "rotator",
    "labrum",
    "muscle",
    "soft tissue",
)

CONTACT_EXCLUSIONS = (
    "fracture",
    "broken",
    "break",
    "concussion",
    "contusion",
    "laceration",
    "surgery",
    "illness",
    "covid",
    "rest",
    "personal",
    "g-league",
    "assignment",
    "suspension",
    "load management",
    "maintenance",
)

FEATURE_COLUMNS = [
    "minutes_last_7d",
    "minutes_last_14d",
    "games_last_7d",
    "is_back_to_back",
    "rest_days",
    "season_minutes_cumulative",
    "age",
    "days_since_last_injury",
]

T = TypeVar("T")


def ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "loadwatch", level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 1.5,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff for flaky NBA API calls."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    sleep_s = base_delay * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def normalize_name(name: str) -> str:
    """Normalize player names for matching across sources."""
    if not isinstance(name, str):
        return ""
    text = name.strip()
    # Official reports often use "Last, First"
    if "," in text:
        last, first = [p.strip() for p in text.split(",", 1)]
        text = f"{first} {last}"
    cleaned = (
        text.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )
    parts = [p for p in cleaned.split() if p and p not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(parts)


def is_soft_tissue(injury_text: str) -> bool:
    text = (injury_text or "").lower()
    if not text:
        return False

    # Strip the boilerplate prefix used on official NBA reports
    reason = text
    if "injury/illness" in reason:
        reason = reason.split("injury/illness", 1)[-1]
    reason = reason.lstrip(" -–—:")

    hard_exclusions = (
        "fracture",
        "broken",
        "break",
        "concussion",
        "contusion",
        "laceration",
        "covid",
        "personal",
        "g-league",
        "g league",
        "assignment",
        "suspension",
        "load management",
        "maintenance",
        "return to competition",
        "not with team",
    )
    if any(ex in reason for ex in hard_exclusions):
        return False
    # Pure illness / rest (after stripping Injury/Illness prefix)
    if reason.strip() in {"illness", "n/a; illness", "rest"} or reason.endswith("; illness"):
        return False
    if "surgery" in reason and not any(
        k in reason for k in ("strain", "sprain", "tendin", "achilles", "meniscus", "ligament")
    ):
        return False
    return any(k in reason for k in SOFT_TISSUE_KEYWORDS)


def season_to_year_start(season: str) -> int:
    return int(season.split("-")[0])
