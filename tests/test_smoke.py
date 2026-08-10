"""Lightweight import / artifact checks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    import src.data_collection  # noqa: F401
    import src.injury_data  # noqa: F401
    import src.feature_engineering  # noqa: F401
    import src.model  # noqa: F401
    import src.explain  # noqa: F401
    import src.utils  # noqa: F401


def test_processed_artifacts_exist():
    processed = ROOT / "data" / "processed"
    assert (processed / "scored_2023-24.parquet").exists() or (
        processed / "scored_2023-24.csv"
    ).exists()
