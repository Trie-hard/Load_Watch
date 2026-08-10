"""End-to-end LoadWatch pipeline: data → features → models → SHAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import compute_shap_top_factors
from src.model import train_and_score
from src.utils import PRIMARY_SEASON, ensure_dirs, setup_logging

logger = setup_logging("loadwatch.pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LoadWatch training pipeline")
    parser.add_argument("--season", default=PRIMARY_SEASON, help="NBA season string, e.g. 2023-24")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Force rebuild of feature matrix even if parquet exists",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation")
    args = parser.parse_args()

    ensure_dirs()
    logger.info("=== LoadWatch pipeline | season=%s ===", args.season)
    meta = train_and_score(season=args.season, rebuild_features=args.rebuild_features)
    logger.info("Training complete: %s", meta.get("metrics"))

    if not args.skip_shap:
        logger.info("Computing SHAP explanations...")
        compute_shap_top_factors(season=args.season)
        logger.info("SHAP complete.")

    logger.info("Done. Launch dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
