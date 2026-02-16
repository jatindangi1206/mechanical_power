#!/usr/bin/env python3
"""
Extract ICU data from the configured source (MIMIC-IV / Amsterdam / HiRID).

Usage:
    python scripts/extract_data.py --config config/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.helpers import load_config, set_seed, setup_logging, ensure_dirs
from src.data.extraction import get_extractor


def main():
    parser = argparse.ArgumentParser(description="Extract ICU ventilation data")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["logs"])
    set_seed(config["project"]["random_seed"])
    ensure_dirs(config)

    output_dir = Path(config["paths"]["raw_data"])

    logger.info(f"Data source: {config['data']['source']}")
    extractor = get_extractor(config)

    tables = extractor.extract_all()

    # Save each table as parquet
    for name, df in tables.items():
        out_path = output_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {name}: {len(df)} rows → {out_path}")

    logger.info("Data extraction complete.")


if __name__ == "__main__":
    main()
