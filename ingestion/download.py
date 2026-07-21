"""
RouteCast — Data Ingestion
==========================

Downloads the LaDe last-mile delivery dataset (Cainiao-AI/LaDe) from its
authoritative Hugging Face source into ./data/raw_full.

This makes the project reproducible from a single command: anyone who
clones the repo runs this script and gets the exact same data from source.
No data is committed to Git — only this script and the dataset link.

Source:  https://huggingface.co/datasets/Cainiao-AI/LaDe
License:  Apache-2.0
Paper:    https://arxiv.org/abs/2306.10675

Usage:
    python ingestion/download.py            # full dataset (~2.88 GB)
    python ingestion/download.py --core     # delivery + pickup + roads only (~730 MB)
"""

import argparse
import sys
from pathlib import Path

REPO_ID = "Cainiao-AI/LaDe"
# Project root = one level up from this script's /ingestion folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_full"

# Files needed for the core RouteCast phases (ETA model, optimizer, simulation).
# The courier GPS trajectory file is excluded here because no phase reads it;
# use the full download (default) if you want the complete source locally.
CORE_PATTERNS = [
    "delivery/*",
    "pickup/*",
    "road-network/*",
    "*.csv",  # the pre-combined five-cities CSVs
]


def download(core_only: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )
    if core_only:
        kwargs["allow_patterns"] = CORE_PATTERNS
        print("Downloading CORE subset (delivery + pickup + road network)...")
    else:
        print("Downloading FULL LaDe dataset (~2.88 GB)...")

    print(f"Destination: {RAW_DIR}")
    snapshot_download(**kwargs)
    print("\n\u2713 Download complete.")
    print(f"Data is in: {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the LaDe dataset for RouteCast.")
    parser.add_argument(
        "--core",
        action="store_true",
        help="Download only the files the project uses (~730 MB) instead of the full 2.88 GB.",
    )
    args = parser.parse_args()
    download(core_only=args.core)