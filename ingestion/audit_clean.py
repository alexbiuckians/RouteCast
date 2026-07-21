"""
RouteCast — Phase 1: Audit & Clean
==================================

Reads the raw LaDe delivery data, audits its quality, computes the delivery-
time (ETA) label, filters implausible records, and writes a clean dataset.

This produces the Phase 1 "Clean Geo Dataset" milestone:
a documented, deduplicated, order-level dataset reproducible from one script.

Input:   data/raw_full/delivery_five_cities.csv
Output:  data/clean/delivery_clean.parquet
         data/clean/audit_report.txt

Run:     python ingestion/audit_clean.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw_full" / "delivery_five_cities.csv"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CLEAN_PARQUET = CLEAN_DIR / "delivery_clean.parquet"
REPORT_PATH = CLEAN_DIR / "audit_report.txt"

# Records with delivery times outside this range are treated as data errors.
# LaDe times are minute-of-day style ("MM-DD HH:MM:SS"); adjust after inspection.
MIN_MINUTES = 1        # deliveries faster than 1 min are implausible
MAX_MINUTES = 8 * 60   # deliveries longer than 8 hours are implausible

# City name normalization (Chinese -> readable English label)
CITY_MAP = {
    "上海市": "Shanghai",
    "重庆市": "Chongqing",
    "杭州市": "Hangzhou",
    "吉林市": "Jilin",
    "烟台市": "Yantai",
}


def parse_time(series: pd.Series, year: int = 2022) -> pd.Series:
    """LaDe timestamps look like '03-18 07:31:58' with no year. Attach a year."""
    return pd.to_datetime(
        str(year) + "-" + series.astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def main() -> None:
    lines = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("RouteCast Phase 1 — Data Audit & Clean")
    log("=" * 60)

    if not RAW_CSV.exists():
        raise SystemExit(f"Raw file not found: {RAW_CSV}\nRun ingestion/download.py first.")

    df = pd.read_csv(RAW_CSV)
    n_raw = len(df)
    log(f"\nRaw records: {n_raw:,}")
    log(f"Columns: {df.shape[1]}")

    # --- Missing-data audit --------------------------------------------------
    log("\nMissing-data rate per column:")
    for col, rate in df.isna().mean().round(4).items():
        flag = "  <-- fully empty, will drop" if rate == 1.0 else ""
        log(f"  {col:<18} {rate:>6.1%}{flag}")

    # Drop columns that are 100% empty (sign_lng / sign_lat in this dataset)
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log(f"\nDropped fully-empty columns: {empty_cols}")

    # --- Deduplicate ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset="order_id")
    log(f"\nDuplicate order_ids removed: {before - len(df):,}")

    # --- Parse times and compute the ETA label -------------------------------
    df["receipt_dt"] = parse_time(df["receipt_time"])
    df["sign_dt"] = parse_time(df["sign_time"])

    bad_times = df[["receipt_dt", "sign_dt"]].isna().any(axis=1).sum()
    log(f"Rows with unparseable receipt/sign time: {bad_times:,}")
    df = df.dropna(subset=["receipt_dt", "sign_dt"])

    # Delivery time in minutes = sign - receipt  (this is the ETA target)
    df["delivery_minutes"] = (df["sign_dt"] - df["receipt_dt"]).dt.total_seconds() / 60.0

    # --- Filter implausible durations ----------------------------------------
    before = len(df)
    df = df[(df["delivery_minutes"] >= MIN_MINUTES) & (df["delivery_minutes"] <= MAX_MINUTES)]
    log(f"Rows dropped for implausible duration "
        f"(<{MIN_MINUTES}min or >{MAX_MINUTES}min): {before - len(df):,}")

    # --- Normalize city labels -----------------------------------------------
    df["city"] = df["from_city_name"].map(CITY_MAP).fillna(df["from_city_name"])

    # --- Summary stats -------------------------------------------------------
    log("\n" + "-" * 60)
    log(f"Clean records: {len(df):,}  ({len(df) / n_raw:.1%} of raw)")
    log("\nDelivery-time (minutes) distribution:")
    desc = df["delivery_minutes"].describe(percentiles=[0.1, 0.5, 0.9]).round(1)
    for k, v in desc.items():
        log(f"  {k:<8} {v:>10}")

    log("\nRecords per city:")
    for city, cnt in df["city"].value_counts().items():
        log(f"  {city:<12} {cnt:>8,}")

    # --- Write outputs -------------------------------------------------------
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    keep = [
        "order_id", "city", "delivery_user_id",
        "poi_lng", "poi_lat", "receipt_lng", "receipt_lat",
        "aoi_id", "from_dipan_id", "typecode",
        "receipt_dt", "sign_dt", "delivery_minutes", "ds",
    ]    
    df[keep].to_parquet(CLEAN_PARQUET, index=False)
    log(f"\nClean dataset written: {CLEAN_PARQUET}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Audit report written:  {REPORT_PATH}")
    log("\nPhase 1 milestone complete.")


if __name__ == "__main__":
    main()