"""
RouteCast — Phase 1: Geospatial Feature Engineering (grid version)
==================================================================

Enriches the clean delivery dataset with spatial + temporal features for
the Phase 2 ETA model.

IMPORTANT — coordinate system:
LaDe's poi_lng/poi_lat and receipt_lng/receipt_lat are NOT WGS84 degrees;
they are provided in a PROJECTED METRIC grid (values in the millions =
meters). So we bin in native meters and compute distance in meters directly.
This is correct and honest: no datum assumptions required.

The zone assignment is isolated in `assign_zone()`. To upgrade to H3 later,
only that one function changes (convert meters -> lat/lng, then h3.latlng_to_cell).

Features produced:
  1. zone (coarse) + zone_fine   -> square metric cells ("which area?")
  2. dist_m                       -> receipt -> delivery, straight-line meters
  3. hour / dayofweek / is_weekend / is_rush  -> temporal
  4. demand_density               -> orders per zone per 15-min window

Input:   data/clean/delivery_clean.parquet
Output:  data/clean/delivery_features.parquet

Run:     python features/build_features.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = PROJECT_ROOT / "data" / "clean" / "delivery_clean.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "clean" / "delivery_features.parquet"

# Grid cell sizes in meters. 700 m ~ a neighbourhood; 200 m ~ a block.
CELL_COARSE_M = 700
CELL_FINE_M = 200

DEMAND_WINDOW = "15min"
RUSH_HOURS = set(range(11, 14)) | set(range(17, 20))


# --- Zone assignment (the one piece that changes if you swap to H3) ---------
def assign_zone(lat_m, lng_m, cell_m):
    """Snap projected-meter coordinates to a square grid cell of size cell_m.

    Returns a string 'row_col' zone id. To switch to H3 later, replace the
    body with:  lat, lng = meters_to_latlng(lat_m, lng_m);
                return h3.latlng_to_cell(lat, lng, resolution)
    """
    row = np.floor(lat_m / cell_m).astype(int)
    col = np.floor(lng_m / cell_m).astype(int)
    return row.astype(str) + "_" + col.astype(str)


def main():
    if not CLEAN_PARQUET.exists():
        raise SystemExit(
            f"Clean file not found: {CLEAN_PARQUET}\n"
            "Run ingestion/audit_clean.py first."
        )

    print("Loading clean dataset...")
    df = pd.read_parquet(CLEAN_PARQUET)
    print(f"  {len(df):,} records")

    # --- 1. Grid zones (delivery destination) --------------------------------
    print(f"Assigning grid zones (coarse={CELL_COARSE_M}m, fine={CELL_FINE_M}m)...")
    df["zone"] = assign_zone(df["poi_lat"].values, df["poi_lng"].values, CELL_COARSE_M)
    df["zone_fine"] = assign_zone(df["poi_lat"].values, df["poi_lng"].values, CELL_FINE_M)

    # --- 2. Distance in meters (receipt -> delivery) -------------------------
    if {"receipt_lat", "receipt_lng"}.issubset(df.columns):
        print("Computing straight-line distance (meters)...")
        df["dist_m"] = np.sqrt(
            (df["poi_lat"] - df["receipt_lat"]) ** 2
            + (df["poi_lng"] - df["receipt_lng"]) ** 2
        )
    else:
        print("  receipt coords missing — re-run audit_clean.py with them in `keep`.")
        df["dist_m"] = np.nan

    # --- 3. Temporal features ------------------------------------------------
    print("Engineering temporal features...")
    t = pd.to_datetime(df["receipt_dt"])
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush"] = df["hour"].isin(RUSH_HOURS).astype(int)

    # --- 4. Demand density (orders per zone per 15-min window) ---------------
    print("Computing demand density...")
    df["time_window"] = t.dt.floor(DEMAND_WINDOW)
    demand = (
        df.groupby(["zone", "time_window"]).size()
        .rename("demand_density").reset_index()
    )
    df = df.merge(demand, on=["zone", "time_window"], how="left")

    # --- Summary (sanity checks) ---------------------------------------------
    print("\n" + "-" * 55)
    print(f"Unique coarse zones: {df['zone'].nunique():,}   "
          f"(avg {len(df)/df['zone'].nunique():.1f} orders/zone)")
    print(f"Unique fine zones:   {df['zone_fine'].nunique():,}")
    print(f"Demand density - mean {df['demand_density'].mean():.1f}, "
          f"median {df['demand_density'].median():.0f}, "
          f"max {df['demand_density'].max():.0f}")
    if df["dist_m"].notna().any():
        print(f"Distance (m)   - median {df['dist_m'].median():.0f}, "
              f"90th pct {df['dist_m'].quantile(0.9):.0f}")
    print(f"Rush-hour share: {df['is_rush'].mean():.1%}")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nFeatures written: {OUT_PARQUET}")
    print(f"Columns now: {df.shape[1]}")
    print("Phase 1 feature engineering complete.")


if __name__ == "__main__":
    main()