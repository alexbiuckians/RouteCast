"""
RouteCast — Phase 3: Courier-to-Order Assignment Optimizer
==========================================================

At a dispatch moment we have a batch of pending orders in a LOCAL AREA and a
set of couriers active in that same area. Each (courier, order) pairing has a
COST in MINUTES:

    cost = predicted_delivery_time  +  travel_time(courier -> pickup)

  - predicted_delivery_time: Phase 2 P50 ETA model
  - travel_time: (courier->pickup meters) / courier speed

We minimize TOTAL cost across the batch with the Hungarian algorithm
(provably optimal) and compare against a greedy nearest-first baseline.

LOCAL DISPATCH (important): batches are drawn PER ZONE, not city-wide. Real
dispatch only considers couriers near an order, so sampling couriers from the
same zone keeps travel distances realistic (sub-km) and makes the assignment
problem meaningful. City-wide sampling would put couriers hundreds of km away
and swamp the cost with noise.

Formulation is one-batch, one-order-per-courier. Batching multiple orders per
courier (min-cost flow / vehicle routing) is documented future work.

Input:   data/clean/delivery_features.parquet
         models/eta_p50.txt
Output:  optimizer/phase3_results.txt

Run:     python optimizer/optimize.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT_ROOT / "data" / "clean" / "delivery_features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "eta_p50.txt"
REPORT = PROJECT_ROOT / "optimizer" / "phase3_results.txt"

FEATURE_COLS = [
    "dist_m", "demand_density", "courier_load",
    "hour_sin", "hour_cos", "dayofweek", "is_weekend", "is_rush",
    "city_code", "courier_avg_min", "zone_avg_min",
]

BATCH_SIZE = 6                    # orders per local dispatch batch
N_BATCHES = 300                   # batches to average over
COURIER_SPEED_M_PER_MIN = 250.0   # ~15 km/h city courier
MIN_POOL = BATCH_SIZE * 2         # a zone needs this many orders to form a batch
RNG = np.random.default_rng(42)


def prep(df):
    df = df.copy()
    df["city"] = df["city"].astype("category")
    df["city_code"] = df["city"].cat.codes
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    load = (df.groupby(["delivery_user_id", "time_window"]).size()
            .rename("courier_load").reset_index())
    df = df.merge(load, on=["delivery_user_id", "time_window"], how="left")
    g = df["delivery_minutes"].mean()
    df["courier_avg_min"] = df.groupby("delivery_user_id")["delivery_minutes"].transform("mean").fillna(g)
    df["zone_avg_min"] = df.groupby("zone")["delivery_minutes"].transform("mean").fillna(g)
    return df


def build_cost_matrix(model, batch, couriers):
    """cost[i,j] = predicted delivery time + travel time, both in MINUTES."""
    n_orders = len(batch)
    cost = np.zeros((n_orders, len(couriers)))
    for j, (_, c) in enumerate(couriers.iterrows()):
        rows = batch.copy()
        rows["courier_avg_min"] = c["courier_avg_min"]
        rows["courier_load"] = c["courier_load"]
        delivery_min = model.predict(rows[FEATURE_COLS])
        dist_m = np.sqrt(
            (batch["poi_lat"].values - c["poi_lat"]) ** 2
            + (batch["poi_lng"].values - c["poi_lng"]) ** 2
        )
        travel_min = dist_m / COURIER_SPEED_M_PER_MIN
        cost[:, j] = delivery_min + travel_min
    return cost


def greedy_cost(cost):
    m = cost.shape[1]
    used, total = set(), 0.0
    for i in np.argsort(cost.min(axis=1)):
        c, j = min((cost[i, j], j) for j in range(m) if j not in used)
        used.add(j)
        total += c
    return total


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun models/train_eta.py first.")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    df = prep(pd.read_parquet(FEATURES))

    lines = []
    def log(m=""):
        print(m); lines.append(m)

    log("=" * 60)
    log("RouteCast Phase 3 — Assignment Optimizer (local dispatch)")
    log("=" * 60)
    log(f"\nBatch size: {BATCH_SIZE} | batches: {N_BATCHES} | "
        f"speed: {COURIER_SPEED_M_PER_MIN:.0f} m/min | per-zone sampling")

    # Only zones with enough orders to form a batch + courier pool
    zone_counts = df["zone"].value_counts()
    eligible_zones = zone_counts[zone_counts >= MIN_POOL].index.to_numpy()
    log(f"Eligible zones (>= {MIN_POOL} orders): {len(eligible_zones):,}")

    opt_totals, greedy_totals, travel_frac = [], [], []
    for _ in range(N_BATCHES):
        z = RNG.choice(eligible_zones)
        pool = df[df["zone"] == z]
        idx = RNG.choice(len(pool), size=BATCH_SIZE * 2, replace=False)
        sample = pool.iloc[idx]
        batch = sample.iloc[:BATCH_SIZE]
        couriers = sample.iloc[BATCH_SIZE:].reset_index()

        cost = build_cost_matrix(model, batch, couriers)
        r, c = linear_sum_assignment(cost)
        opt_totals.append(cost[r, c].sum())
        greedy_totals.append(greedy_cost(cost))

    opt = np.mean(opt_totals)
    grd = np.mean(greedy_totals)
    improvement = (grd - opt) / grd

    log("\n" + "=" * 60)
    log("RESULTS  (avg total cost per batch, in minutes)")
    log("=" * 60)
    log(f"\nGreedy nearest-first : {grd:.1f} min")
    log(f"Optimal (Hungarian)  : {opt:.1f} min")
    log(f"Improvement          : {improvement:.1%} less total cost")
    log(f"\nPer order: greedy {grd/BATCH_SIZE:.1f} min  vs  optimal {opt/BATCH_SIZE:.1f} min")
    log(f"Saved: {(grd-opt)/BATCH_SIZE:.1f} min per order on average")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport saved: {REPORT}")
    log("Phase 3 complete.")


if __name__ == "__main__":
    main()