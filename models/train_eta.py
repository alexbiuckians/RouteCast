"""
RouteCast — Phase 2: Quantile ETA Model
=======================================

Trains three LightGBM quantile-regression models to predict delivery time
(delivery_minutes) at the 10th, 50th, and 90th percentiles.

Why quantile regression: delivery time ranges from ~29 min (P10) to ~229 min
(P90), so a single point estimate is nearly useless. Predicting a BAND
(P10-P90) plus a median (P50) tells a dispatcher both the typical time and a
realistic worst case.

Features (11), including three higher-signal engineered features:
  - courier_avg_min : each courier's mean delivery time (TRAIN-only)
  - zone_avg_min    : each zone's mean delivery time (TRAIN-only)
  - courier_load    : orders that courier handles in the same 15-min window
  - hour_sin/hour_cos : cyclical hour-of-day encoding
  - plus dist_m, demand_density, dayofweek, is_weekend, is_rush, city_code

Leakage control: courier/zone averages are computed on training rows only,
then mapped onto test rows. Test delivery times never influence the features
used to predict them; unseen couriers/zones fall back to the global mean.

Result (held-out test): P50 MAE 44.7 min (32% better than mean-baseline);
P90 predictions cover 88.9% of actual deliveries (well-calibrated).

Input:   data/clean/delivery_features.parquet
Output:  models/eta_p10.txt / p50 / p90
         models/phase2_results.txt

Run:     python models/train_eta.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT_ROOT / "data" / "clean" / "delivery_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT = MODEL_DIR / "phase2_results.txt"

TARGET = "delivery_minutes"
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

FEATURES_LIST = [
    "dist_m", "demand_density", "courier_load",
    "hour_sin", "hour_cos", "dayofweek", "is_weekend", "is_rush",
    "city_code", "courier_avg_min", "zone_avg_min",
]


def add_group_mean(train_df, test_df, key, target, new_col, global_mean):
    """Per-group mean of `target` computed on TRAIN only; mapped onto both
    splits. Prevents leakage: test target values never enter the feature."""
    means = train_df.groupby(key)[target].mean()
    train_df[new_col] = train_df[key].map(means).fillna(global_mean)
    test_df[new_col] = test_df[key].map(means).fillna(global_mean)


def main():
    if not FEATURES.exists():
        raise SystemExit(f"Feature file not found: {FEATURES}\nRun build_features.py first.")

    lines = []
    def log(m=""):
        print(m); lines.append(m)

    log("=" * 60)
    log("RouteCast Phase 2 — Quantile ETA Model")
    log("=" * 60)

    df = pd.read_parquet(FEATURES)
    log(f"\nRecords: {len(df):,}")

    df["city"] = df["city"].astype("category")
    df["city_code"] = df["city"].cat.codes

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Courier load: orders per courier per 15-min window
    load = (
        df.groupby(["delivery_user_id", "time_window"]).size()
        .rename("courier_load").reset_index()
    )
    df = df.merge(load, on=["delivery_user_id", "time_window"], how="left")

    # Split FIRST, then build leakage-prone features on train only
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = train_df[TARGET].mean()

    add_group_mean(train_df, test_df, "delivery_user_id", TARGET, "courier_avg_min", global_mean)
    add_group_mean(train_df, test_df, "zone", TARGET, "zone_avg_min", global_mean)

    log(f"Train: {len(train_df):,}   Test: {len(test_df):,}")
    log(f"Features ({len(FEATURES_LIST)}): {FEATURES_LIST}")

    X_train, y_train = train_df[FEATURES_LIST], train_df[TARGET]
    X_test, y_test = test_df[FEATURES_LIST], test_df[TARGET]

    preds = {}
    for name, q in QUANTILES.items():
        log(f"\nTraining {name} (quantile={q})...")
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        model.booster_.save_model(str(MODEL_DIR / f"eta_{name}.txt"))

    p10, p50, p90 = preds["p10"], preds["p50"], preds["p90"]

    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    mae = mean_absolute_error(y_test, p50)
    base = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
    log(f"\nP50 MAE: {mae:.1f} min   (mean-baseline {base:.1f})")
    log(f"  Improvement over baseline: {(1 - mae/base):.1%}")

    cov10 = float(np.mean(y_test.values <= p10))
    cov90 = float(np.mean(y_test.values <= p90))
    log(f"\nCalibration:")
    log(f"  P10 covers {cov10:.1%}  (target ~10%)")
    log(f"  P90 covers {cov90:.1%}  (target ~90%)")

    in_band = float(np.mean((y_test.values >= p10) & (y_test.values <= p90)))
    log(f"\nP10-P90 band contains actual: {in_band:.1%}")
    log(f"Average band width: {np.mean(p90 - p10):.0f} min")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nSaved to: {MODEL_DIR}")
    log("Phase 2 complete.")


if __name__ == "__main__":
    main()