"""
RouteCast — Phase 2b: Feature Importance
========================================

Loads the trained P50 ETA model and reports which of the 11 features drive
its predictions. Confirms whether the engineered features (courier_avg_min,
zone_avg_min) carry the signal — a useful README insight.

Input:   models/eta_p50.txt
Output:  models/feature_importance.txt
         models/feature_importance.png

Run:     python models/feature_importance.py
"""

from pathlib import Path
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "eta_p50.txt"
OUT_TXT = PROJECT_ROOT / "models" / "feature_importance.txt"
OUT_PNG = PROJECT_ROOT / "models" / "feature_importance.png"

# Must match the feature order used in train_eta.py
FEATURE_COLS = [
    "dist_m", "demand_density", "courier_load",
    "hour_sin", "hour_cos", "dayofweek", "is_weekend", "is_rush",
    "city_code", "courier_avg_min", "zone_avg_min",
]


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun models/train_eta.py first.")

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    imp = (
        pd.DataFrame({"feature": FEATURE_COLS, "gain": gain, "splits": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(1)

    lines = ["Feature importance (P50 model), by gain:\n"]
    for _, r in imp.iterrows():
        lines.append(f"  {r['feature']:<16} {r['gain_pct']:>5}%   (splits: {int(r['splits'])})")
    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report, encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"][::-1], imp["gain_pct"][::-1], color="#2b6cb0")
    plt.xlabel("Importance (% of total gain)")
    plt.title("What drives the ETA prediction (P50 model)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"\nChart saved: {OUT_PNG}")
    print(f"Report saved: {OUT_TXT}")


if __name__ == "__main__":
    main()