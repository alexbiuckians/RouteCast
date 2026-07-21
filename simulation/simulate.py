"""
RouteCast — Phase 4: Dispatch Simulation (focused what-if)
=========================================================
 
ONE clear operational question:
    "During the peak hour, how does the number of couriers on shift affect
     delivery time in a busy dispatch area?"
 
Discrete-event simulation (SimPy), calibrated on REAL data:
  - Arrivals follow a Poisson process whose rate is fit to the observed
    peak-hour order volume in a busy dispatch AREA (a cluster of adjacent
    zones — real dispatch covers more than one 700 m hex).
  - Service (delivery) times are SAMPLED from the empirical delivery-time
    distribution in that area — no invented numbers.
  - Orders queue when all couriers are busy.
 
What is real vs modelled:
  REAL      -> arrival rate, delivery-time distribution (from LaDe)
  MODELLED  -> the specific arrival sequence + queue outcomes (the what-if)
This is standard operations-research methodology: real data calibrates a model
used to explore staffing counterfactuals that history alone cannot answer.
 
Insight: Phase 3 showed assignment optimization gives single-digit gains.
This phase shows STAFFING is the larger, more actionable lever.
 
Input:   data/clean/delivery_features.parquet
Output:  simulation/phase4_results.txt
         simulation/couriers_vs_delaytime.png
 
Run:     python simulation/simulate.py
"""
 
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT_ROOT / "data" / "clean" / "delivery_features.parquet"
REPORT = PROJECT_ROOT / "simulation" / "phase4_results.txt"
PNG = PROJECT_ROOT / "simulation" / "couriers_vs_delaytime.png"
 
PEAK_HOUR = 18            # 6pm dinner peak
AREA_RADIUS = 3          # cluster (2*r+1)^2 zones into one dispatch area
SIM_MINUTES = 60
N_RUNS = 30
COURIER_LEVELS = [3, 5, 8, 10, 12, 15, 20, 25, 30]
RNG = np.random.default_rng(42)
 
 
def calibrate(df):
    """Fit arrival rate + service-time sample from the busiest peak-hour AREA
    (a cluster of adjacent zones), not a single hex."""
    peak = df[df["hour"] == PEAK_HOUR].copy()
    rc = peak["zone"].str.split("_", expand=True)
    peak["zrow"] = pd.to_numeric(rc[0], errors="coerce")
    peak["zcol"] = pd.to_numeric(rc[1], errors="coerce")
    peak = peak.dropna(subset=["zrow", "zcol"])
 
    # Find the busiest area center by summing order counts in each zone's
    # (2r+1)x(2r+1) neighbourhood.
    counts = peak.groupby(["zrow", "zcol"]).size().rename("n").reset_index()
    best = None
    for _, row in counts.nlargest(50, "n").iterrows():  # search near dense zones
        cr, cc = row["zrow"], row["zcol"]
        area = peak[(peak["zrow"].between(cr - AREA_RADIUS, cr + AREA_RADIUS))
                    & (peak["zcol"].between(cc - AREA_RADIUS, cc + AREA_RADIUS))]
        if best is None or len(area) > best[1]:
            best = ((cr, cc), len(area), area)
    (cr, cc), n_area, area = best
 
    n_days = area["ds"].nunique()
    orders_per_hour = len(area) / max(n_days, 1)
    arrival_rate_per_min = orders_per_hour / 60.0
    delivery_samples = area["delivery_minutes"].values
    return (cr, cc), arrival_rate_per_min, delivery_samples, orders_per_hour, n_days
 
 
def run_sim(n_couriers, arrival_rate, delivery_samples, seed):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    couriers = simpy.Resource(env, capacity=n_couriers)
    times_in_system = []
 
    def order(env, t_arrival):
        with couriers.request() as req:
            yield req
            service = float(rng.choice(delivery_samples))
            yield env.timeout(service)
            times_in_system.append(env.now - t_arrival)
 
    def arrivals(env):
        while True:
            yield env.timeout(rng.exponential(1.0 / arrival_rate))
            if env.now > SIM_MINUTES:
                break
            env.process(order(env, env.now))
 
    env.process(arrivals(env))
    env.run(until=SIM_MINUTES + max(delivery_samples) + 10)
    return np.mean(times_in_system) if times_in_system else np.nan
 
 
def main():
    if not FEATURES.exists():
        raise SystemExit(f"Feature file not found: {FEATURES}\nRun build_features.py first.")
 
    df = pd.read_parquet(FEATURES)
    center, rate, samples, oph, n_days = calibrate(df)
 
    lines = []
    def log(m=""):
        print(m); lines.append(m)
 
    log("=" * 60)
    log("RouteCast Phase 4 — Peak-Hour Dispatch Simulation")
    log("=" * 60)
    block = 2 * AREA_RADIUS + 1
    log(f"\nDispatch area: {block}x{block}-zone cluster around {center}")
    log(f"Peak hour: {PEAK_HOUR}:00  |  days observed: {n_days}")
    log(f"Calibrated arrival rate: {rate:.2f} orders/min ({oph:.0f}/hour)")
    log(f"Delivery time (real): median {np.median(samples):.0f} min, "
        f"mean {np.mean(samples):.0f} min")
    log(f"Runs per staffing level: {N_RUNS}\n")
 
    results = []
    for n in COURIER_LEVELS:
        vals = [run_sim(n, rate, samples, seed=int(RNG.integers(1e9)))
                for _ in range(N_RUNS)]
        avg = np.nanmean(vals)
        results.append((n, avg))
        log(f"  {n:>2} couriers -> avg {avg:7.1f} min in system (wait + delivery)")
 
    log("\nMarginal effect of adding couriers:")
    for i in range(1, len(results)):
        n0, t0 = results[i-1]
        n1, t1 = results[i]
        log(f"  {n0:>2}->{n1:<2}: {t0-t1:+7.1f} min/order")
 
    # Identify the "elbow": smallest N within 10% of the best achievable time
    best_time = min(t for _, t in results)
    elbow = next(n for n, t in results if t <= best_time * 1.10)
    log(f"\nEfficient staffing level (within 10% of best): {elbow} couriers")
    log(f"  Understaffed (3) : {results[0][1]:.0f} min/order")
    log(f"  Efficient ({elbow}) : "
        f"{next(t for n,t in results if n==elbow):.0f} min/order")
 
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, "o-", color="#c05621", linewidth=2, markersize=8)
    plt.axvline(elbow, ls="--", color="gray", alpha=0.7,
                label=f"efficient level ≈ {elbow}")
    plt.xlabel("Couriers on shift (peak hour)")
    plt.ylabel("Avg time in system per order (min)")
    plt.title("Staffing vs. delivery time — peak-hour dispatch area")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG, dpi=140)
    log(f"\nChart saved: {PNG}")
    log("\nTakeaway: adding couriers sharply cuts delivery time while understaffed,")
    log("then flattens (diminishing returns). The simulation locates the efficient")
    log("peak-hour staffing level — a larger operational lever than assignment.")
 
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report saved: {REPORT}")
    log("Phase 4 complete.")
 
 
if __name__ == "__main__":
    main()