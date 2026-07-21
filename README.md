# RouteCast — Last-Mile Delivery Dispatch Intelligence
 
RouteCast is an end-to-end operations-research and machine-learning system for last-mile parcel delivery. It forecasts delivery times with calibrated uncertainty, optimizes courier-to-order assignment, and simulates peak-hour staffing — built entirely on **real industrial data**, not synthetic data.
<<<<<<< HEAD
 
**Live dashboard:** https://alexbiuckians.github.io/RouteCast/dashboard/
 
=======

**DASHBOARD**:https://alexbiuckians.github.io/RouteCast/dashboard/

>>>>>>> 36cd8a2b6e97d8a6abff7947b2711ad9ebdec2df
## The dataset
 
**Source:** [Cainiao-AI/LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) — the first comprehensive last-mile delivery dataset from industry (Alibaba Cainiao), released under Apache-2.0. Paper: [arXiv:2306.10675](https://arxiv.org/abs/2306.10675)
 
This project uses the **package-delivery** records across three high-volume cities — Shanghai, Hangzhou, Chongqing: **472,419 raw orders**, cleaned to **445,295** with a documented audit.
 
> **Domain note.** LaDe is *parcel* last-mile delivery, not hot-food delivery. Delivery times reflect a courier's full handling window (receipt → sign-off), so median times are ~90 min rather than the ~30 min typical of food apps.
 
> **Coordinate note.** LaDe positions are a *projected metric grid* (values in the millions = meters), not WGS84 degrees. Spatial features are therefore binned in native meters — no datum assumptions, correct metric distances.
 
## Pipeline
 
```
ingestion/   download.py         Reproducible dataset fetch from Hugging Face
             audit_clean.py      Quality audit, ETA label, cleaning
features/    build_features.py   Grid zones, distance, temporal, demand density
models/      train_eta.py        Quantile ETA models (P10/P50/P90)
             feature_importance.py
optimizer/   optimize.py         Hungarian vs greedy assignment
             optimize_radius.py  Benefit vs dispatch radius
simulation/  simulate.py         Discrete-event peak-hour staffing sim
dashboard/   index.html          Self-contained results dashboard
```
 
Reproducible from scratch:
```bash
pip install -r requirements.txt
python ingestion/download.py         # fetches LaDe from source (~2.88 GB)
python ingestion/audit_clean.py
python features/build_features.py
python models/train_eta.py
python models/feature_importance.py
python optimizer/optimize.py
python optimizer/optimize_radius.py
python simulation/simulate.py
# then open dashboard/index.html
```
 
## Results
 
### Phase 1 — Clean geospatial foundation
- **445,295** clean orders (94.3% retention; 27,124 implausible-duration rows removed)
- **8,395** metric zones (~53 orders each), binned in native meters
- Delivery time spans **P10 = 29 min to P90 = 229 min** — a wide, right-skewed spread that motivates quantile prediction over a single point estimate
- Distance median ~2.0 km; demand density up to 34 orders per zone per 15 min
> **On spatial indexing.** H3 hexagonal indexing was evaluated for zone binning,  but LaDe's coordinates use a non-standard projection that does not cleanly convert to WGS84 degrees — candidate coordinate transforms were tested and none placed points in the correct cities. Metric-grid binning was therefore the correct, validated choice for this dataset.
 
### Phase 2 — Calibrated quantile ETA model
Three LightGBM quantile models predict delivery time at P10/P50/P90.
 
| Metric | Result |
|---|---|
| P50 median error (MAE) | **44.7 min** — 32% better than mean-baseline (65.8) |
| P90 calibration | **88.9%** of actual deliveries covered (target 90%) |
| P10 calibration | 10.8% covered (target 10%) |
| P10–P90 band | contains the actual time 78% of the time |
 
**Leakage-free by construction:** courier- and zone-level historical averages are computed on the training split only, then mapped onto test. This is what makes the 44.7-min result trustworthy rather than inflated.
 
**Feature importance:** the two engineered history features —
`zone_avg_min` (46%) and `courier_avg_min` (17%) — account for **~63%** of the model's predictive gain, confirming that *where* a parcel is going and *who* carries it dominate raw distance.
 
### Phase 3 — Assignment optimization
The Phase 2 predictions feed a cost matrix; the Hungarian algorithm (`scipy.linear_sum_assignment`) finds the provably-optimal courier-to-order assignment, compared against a greedy nearest-first baseline.
 
| Dispatch radius | Optimal vs greedy |
|---|---|
| ~0.7 km (single zone) | 1.2% |
| ~2.0 km (3×3 zones) | 2.4% |
| ~3.5 km (5×5 zones) | 3.4% |
 
**Finding:** the optimizer's advantage *grows with dispatch radius* — negligible when couriers are tightly clustered (greedy is already near-optimal), larger when supply is dispersed and greedy's local choices create pile-ups. This points to staffing, not assignment, as the larger operational lever.
 
### Phase 4 — Peak-hour staffing simulation
A discrete-event simulation (SimPy) of the busiest peak-hour dispatch area, calibrated on real demand and delivery-time distributions.
 
| Couriers on shift | Avg time per order |
|---|---|
| 3 (understaffed) | 87 min |
| 10 (efficient) | 55 min |
| 15+ | flat (~53 min) |
 
**Result:** staffing the peak-hour hotspot at **~10 couriers cuts average order time from 87 to 55 min (a 37% reduction)**; beyond that, added couriers yield negligible benefit. The efficient level is located automatically at the curve's elbow.
 
> **Real vs modelled:** arrival rate and delivery-time distribution come from LaDe (real); the arrival sequence and queue outcomes are simulated. This is standard operations-research methodology — real data calibrating a model used to explore staffing counterfactuals that history alone cannot answer.
 
## Tech stack
Python · pandas · LightGBM · SciPy · SimPy · matplotlib
 
## License
<<<<<<< HEAD
Code: MIT. Data: Apache-2.0 ([Cainiao-AI/LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe)), not redistributed here — the ingestion script fetches it directly from source. 
=======
Code: MIT. Data: Apache-2.0 (Cainiao-AI/LaDe), not redistributed here — the ingestion script fetches it directly from source.
https://huggingface.co/datasets/Cainiao-AI/LaDe
>>>>>>> 36cd8a2b6e97d8a6abff7947b2711ad9ebdec2df
