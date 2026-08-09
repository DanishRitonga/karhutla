# Karhutla — Caveats & Known Limitations

**Date:** 2026-08-09 | Consolidation of verified limitations across the data pipeline, labels, and model. Each item maps to a design-log leakage/limitation clause where relevant.

---

## 1. Label source under-represents peatland fires (design-log L3) — THE key caveat

The system's ground truth is **FIRMS VIIRS 375 m hotspots** (confidence {n,h}, k=2 in a 7-day
window). Peat is only 660/3,598 Riau cells (18%), and the *actual* positive label cell-days on
peat are a small minority:

| Year | pos cell-days | on-peat |
|---|---|---|
| 2019 | 21,652 | 22% |
| 2020 | 4,411 | 22% |
| 2021 | 3,345 | 16% |
| 2022 | 3,192 | **3%** |
| 2023 | 4,941 | **7%** |

Consequences:
- The model learns **"VIIRS-detectable fire risk," not "peatland fire risk."** Alert rate on
  peat cells (21.8%) ≈ non-peat (23.9%) on 2023-09-25 — the model is faithful to its labels,
  and those labels are dominated by mineral/agricultural burns in most years.
- **Deep peat smoldering is the known blind spot**: smoldering combustion (~500–700 °C
  surface signal, below MODIS/VIIRS thresholds), burns at ~39 cm average depth (Whitburn 2016),
  is invisible until it escapes, and is systematically undercounted at 375 m–1 km resolution
  (van Wees 2022; Page 2009). The EWS cannot alert on a fire VIIRS cannot see.
- Paper must state explicitly: performance is measured against "hotspot detectable by VIIRS
  375 m" — the same operational benchmark an EWS would be held to — and the deep-peat
  blind spot is a documented residual risk, not an overclaim.

## 2. Class imbalance is extreme

~0.3% positive cell-days (train 2019–2022: 32,600 pos of ~11.8 M eligible). Handled via
stratified sampling (`--pos-frac`), BCEWithLogitsLoss `pos_weight`, and PR-AUC as the primary
metric (ROC-AUC reported only as secondary — a ~1% base rate makes ROC optimistic). Accuracy
is deliberately not reported (misleading under imbalance).

## 3. The 20k vs 50k sample-size discrepancy (do not mix numbers)

The training agent's early runs used `--n-train 20000`; the canonical paper numbers are the
**50k runs** in `outputs_tabonly/`. They differ materially:
- 20k op LightGBM: PR-AUC 0.774; 50k op LightGBM: **0.712**.
- 20k env LightGBM: 0.606; 50k env LightGBM: 0.503.

The 20k draw was a favorable sample; larger N converges to the true (harder) distribution.
**Use only 50k values in the paper**; never mix with agent test1–test5 tables.

## 4. Seasonal-matching results are a diagnostic, not a config

1:1 seasonal negative matching (Sinato & Rivas 2026) collapses LightGBM in the env regime
(0.477→0.305) — the "glorified calendar" effect. This is reported as evidence that trees
exploit seasonal/calendar proxies, NOT as an alternative training protocol. The production
system uses random sampling. Do not present seasonal numbers as if they are the system's
expected performance.

## 5. Sensor/data-quality caveats (design-log L2, O4, L10)

- **Dynamic World cloud sparsity**: DW is Sentinel-2-derived; Riau is among the world's
  cloudiest regions. 3 severely sparse months found: 2022-02, 2022-10, and **2023-01 (0 Riau
  cells)**. DW missingness is handled by 0-filling + a `dw_available` mask channel (dropped
  in the model's 22-channel remap). Test-year January has no DW signal — evaluate that month
  with the mask caveat in mind.
- **Sentinel-1 swath/revisit sparsity**: S1 is swath-limited (single-date coverage is
  partial); forward-filled daily with gap cap = 14 days. Gaps > 14 days are deliberately
  NaN-invalidated and masked via `sar_available`. DESCENDING orbit is the weaker signal
  (25.3% invalidated vs ASC 2.2%).
- **ERA5-Land missing cells**: 1,149 grid cells fall outside the ERA5 0.1° footprint;
  spatially filled from the nearest valid cell (min 5 km, median 10 km). 26 of these are
  Riau cells inside the training region. Also, ERA5-Land precipitation is wet-biased
  (~14.5× CHIRPS magnitude) and can exceed physical plausibility on isolated days; the
  normalization absorbs scale, but magnitudes must never be quoted as truth.
- **CHIRPS vs ERA5 precip correlation**: Pearson 0.37 (Spearman 0.52–0.55) — correlated but
  not redundant; both retained as channels. `sat` variant chosen specifically to avoid
  ERA5-derived signal duplication (design-log L10).
- **FIRMS label noise**: hotspot false positives over smoldering peat and cloud-shadow
  confusion are mitigated by confidence ≥ nominal + k=2 persistence, but residual noise
  remains. FRP distribution must be reported (design-log §5).

## 6. Grid/tile artifacts

- **Patch margin**: model features are 15×15 patches; cells within 7 of the grid edge were
  originally unscored (263 Riau cells). Fixed in `risk_map.py` (commit 3d7b03c) by spatial
  edge-padding, but note the model was *trained* only on margin-safe cells — edge predictions
  use replicated-boundary patches and are slightly out-of-training-distribution.
- **is_riau = cell-center-in-polygon**: 5 km cells whose center falls outside the Riau
  boundary are feature-only (no label). Sea cells legitimately lack CHIRPS (464 cells).

## 7. Temporal leakage controls (design-log L1–L10, adhered to)

- Train 2019–2021 / val 2022 / test 2023 — no temporal overlap.
- Fire-history channel derived from RAW FIRMS daily counts over the previous T_IN=14 days
  (inclusive), never from k=2 label windows (that would leak `(t, t+7]` labels backward).
- Features use only `t-13..t`; labels only `t+1..t+7`.
- Normalization stats computed on train only, applied to val/test.

## 8. Scope limits (design-log §9, §15)

- The model is a **per-cell spatiotemporal classification**, not full-map fire spread
  forecasting; it does not model fire-front propagation.
- LLM/agentic layer (Phase-5 optional) is excluded from the core science and evaluated
  separately (citation precision + action clarity) — no leakage into the prediction pipeline.
- Study area is Riau only; cross-province generalization (Sumsel, Kalbar) is untested.
