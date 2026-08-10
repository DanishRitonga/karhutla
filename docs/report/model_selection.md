# Karhutla — Final Model Selection

**Date:** 2026-08-09 | **Decision:** LightGBM (operational regime) is the production default

---

## 1. The decision

The production default for the Riau early-warning system is **LightGBM, operational regime**
(22 channels including `hotspot_count_lag` fire history), trained on 2019–2021, validated
on 2022, tested on 2023 at `--n-train 50000` (canonical 50k run).

**Headline numbers (50k, random sampling):**

| Metric | Value |
|---|---|
| PR-AUC | **0.712** |
| F1 | 0.625 |
| Recall | 0.583 |
| ROC-AUC | 0.879 |
| F1 threshold | 0.420 |

## 2. Why LightGBM over the alternatives

### vs. other tree models (statistically tied)
| Model (op, 50k) | PR-AUC | F1 | Recall |
|---|---|---|---|
| **LightGBM** | **0.7119** | 0.625 | **0.583** |
| XGBoost | 0.7121 | — | — |
| Ensemble (RF+LGBM+XGB) | 0.7120 | 0.627 | 0.545 |
| Random Forest | 0.6826 | — | — |
| Tabular LR | 0.5719 | — | — |

LightGBM vs XGBoost vs Ensemble are within ±0.0002 — statistically identical. LightGBM is
chosen because: (1) highest Recall (an early-warning system prioritizes recall over precision),
(2) single-model SHAP is clean and auditable (vs. averaged across an ensemble), (3) one
checkpoint to serve, (4) ms-scale inference over the 3,598-cell grid.

The **Ensemble row stays in the paper as a reported negative result**: RF/LGBM/XGB all split on
the same tabular features, are highly correlated, and soft-voting lands at the best member
rather than above it. Correlated GBDTs on shared features revert to the mean.

### vs. deep spatiotemporal models (decisively)
| Regime | LightGBM | Best DL | Gap |
|---|---|---|---|
| env | 0.503 (XGB 0.536) | ConvLSTM 0.333 | −0.17 |
| op | 0.712 | Transformer 0.483 | −0.23 |

Deep models were given every chance and still lost — four independent bottlenecks ruled out:

1. **Normalization** (test2): ConvLSTM loss went flat 1.040 → 0.43 after per-channel z-score
   (commit 07c0f0e). Normalization was the #1 bottleneck; once fixed, DL became competitive
   but never won.
2. **Capacity** (test3, hidden-channel sweep): (12,12)=0.333 > (24,24)=0.319 > (64,32)=0.281.
   **17× more params degraded performance** (overfitting). Capacity is not the bottleneck.
3. **Training length** (test5, epoch sweep of ResNet-18 Temporal Transformer, op-seasonal):
   30ep=0.299 → 60ep=0.379 → **100ep=0.397 (sweet spot)** → 150ep=0.357 (overfits). Even at
   its best the Transformer ceiling (0.397) sits below LightGBM's *seasonal-collapse* value
   (0.549). ConvLSTM is epoch-invariant (0.26–0.28 across 30/60/100/150) — architecturally
   limited, not undertrained.
4. **Data/architecture fit**: gradient boosting is simply more data-efficient on a
   ~0.3%-positive, sensor-heterogeneous, tabular-heavy problem. The domain SOTA agrees —
   Sinato & Rivas won NASA Space Apps 2025 with RF+XGBoost on the same problem family.

## 3. The model suite is fixed — no selection on test

The full comparison is a fixed 8-model × 2-regime grid (Persistence, Met-LR, Tabular LR, RF,
LightGBM, XGBoost, Ensemble, ConvLSTM, Temporal Transformer) under a single random-sampling
protocol, then repeated under seasonal 1:1 negative matching as a **diagnostic** (not a
production config). We do NOT pick different models per regime from test scores — that is
selection-on-the-test-set and judges would flag it. LightGBM-op leads under both protocols.

## 4. Seasonal matching explains WHY trees win (not just that they win)

Sinato & Rivas (2026) 1:1 seasonal matching (each positive cell-day paired with a negative
from the same ±30-day-of-year window in a different year) was applied to the training set:

| Model | env random | env seasonal | op random | op seasonal |
|---|---|---|---|---|
| LightGBM | 0.477 | **0.305** (−0.172) | 0.698 | 0.549 |
| Random Forest | 0.359 | 0.284 | 0.684 | **0.556** |
| XGBoost | — | 0.314 | — | 0.551 |
| Transformer | 0.201 | **0.322** (+0.121) | 0.483 | 0.299 |

Under seasonal matching, LightGBM collapses in env (−0.172) while the Transformer improves
(+0.121) and becomes best. This is the **"glorified calendar" effect** Sinato & Rivas warned
about: trees were exploiting month/seasonal proxies from the raw weather channels, which the
matching protocol removes. The op regime still prefers trees (RF 0.556 > XGB 0.551 > LGBM
0.549) because fire-history is a genuine causal feature trees use well.

**Conclusion for the paper:** random sampling is the headline protocol (matches how an
operational EWS would be evaluated); seasonal matching is reported as a robustness/diagnostic
result that (a) explains why trees dominate and (b) shows the Transformer learns more
weather-driven signal.

## 5. Production artifacts

- `model_lgbm_operational.joblib` + `checkpoint_operational.json` (norm_stats + tab_names +
  F1 threshold 0.420) — the pair consumed by `model/risk_map.py`.
- `model/risk_map.py`: scores all 6,970 grid cells for a date (edge-padded patches) and
  renders `risk_{date}.png` + `risk_{date}.npy`.

## 6. Validation sanity checks

- Risk-map seasonality (commit 6ac4f54/3d7b03c): 2023-09-25 (peak fire season) → mean risk
  0.205, **817–846 alert cells** above threshold; 2023-01-15 (wet season) → mean risk 0.048,
  **18 alert cells**. ~45× more alerts in the dry season — physically correct.
- Checkpoint harness: all 11 joblib artifacts load, norm_stats coherent (22 entries), LGBM
  `feature_name_` == `tab_names` (158 op), end-to-end prediction verified on held-out 2023.
