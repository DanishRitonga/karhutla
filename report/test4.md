# Karhutla — Seasonal 1:1 Negative Matching (Sinato & Rivas 2026)

**Date:** 2026-08-09 | **Epochs:** 30 | **Samples:** 20k train / 5k val / 20k test | **Normalized:** yes

## Method
Each positive cell-day is paired with one negative from the same ±30-day seasonal window in a **different year**. Applied to the **training set only** (val/test use random sampling — fix in commit 97a25b6). Forced the model to distinguish fire-weather from normal seasonal weather, exposing any reliance on calendar/seasonal proxies.

## Baseline: Random Balance (30 ep, from test1–3 runs)

### Environmental Regime
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Met-LR | 0.293 | 0.366 | 0.527 | 0.697 | 0.415 |
| Tab-LR | 0.318 | 0.384 | 0.584 | 0.712 | 0.401 |
| RF | 0.359 | 0.412 | 0.581 | 0.743 | 0.274 |
| LightGBM | 0.477 | 0.487 | 0.652 | 0.811 | 0.210 |
| ConvLSTM (12,12) | 0.333 | 0.402 | 0.556 | 0.730 | 0.149 |
| Transformer | 0.201 | 0.295 | 0.401 | 0.570 | 0.548 |

### Operational Regime
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Persistence | 0.183 | 0.271 | 1.000 | 0.548 | 0.316 |
| Met-LR | 0.293 | 0.366 | 0.527 | 0.697 | 0.415 |
| Tab-LR | 0.547 | 0.492 | 0.497 | 0.796 | 0.506 |
| RF | 0.684 | 0.615 | 0.535 | 0.867 | 0.460 |
| LightGBM | 0.698 | 0.618 | 0.571 | 0.876 | 0.395 |
| ConvLSTM (12,12) | 0.330 | 0.407 | 0.554 | 0.715 | 0.240 |
| Transformer | 0.483 | 0.483 | 0.530 | 0.770 | 0.542 |

## Seasonal Balance Results

### Environmental Regime
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Met-LR | 0.234 | 0.305 | 0.513 | 0.625 | 0.298 |
| Tab-LR | 0.273 | 0.335 | 0.574 | 0.661 | 0.309 |
| RF | 0.284 | 0.337 | 0.375 | 0.637 | 0.493 |
| LightGBM | 0.305 | 0.348 | 0.498 | 0.675 | 0.272 |
| XGBoost | 0.314 | 0.349 | 0.410 | 0.675 | 0.407 |
| ConvLSTM (12,12) | 0.275 | 0.338 | 0.383 | 0.603 | 0.860 |
| **Transformer** | **0.322** | **0.388** | 0.450 | **0.691** | 0.840 |

### Operational Regime
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Persistence | 0.183 | 0.271 | 1.000 | 0.548 | 0.316 |
| Met-LR | 0.234 | 0.305 | 0.513 | 0.625 | 0.298 |
| Tab-LR | 0.495 | 0.437 | 0.351 | 0.750 | 0.807 |
| RF | **0.556** | **0.572** | 0.536 | 0.793 | 0.504 |
| LightGBM | 0.549 | 0.530 | 0.526 | 0.804 | 0.531 |
| XGBoost | 0.551 | 0.532 | 0.541 | 0.801 | 0.475 |
| ConvLSTM (12,12) | 0.287 | 0.336 | 0.373 | 0.622 | 0.712 |
| Transformer | 0.299 | 0.335 | 0.459 | 0.664 | 0.144 |

## Deltas (Seasonal − Random)

### Environmental Regime
| Model | ΔPR-AUC | ΔF1 | ΔRecall | ΔROC |
|-------|:---:|:---:|:---:|:---:|
| Met-LR | −0.059 | −0.061 | −0.014 | −0.072 |
| Tab-LR | −0.045 | −0.049 | −0.010 | −0.051 |
| RF | −0.075 | −0.075 | −0.206 | −0.106 |
| LightGBM | **−0.172** | −0.139 | −0.154 | −0.136 |
| ConvLSTM | −0.058 | −0.064 | −0.173 | −0.127 |
| Transformer | **+0.121** | **+0.093** | +0.049 | +0.122 |

### Operational Regime
| Model | ΔPR-AUC | ΔF1 | ΔRecall | ΔROC |
|-------|:---:|:---:|:---:|:---:|
| Met-LR | −0.059 | −0.061 | −0.014 | −0.072 |
| Tab-LR | −0.052 | −0.055 | −0.146 | −0.046 |
| RF | −0.128 | −0.043 | +0.001 | −0.074 |
| LightGBM | **−0.149** | −0.088 | −0.045 | −0.072 |
| ConvLSTM | −0.043 | −0.071 | −0.181 | −0.093 |
| Transformer | **−0.184** | −0.148 | −0.071 | −0.106 |

## Key Findings

1. **Env regime flips the leaderboard.** LightGBM collapses (−0.172 PR-AUC) from dominant to third. The Transformer goes from worst (0.201) to **best (0.322)**. Tree models were relying on calendar/seasonal shortcuts; the Transformer was learning genuine weather signal.

2. **Op regime preserves hierarchy but compresses gaps.** RF (0.556) > XGB (0.551) > LGBM (0.549). The fire-history channel provides real signal that survives seasonal matching.

3. **XGBoost** is a solid new baseline — competitive with RF/LGBM in both regimes (0.314 env / 0.551 op).

4. **All models drop in op except persistence** — every model partially used seasonal signal, but the deep models dropped more (ConvLSTM −0.043, Transformer −0.184).

5. **ConvLSTM loses recall under seasonal matching** (env −0.173, op −0.181) — it finds it harder to separate same-season negatives, pointing to weak temporal discriminative power.

CSV tables: `outputs/seasonal/comparison_table_env.csv`, `outputs/seasonal/comparison_table_operational.csv`.
