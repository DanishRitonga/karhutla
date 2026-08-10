# Karhutla — Transformer Training Beyond 30 Epochs (op, seasonal)

**Date:** 2026-08-09 | **Samples:** 20k train / 5k val / 20k test | **Balance:** seasonal | **Regime:** operational | **Model:** Temporal Transformer (d_model=256)

## Motivation
At 30 epochs the Transformer scored 0.299 PR-AUC (seasonal op). Question: was it still improving at epoch 30? Prior runs (random balance) showed a long plateau then a late escape. Tested 60 epochs to observe the trajectory.

## Results — 60 Epochs

### Transformer (op, seasonal)
| Epochs | Test PR-AUC | F1 | Recall | ROC-AUC | Best Val PR-AUC | Best Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 30 | 0.299 | 0.335 | 0.459 | 0.664 | — | — |
| **60** | **0.379** | 0.409 | 0.534 | 0.745 | **0.483** | 59 |

### Trajectory (val PR-AUC per epoch)
- **Epochs 1–32:** oscillates 0.23–0.35 — the long plateau/escape pattern
- **Epochs 33–60:** climbs to 0.33–0.48; **best checkpoint at epoch 59/60 (0.483)**

The Transformer had **not converged at epoch 60** — best epoch was the second-to-last, and the curve was still rising. Extended training directly improves it.

### Full table (60 ep, op, seasonal)
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Persistence | 0.183 | 0.271 | 1.000 | 0.548 | 0.316 |
| Met-LR | 0.234 | 0.305 | 0.513 | 0.625 | 0.298 |
| Tab-LR | 0.495 | 0.437 | 0.351 | 0.750 | 0.807 |
| RF | 0.556 | 0.572 | 0.536 | 0.793 | 0.504 |
| LightGBM | 0.549 | 0.530 | 0.526 | 0.804 | 0.531 |
| XGBoost | 0.551 | 0.532 | 0.541 | 0.801 | 0.475 |
| ConvLSTM | 0.261 | 0.350 | 0.470 | 0.646 | 0.837 |
| **Transformer** | **0.379** | 0.409 | 0.534 | 0.745 | 0.913 |

## Key Findings
1. **Longer training helps the Transformer decisively** in op + seasonal: 0.299 → 0.379 test PR-AUC (+0.080) when going 30 → 60 epochs.
2. **Still not converged at 60.** Best val epoch was 59; the curve had no plateau at 60. Further gains expected at 100+ epochs (in progress).
3. **ConvLSTM did not share the effect** — best at epoch 39 (val 0.310), then degraded; final test 0.261 (vs 0.287 at 30 ep). The plateau-then-escape is Transformer-specific.
4. The late escape (epochs ~30–60) resembles the random-balance 30-epoch run where the Transformer only escaped around epoch 14 — seasonal matching (harder counterfactuals) appears to delay the escape.

## Extended Runs — 100 & 150 Epochs

### Transformer (op, seasonal) — full sweep
| Epochs | Test PR-AUC | F1 | Recall | ROC-AUC | Best Val PR-AUC | Best Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 30 | 0.299 | 0.335 | 0.459 | 0.664 | — | — |
| 60 | 0.379 | 0.409 | 0.534 | 0.745 | 0.483 | 59 |
| **100** | **0.397** | **0.423** | 0.452 | 0.713 | **0.498** | 86 |
| 150 | 0.357 | 0.409 | 0.550 | 0.721 | 0.487 | 133 |

### Full tables (op, seasonal)

**100 epochs** (`outputs/seasonal_100ep/comparison_table_operational.csv`)
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Persistence | 0.183 | 0.271 | 1.000 | 0.548 | 0.316 |
| Met-LR | 0.234 | 0.305 | 0.513 | 0.625 | 0.298 |
| Tab-LR | 0.495 | 0.437 | 0.351 | 0.750 | 0.807 |
| RF | 0.556 | 0.572 | 0.536 | 0.793 | 0.504 |
| LightGBM | 0.549 | 0.530 | 0.526 | 0.804 | 0.531 |
| XGBoost | 0.551 | 0.532 | 0.541 | 0.801 | 0.475 |
| ConvLSTM | 0.264 | 0.358 | 0.482 | 0.637 | 0.975 |
| **Transformer** | **0.397** | 0.423 | 0.452 | 0.713 | 0.904 |

**150 epochs** (`outputs/seasonal_150ep/comparison_table_operational.csv`)
| Model | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|-------|:---:|:---:|:---:|:---:|:---:|
| Persistence | 0.183 | 0.271 | 1.000 | 0.548 | 0.316 |
| Met-LR | 0.234 | 0.305 | 0.513 | 0.625 | 0.298 |
| Tab-LR | 0.495 | 0.437 | 0.351 | 0.750 | 0.807 |
| RF | 0.556 | 0.572 | 0.536 | 0.793 | 0.504 |
| LightGBM | 0.549 | 0.530 | 0.526 | 0.804 | 0.531 |
| XGBoost | 0.551 | 0.532 | 0.541 | 0.801 | 0.475 |
| ConvLSTM | 0.275 | 0.325 | 0.349 | 0.619 | 0.856 |
| Transformer | 0.357 | 0.409 | 0.550 | 0.721 | 0.004 |

## Key Findings (extended runs)
1. **100 epochs is the sweet spot.** Test PR-AUC peaks at **0.397** (+0.098 over 30 ep, +0.018 over 60 ep). Val PR-AUC saturates at ~0.48–0.50 (best 0.498 @ ep86).
2. **Overfitting sets in by 150.** Test PR-AUC drops to 0.357 — training past ~130 epochs degrades the checkpoint even though val holds at 0.487.
3. **The late-escape pattern is consistent:** plateau ~0.25–0.35 for the first 30–40 epochs, then escape into the 0.42–0.50 range. Seasonal matching delays the escape vs random balance (~14 ep).
4. **ConvLSTM unaffected by epoch count** (~0.26–0.28 across 30/60/100/150) — its ceiling is architectural/data-limited, not training-limited.

CSV: `outputs/seasonal_60ep/comparison_table_operational.csv`, `outputs/seasonal_100ep/comparison_table_operational.csv`, `outputs/seasonal_150ep/comparison_table_operational.csv`.
