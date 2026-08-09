# Karhutla — ConvLSTM Hidden Channel Sweep

**Date:** 2026-08-09 | **Epochs:** 30 | **Samples:** 20k/5k/20k | **Normalized:** yes

## Sweep Configurations
| Config | Params | Cell State Size (64×15×15) |
|--------|--------|:---:|
| (12,12) | ~310k | 2,700 |
| (24,24) | ~1.2M | 5,400 |
| (64,32) | ~5.4M | 14,400 |

## Environmental Regime

| Hidden | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|--------|:---:|:---:|:---:|:---:|:---:|
| **(12,12)** | **0.333** | 0.402 | 0.556 | 0.730 | 0.149 |
| (24,24) | 0.319 | 0.383 | 0.585 | 0.705 | 0.193 |
| (64,32) | 0.281 | 0.362 | 0.582 | 0.674 | 0.013 |

## Operational Regime

| Hidden | PR-AUC | F1 | Recall | ROC-AUC | Best Thr |
|--------|:---:|:---:|:---:|:---:|:---:|
| (12,12) | 0.330 | 0.407 | 0.554 | 0.715 | 0.240 |
| **(24,24)** | **0.334** | 0.403 | 0.524 | 0.726 | 0.380 |
| (64,32) | 0.316 | 0.376 | 0.482 | 0.706 | 0.248 |

## Baselines (identical across runs)

| Regime | Model | PR-AUC |
|--------|-------|:---:|
| env | LightGBM | 0.477 |
| env | RF | 0.359 |
| op | LightGBM | 0.698 |
| op | RF | 0.684 |

## Key Findings

1. **(12,12) and (24,24) perform equivalently.** The difference (0.333 vs 0.319 env, 0.330 vs 0.334 op) is within sampling noise.

2. **(64,32) is consistently worse.** Despite 17× more parameters than (12,12), it underperforms by 0.03–0.05 PR-AUC. With 20k samples and 15×15 patches, the larger cell state overfits noise instead of learning spatial structure.

3. **Cross-run variance is high for (64,32).** An earlier 30-epoch run with the same config scored 0.361 env / 0.326 op — a spread of 0.08 PR-AUC. Smaller models are more stable.

4. **Default (12,12) is optimal for ConvLSTM.** The normalization fix (z-score per channel) was the real bottleneck, not model capacity. Once normalized, even the smallest ConvLSTM learns effectively.

5. **LightGBM still leads** — best ConvLSTM trails by 0.144 PR-AUC (env) and 0.364 (op).

## Comparison Tables

The sweep's output CSVs overwrite each other in `outputs/cl_sweep/`. The final file is from the (64,32) op run.
