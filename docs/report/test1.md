# Karhutla — Model Architecture Sweep Report

**Date:** 2026-08-08 | **GPU:** RTX 5070 Ti (16 GB) | **Samples:** 20k train / 5k val / 20k test | **Epochs:** 15

## Architecture Changes (models.py + train.py)

| Component | Baseline | Upgraded |
|-----------|----------|----------|
| ConvLSTM `hidden_channels` | (24, 24) | (64, 32) |
| Transformer `d_model` | 48 | 256 |
| Transformer `dim_ff` | 128 | 512 |
| Transformer frame encoder | 2-layer Conv2d+AvgPool (0.04M) | ResNet-18 backbone (11M) |

---

## Results

### Environmental Regime

| Model | PR-AUC (base) | PR-AUC (upgr) | F1 | Recall | ROC-AUC | Δ PR-AUC |
|-------|:------------:|:------------:|:---:|:------:|:-------:|:--------:|
| LightGBM | **0.485** | **0.485** | 0.488 | 0.585 | 0.813 | — |
| RF | 0.360 | 0.360 | 0.414 | 0.627 | 0.743 | — |
| Transformer | 0.253 | 0.235 | 0.325 | 0.573 | 0.637 | **−0.018** |
| ConvLSTM | 0.158 | 0.166 | 0.273 | 0.957 | 0.517 | **+0.008** |

### Operational Regime

| Model | PR-AUC (base) | PR-AUC (upgr) | F1 | Recall | ROC-AUC | Δ PR-AUC |
|-------|:------------:|:------------:|:---:|:------:|:-------:|:--------:|
| LightGBM | **0.689** | **0.689** | 0.610 | 0.576 | 0.873 | — |
| RF | 0.685 | 0.685 | 0.616 | 0.534 | 0.868 | — |
| Transformer | 0.237 | 0.187 | 0.284 | 0.760 | 0.568 | **−0.050** |
| ConvLSTM | 0.157 | 0.165 | 0.273 | 0.934 | 0.522 | **+0.008** |
| Persistence | 0.183 | 0.183 | 0.271 | 1.000 | 0.548 | — |

---

## Key Findings

1. **LightGBM dominates both regimes by a large margin.** The upgraded deep models did not close the gap.

2. **ConvLSTM (+0.008 PR-AUC)** — negligible gain from 5× hidden capacity. Training loss remains flat (1.040→1.040) across all epochs in all runs. The ConvLSTM architecture does not learn from this data regardless of capacity.

3. **Transformer with ResNet-18 regressed (−0.018 env / −0.050 op).** Training dynamics explain why:
   - Epoch 1: loss ~0.96, PR-AUC > 0.32 (brief learning)
   - Epoch 3+: loss spikes to 1.04, PR-AUC collapses
   - The default `lr=1e-3` is too high for an 11M-param ResNet backbone on 15×15 patches. The model destabilizes and never recovers.

4. **Deep models remain inefficient on this rare-event problem.** With ~25% synthetic positive ratio and small spatial patches (15×15), gradient boosting is far more data-efficient. Three likely paths forward:
   - Lower learning rate + warm-up for the ResNet Transformer
   - Larger spatial context (>15×15 patches)
   - Different architecture (Swin, ViT-pretrained, or spatiotemporal attention pooling)

CSV comparison tables saved to `outputs/baseline/` and `outputs/upgraded/`.
