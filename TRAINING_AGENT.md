# Training Agent Instructions — Karhutla Riau Peatland Fire EWS

You are running the model-training step for a Datathon 2026 project on a **separate device** with enough RAM/CPU/GPU. The data and code are already committed to git and hosted on HuggingFace. Follow exactly — do not improvise the pipeline.

## 1. Get the code

```bash
git clone https://github.com/DanishRitonga/karhutla.git
cd karhutla
git checkout dnsh
```

## 2. Get the data (from HuggingFace)

Dataset repo: `danishritonga/karhutla` (type: `dataset`).

The training pipeline auto-downloads the tensors from HuggingFace if they are
not found locally — `model/train.py` (via `model/data.load_tensors()`) pulls
`tensors/` from the repo into `--tensor-dir` (default `data/output/tensors`)
on first run. Just make sure you are logged in:

```bash
# login once
huggingface-cli login
```

Optionally pre-download the pre-built tensors yourself (DO NOT rebuild them —
they took hours to assemble and are 100% validated, 0 NaN):

```bash
huggingface-cli download danishritonga/karhutla tensors/ --repo-type=dataset --local-dir data/output
```

Either way you must end up with:
- `data/output/tensors/data.npy`   → shape `(1826, 82, 85, 23)`, float32
- `data/output/tensors/labels.npy` → shape `(1826, 82, 85)`, int8 (values -1/0/1)
- `data/output/tensors/meta.json`

Verify:
```bash
python -c "import numpy as np; d=np.load('data/output/tensors/data.npy'); l=np.load('data/output/tensors/labels.npy'); print(d.shape, d.dtype, l.shape, l.dtype, 'NaN:', int(np.isnan(d).sum()))"
# expect: (1826, 82, 85, 23) float32 (1826, 82, 85) int8 NaN: 0
```

> **Do NOT run `data/loader/tensor_assembly.py`.** It needs raw VIIRS zips and every GEE source. The tensors are the validated artifact.

## 3. Set up the environment

Requires Python 3.12 (uv is optional — plain venv is fine):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch scikit-learn lightgbm shap numpy pandas
```

## 4. Run training — both regimes

**Environmental regime** (21 channels, excludes fire history — RQ1/RQ2):
```bash
python model/train.py --regime env --epochs 10 --out-dir outputs
```

**Operational regime** (22 channels, includes `hotspot_count_lag` fire history):
```bash
python model/train.py --regime operational --epochs 10 --out-dir outputs
```

> **Longer-epochs experiment (ConvLSTM + Transformer convergence).** Prior runs
> used 30 epochs; both DL models were still improving at epoch 30 (ConvLSTM env
> loss still decreasing; Transformer op had a ~14-epoch "escape" phase then rapid
> learning). The training loop restores the best-val-PR-AUC checkpoint, so
> overfitting is protected. Run the two spatiotemporal models for 60+ epochs to
> check convergence. Either run the full suite with `--epochs 60`, or to save
> wall-clock time, re-run just the DL models via a slimmed call (tabular models
> are epoch-free and unaffected):
>
> ```bash
> python model/train.py --regime env --epochs 60 --n-train 20000 --n-val 5000 --out-dir outputs_long
> python model/train.py --regime operational --epochs 60 --n-train 20000 --n-val 5000 --out-dir outputs_long
> ```
>
> Optionally also try the **seasonal 1:1 negative matching** (Sinato & Rivas
> 2026): `--balance seasonal` (training-set only; val/test always random).
> Earlier 30-epoch seasonal results: LightGBM collapses (0.477→0.305 env) while
> Transformer improves (0.201→0.322 env) — trees were exploiting calendar
> proxies. A 60-epoch seasonal run tells us if the Transformer closes the gap
> further.

Model inventory (train.py runs them all):
| Model | Family |
|---|---|
| PersistenceBaseline | baseline |
| Meteorological LR (ERA5-only) | tabular |
| Logistic Regression | tabular |
| Random Forest | tabular |
| LightGBM | tabular |
| XGBoost (Sinato 2026 config: 300 trees, depth 8, lr 0.05) | tabular |
| ConvLSTM (hidden (12,12)) | spatiotemporal |
| Temporal Transformer (ResNet-18 frame encoder, d_model=256) | spatiotemporal |

Train 2019–2021 · Val 2022 · Test 2023. Metrics: PR-AUC primary, then F1, Recall, ROC-AUC.

## 5. Expected outputs (in `outputs/`)

- `comparison_table_env.csv` / `comparison_table_operational.csv`
- `shap_importance_env.png` / `shap_importance_operational.png`
- `attention_heatmap_env.png` / `attention_heatmap_operational.png`

## 6. Memory note (why we don't run here)

`X_train` alone is `[50000, 14, 15, 15, 21]` float32 ≈ 2.6 GB, plus val/test and tabular copies. If you hit OOM, drop `--n-train` to 20000 and `--n-val` to 5000 (keep `--n-test 20000`).

## 7. Report back

Return to the user: the two comparison tables, and a one-line summary per model per regime (PR-AUC/F1/Recall/ROC-AUC). Do not modify any source files.
