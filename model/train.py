"""train.py
=========
Main training entry point for the peatland-fire spatiotemporal models.

Loads pre-built tensors from tensor_assembly.py, trains all models
(ConvLSTM, Temporal Transformer, tabular baselines), evaluates on the
held-out test year (2023), and writes a comparison table + plots.

Usage (from datathon root after ``tensor_assembly.py`` has run)::

    uv run --python 3.12 python model/train.py \\
        --tensor-dir data/output/tensors \\
        --train 2019 2021 \\
        --val 2022 2022 \\
        --test 2023 2023 \\
        --regime env \\
        --epochs 10 \\
        --out-dir outputs

Run both regimes for the paper's Table 3::

    uv run python model/train.py --regime env
    uv run python model/train.py --regime operational
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from model.data import (
    load_tensors,
    eligible_mask,
    extract,
    to_tabular,
    compute_norm_stats,
    apply_norm,
    ENV_CHANNELS,
    OPERATIONAL_CHANNELS,
    JETT_CHANNEL_NAMES,
    JETT_N_CHANNELS,
    FIRE_HISTORY_IDX,
)
from model.models import (
    ConvLSTMHotspot,
    TemporalTransformerHotspot,
    PersistenceBaseline,
)
from model.train_eval import train_torch_model, predict_torch_model, evaluate_probs
from model.interpret import shap_summary_for_lightgbm, attention_heatmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")

DEVICE = "cuda"


def _sample_cell_days(
    eligible: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    yr_lo: int,
    yr_hi: int,
    n_samples: int,
    pos_frac: float = 0.25,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed + yr_lo)
    yr_mask = (years[:, None, None] >= yr_lo) & (years[:, None, None] <= yr_hi)
    mask = eligible & yr_mask
    pos = np.argwhere(mask & (labels == 1))
    neg = np.argwhere(mask & (labels == 0))
    n_pos = min(len(pos), int(n_samples * pos_frac))
    n_neg = min(len(neg), n_samples - n_pos)
    sel_pos = pos[rng.choice(len(pos), size=n_pos, replace=False)]
    sel_neg = neg[rng.choice(len(neg), size=n_neg, replace=False)]
    sel = np.concatenate([sel_pos, sel_neg], axis=0)
    rng.shuffle(sel)
    return sel[:n_samples]


def _sample_seasonal(
    eligible: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    doys: np.ndarray,
    yr_lo: int,
    yr_hi: int,
    n_samples: int,
    seasonal_margin: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """Seasonal 1:1 negative matching à la Sinato & Rivas (2026).

    Each positive cell-day is paired with one negative from the same
    ±seasonal_margin day-of-year window in a *different* year within
    [yr_lo, yr_hi]. This forces the model to distinguish fire-driving
    weather from normal seasonal weather, preventing it from becoming a
    "glorified calendar."
    """
    rng = np.random.default_rng(seed + yr_lo)
    yr_mask = (years[:, None, None] >= yr_lo) & (years[:, None, None] <= yr_hi)
    mask = eligible & yr_mask

    pos_ix = np.argwhere(mask & (labels == 1))
    neg_ix = np.argwhere(mask & (labels == 0))

    # index negatives by (day_of_year, year) for fast lookup
    neg_by_year = {}
    for i in neg_ix:
        t, r, c_ = int(i[0]), int(i[1]), int(i[2])
        d = int(doys[t])
        y = int(years[t])
        neg_by_year.setdefault((y, d), []).append(i)

    sel = []
    for pi in pos_ix:
        t, r, c_ = int(pi[0]), int(pi[1]), int(pi[2])
        target_doy = int(doys[t])
        target_yr = int(years[t])
        candidates = []
        for (y, d), idxs in neg_by_year.items():
            if y == target_yr:
                continue
            doy_diff = abs(d - target_doy)
            if doy_diff > seasonal_margin and doy_diff < 365 - seasonal_margin:
                continue
            candidates.extend(idxs)
        if not candidates:
            continue  # rare; skip this positive
        sel.append(pi)
        sel.append(candidates[rng.integers(len(candidates))])
        if len(sel) >= n_samples * 2:
            break

    sel = np.array(sel, dtype=int)
    rng.shuffle(sel)
    logger.info("seasonal 1:1 — matched %d pos (total %d samples, %.1f %% of pop)", len(sel) // 2, len(sel),
                100.0 * len(sel) / max(mask.sum(), 1))
    return sel[:n_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate spatiotemporal fire models")
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/output/tensors"))
    parser.add_argument("--train", type=int, nargs=2, default=(2019, 2021))
    parser.add_argument("--val", type=int, nargs=2, default=(2022, 2022))
    parser.add_argument("--test", type=int, nargs=2, default=(2023, 2023))
    parser.add_argument("--regime", choices=["env", "operational"], default="env")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conv-lstm-hidden", type=int, nargs="+", default=(12, 12),
                        help="ConvLSTM hidden channels, e.g. 12 12 / 24 24 / 64 32")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=20000)
    parser.add_argument("--pos-frac", type=float, default=0.25)
    parser.add_argument("--balance", choices=["random", "seasonal"], default="random",
                        help="random: sample negatives randomly (default). seasonal: 1:1 seasonal matching (Sinato & Rivas 2026).")
    parser.add_argument("--seasonal-margin", type=int, default=30,
                        help="day-of-year margin for seasonal matching (default 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regime_label = args.regime
    channels = ENV_CHANNELS if args.regime == "env" else OPERATIONAL_CHANNELS
    n_ch = len(channels)

    # -- 1. load tensors ------------------------------------------------
    logger.info("loading tensors from %s", args.tensor_dir)
    fields, labels, meta = load_tensors(args.tensor_dir)
    dates = pd.to_datetime(meta["dates"])
    years = dates.year.to_numpy()
    doys = dates.dayofyear.to_numpy()
    logger.info("fields %s, labels %s", fields.shape, labels.shape)
    logger.info("dates %s → %s", dates[0].date(), dates[-1].date())

    # -- 2. sample cell-days --------------------------------------------
    eligible = eligible_mask(labels, meta)
    logger.info("eligible cell-days: %d (%.1f%% pos)",
                int(eligible.sum()), 100 * labels[eligible].mean())
    if args.balance == "seasonal":
        logger.info("using seasonal 1:1 negative matching (margin ±%d days)", args.seasonal_margin)
        train_cd = _sample_seasonal(eligible, labels, years, doys, *args.train, args.n_train, args.seasonal_margin, args.seed)
        val_cd   = _sample_seasonal(eligible, labels, years, doys, *args.val,   args.n_val,   args.seasonal_margin, args.seed)
        test_cd  = _sample_seasonal(eligible, labels, years, doys, *args.test,  args.n_test,  args.seasonal_margin, args.seed)
    else:
        train_cd = _sample_cell_days(eligible, labels, years, *args.train, args.n_train, args.pos_frac, args.seed)
        val_cd   = _sample_cell_days(eligible, labels, years, *args.val,   args.n_val,   args.pos_frac, args.seed)
        test_cd  = _sample_cell_days(eligible, labels, years, *args.test,  args.n_test,  args.pos_frac, args.seed)
    logger.info("samples: train=%d val=%d test=%d", len(train_cd), len(val_cd), len(test_cd))

    # -- 3. extract patches + tabular features ---------------------------
    X_train, y_train, d_train = extract(fields, labels, train_cd)
    X_val, y_val, _ = extract(fields, labels, val_cd)
    X_test, y_test, _ = extract(fields, labels, test_cd)
    logger.info("patches: train %s val %s test %s", X_train.shape, X_val.shape, X_test.shape)

    # -- 3b. per-channel z-score normalisation (train stats only) ---------
    norm_stats = compute_norm_stats(X_train)
    X_train = apply_norm(X_train, norm_stats)
    X_val = apply_norm(X_val, norm_stats)
    X_test = apply_norm(X_test, norm_stats)
    logger.info("normalised: %d channels (min/max μ=%.3g/%.3g σ=%.3g/%.3g)",
                len(norm_stats),
                min(s["mean"] for s in norm_stats), max(s["mean"] for s in norm_stats),
                min(s["std"] for s in norm_stats), max(s["std"] for s in norm_stats))

    X_train_tab, tab_names = to_tabular(X_train, channels)
    X_val_tab, _          = to_tabular(X_val, channels)
    X_test_tab, _         = to_tabular(X_test, channels)

    pos_count = int(y_train.sum())
    logger.info("tabular features: %d, pos=%d (%.2f%%)", X_train_tab.shape[1], pos_count,
                100 * pos_count / len(y_train))

    # -- 4. evaluate -----------------------------------------------
    mode = "Env" if args.regime == "env" else "Op"
    results = []
    logger.info("=== %s regime (%d channels) ===", mode, n_ch)

    # --- Persistence baseline ---
    pos_in_tab_features = None
    for i, c in enumerate(channels):
        if c == FIRE_HISTORY_IDX:
            pos_in_tab_features = i * 5 + 2
            break
    if pos_in_tab_features is not None:
        logger.info("  PersistenceBaseline")
        pers = PersistenceBaseline(pos_in_tab_features)
        pers_probs = pers.predict_proba(X_test_tab)[:, 1]
        pers_met = evaluate_probs(y_test, pers_probs)
        logger.info("    Persist PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                     pers_met["PR-AUC"], pers_met["F1"], pers_met["Recall"], pers_met["ROC-AUC"])
        results.append(("Persistence", "Persistence", pers_met))

    # --- Meteorological LR (ERA5 only) ---
    met_ch = [c for c in channels if c < 8]
    if met_ch:
        logger.info("  Meteorological LR (ERA5)")
        Xmt, _ = to_tabular(X_train, met_ch)
        mtr = LogisticRegression(max_iter=2000, class_weight="balanced")
        mtr.fit(Xmt, y_train)
        Xmt_test, _ = to_tabular(X_test, met_ch)
        lr_met_prob = mtr.predict_proba(Xmt_test)[:, 1]
        lr_met_met = evaluate_probs(y_test, lr_met_prob)
        logger.info("    Met-LR  PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                     lr_met_met["PR-AUC"], lr_met_met["F1"], lr_met_met["Recall"], lr_met_met["ROC-AUC"])
        results.append(("Meteorological", "Logistic Regression", lr_met_met))

    # --- Tabular LR ---
    logger.info("  Tabular Logistic Regression")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train_tab, y_train)
    lr_prob = lr.predict_proba(X_test_tab)[:, 1]
    lr_met = evaluate_probs(y_test, lr_prob)
    logger.info("    Linear  PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                 lr_met["PR-AUC"], lr_met["F1"], lr_met["Recall"], lr_met["ROC-AUC"])
    results.append(("Tabular", "Logistic Regression", lr_met))

    # --- Tabular Random Forest ---
    logger.info("  Tabular Random Forest")
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                class_weight="balanced_subsample", random_state=args.seed)
    rf.fit(X_train_tab, y_train)
    rf_prob = rf.predict_proba(X_test_tab)[:, 1]
    rf_met = evaluate_probs(y_test, rf_prob)
    logger.info("    RF      PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                 rf_met["PR-AUC"], rf_met["F1"], rf_met["Recall"], rf_met["ROC-AUC"])
    results.append(("Tabular", "Random Forest", rf_met))

    # --- Tabular LightGBM ---
    logger.info("  Tabular LightGBM")
    lgb = LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31,
                         subsample=0.8, colsample_bytree=0.8, random_state=args.seed,
                         scale_pos_weight=sum(y_train == 0) / max(sum(y_train == 1), 1),
                         verbose=-1)
    lgb.fit(X_train_tab, y_train)
    lgb_prob = lgb.predict_proba(X_test_tab)[:, 1]
    lgb_met = evaluate_probs(y_test, lgb_prob)
    logger.info("    LGBM    PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                 lgb_met["PR-AUC"], lgb_met["F1"], lgb_met["Recall"], lgb_met["ROC-AUC"])
    results.append(("Tabular", "LightGBM", lgb_met))

    # --- Tabular XGBoost (Sinato & Rivas 2026 config) ---
    logger.info("  Tabular XGBoost")
    xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=args.seed, verbosity=0)
    xgb.fit(X_train_tab, y_train)
    xgb_prob = xgb.predict_proba(X_test_tab)[:, 1]
    xgb_met = evaluate_probs(y_test, xgb_prob)
    logger.info("    XGB     PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                 xgb_met["PR-AUC"], xgb_met["F1"], xgb_met["Recall"], xgb_met["ROC-AUC"])
    results.append(("Tabular", "XGBoost", xgb_met))

    # --- ConvLSTM ---
    hidden = tuple(args.conv_lstm_hidden)
    logger.info("  ConvLSTM hidden=%s", hidden)
    cls = ConvLSTMHotspot(in_channels=n_ch, hidden_channels=hidden, kernel_size=3, dropout=0.2)
    cls = train_torch_model(cls, X_train[..., channels], y_train,
                            X_val[..., channels], y_val,
                            lr=args.lr, epochs=args.epochs,
                            batch_size=args.batch_size, device=DEVICE, verbose=args.verbose)
    cl_prob = predict_torch_model(cls, X_test[..., channels], device=DEVICE)
    cl_met = evaluate_probs(y_test, cl_prob)
    logger.info("    ConvLSTM PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                 cl_met["PR-AUC"], cl_met["F1"], cl_met["Recall"], cl_met["ROC-AUC"])
    results.append(("Spatiotemporal", "ConvLSTM", cl_met))

    # --- Temporal Transformer ---
    d_model = 256
    logger.info("  Temporal Transformer (d_model=%d)", d_model)
    tt = TemporalTransformerHotspot(in_channels=n_ch, seq_len=14, d_model=d_model,
                                    n_heads=4, dim_ff=512, dropout=0.2)
    tt = train_torch_model(tt, X_train[..., channels], y_train,
                           X_val[..., channels], y_val,
                           lr=args.lr, epochs=args.epochs,
                           batch_size=args.batch_size, device=DEVICE, verbose=args.verbose)
    tt_prob = predict_torch_model(tt, X_test[..., channels], device=DEVICE)
    tt_met = evaluate_probs(y_test, tt_prob)
    logger.info("    Transf   PR-AUC=%.3f F1=%.3f Rec=%.3f ROC=%.3f",
                 tt_met["PR-AUC"], tt_met["F1"], tt_met["Recall"], tt_met["ROC-AUC"])
    results.append(("Spatiotemporal", "Temporal Transformer", tt_met))

    # -- 5. write comparison table ----------------------------------
    rows = []
    for family, model, met in results:
        rows.append({
            "Regime": regime_label,
            "Family": family,
            "Model": model,
            "PR-AUC": met["PR-AUC"],
            "F1": met["F1"],
            "Recall": met["Recall"],
            "ROC-AUC": met["ROC-AUC"],
            "Best Threshold": met.get("best_threshold", float("nan")),
        })
    table = pd.DataFrame(rows)
    table_path = args.out_dir / f"comparison_table_{regime_label}.csv"
    table.to_csv(table_path, index=False, float_format="%.4f")
    logger.info("comparison table → %s", table_path)
    print(table.round(4).to_string(index=False))

    # -- 6. explainability plots ------------------------------------
    shim_path = args.out_dir / f"shap_importance_{regime_label}.png"
    shap_summary_for_lightgbm(lgb, X_train_tab, tab_names, shim_path)

    # attention on a single positive test sample
    pos_idx = np.flatnonzero(y_test == 1)
    if len(pos_idx):
        idx = pos_idx[0]
        attn_path = args.out_dir / f"attention_heatmap_{regime_label}.png"
        prob, _ = attention_heatmap(tt, X_test[idx][..., channels], DEVICE, attn_path)
        logger.info("attention heatmap → %s (prob=%.3f)", attn_path, prob)

    logger.info("all done — %d models, %s regime", len(results), regime_label)


if __name__ == "__main__":
    main()
