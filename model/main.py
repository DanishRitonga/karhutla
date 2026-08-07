"""
main.py
=======
End-to-end demo run of every model in Table 3, across both feature regimes
(environmental vs. operational, Section 3.2), on synthetic data shaped
exactly like the paper's [N, 14, 15, 15, C] tensor.

Run:  python3 main.py
Outputs land in ./outputs/:
  - comparison_table.csv     (mirrors Table 3, PR-AUC/F1/Recall/ROC-AUC per model x regime)
  - shap_importance.png      (SHAP feature importance, tabular LightGBM, environmental regime)
  - attention_heatmap.png    (Temporal Transformer self-attention over the 14 input days)
"""
import time
import numpy as np
import pandas as pd

import data
import models
import train_eval as te
import interpret

SEED = 42
DEVICE = "cpu"
OUT_DIR = "outputs"


def run_regime(regime_name, channels, X_full, y, day_index, met_channels):
    print(f"\n{'=' * 70}\nREGIME: {regime_name}  ({len(channels)} channels)\n{'=' * 70}")
    X = X_full[..., channels]
    train_mask, test_mask = data.temporal_split(day_index, test_frac=0.2)
    # carve a small validation slice out of the training period for the torch models
    train_days = day_index[train_mask]
    val_cutoff = np.quantile(train_days, 0.85)
    val_mask = train_mask & (day_index > val_cutoff)
    fit_mask = train_mask & ~val_mask

    fire_hist_pos = channels.index(data.FIRE_HISTORY_IDX) if data.FIRE_HISTORY_IDX in channels else None

    results = []

    # ---- 1. Persistence baseline (tabular repr, uses fire-history column) --
    X_tab, feat_names = data.to_tabular(X, list(range(len(channels))))
    # locate the fire-history "center_last" column inside the tabular block, if present
    persist_col = None
    if fire_hist_pos is not None:
        name = f"center_last__{data.CHANNEL_NAMES[channels[fire_hist_pos]]}"
        persist_col = feat_names.index(name)
    pers = models.PersistenceBaseline(persist_col)
    pers.fit(X_tab[fit_mask], y[fit_mask])
    prob = pers.predict_proba(X_tab[test_mask])[:, 1]
    results.append({"regime": regime_name, "model": "Persistence", **te.evaluate_probs(y[test_mask], prob)})

    # ---- 2. Meteorological logistic regression (ERA5 vars only) -----------
    X_met, _ = data.to_tabular(X_full, met_channels)
    met_lr = models.make_meteorological_lr()
    met_lr.fit(X_met[fit_mask], y[fit_mask])
    prob = met_lr.predict_proba(X_met[test_mask])[:, 1]
    results.append({"regime": regime_name, "model": "Meteorological LR", **te.evaluate_probs(y[test_mask], prob)})

    # ---- 3. Tabular baselines: LR, RF, LightGBM ----------------------------
    tab_lr = models.make_tabular_lr()
    tab_lr.fit(X_tab[fit_mask], y[fit_mask])
    prob = tab_lr.predict_proba(X_tab[test_mask])[:, 1]
    results.append({"regime": regime_name, "model": "Tabular LR", **te.evaluate_probs(y[test_mask], prob)})

    tab_rf = models.make_tabular_rf()
    tab_rf.fit(X_tab[fit_mask], y[fit_mask])
    prob = tab_rf.predict_proba(X_tab[test_mask])[:, 1]
    results.append({"regime": regime_name, "model": "Tabular RF", **te.evaluate_probs(y[test_mask], prob)})

    n_pos, n_neg = y[fit_mask].sum(), len(y[fit_mask]) - y[fit_mask].sum()
    tab_lgb = models.make_tabular_lightgbm(scale_pos_weight=n_neg / max(n_pos, 1))
    tab_lgb.fit(X_tab[fit_mask], y[fit_mask])
    prob = tab_lgb.predict_proba(X_tab[test_mask])[:, 1]
    results.append({"regime": regime_name, "model": "Tabular LightGBM", **te.evaluate_probs(y[test_mask], prob)})

    # ---- 4. ConvLSTM --------------------------------------------------------
    print("\n[ConvLSTM]")
    t0 = time.time()
    convlstm = models.ConvLSTMHotspot(in_channels=len(channels), hidden_channels=(12, 12))
    convlstm = te.train_torch_model(convlstm, X[fit_mask], y[fit_mask], X[val_mask], y[val_mask],
                                     epochs=4, batch_size=32, device=DEVICE)
    prob = te.predict_torch_model(convlstm, X[test_mask], device=DEVICE)
    results.append({"regime": regime_name, "model": "ConvLSTM", **te.evaluate_probs(y[test_mask], prob)})
    print(f"  ({time.time() - t0:.1f}s)")

    # ---- 5. Temporal Transformer ---------------------------------------------
    print("\n[Temporal Transformer]")
    t0 = time.time()
    transformer = models.TemporalTransformerHotspot(in_channels=len(channels), seq_len=data.T_IN,
                                                      d_model=48, n_layers=2)
    transformer = te.train_torch_model(transformer, X[fit_mask], y[fit_mask], X[val_mask], y[val_mask],
                                        epochs=4, batch_size=32, device=DEVICE)
    prob = te.predict_torch_model(transformer, X[test_mask], device=DEVICE)
    results.append({"regime": regime_name, "model": "Temporal Transformer", **te.evaluate_probs(y[test_mask], prob)})
    print(f"  ({time.time() - t0:.1f}s)")

    extras = {
        "X_tab_test": X_tab[test_mask], "feat_names": feat_names, "lgb_model": tab_lgb,
        "X_test": X[test_mask], "y_test": y[test_mask], "transformer": transformer,
    }
    return results, extras


def main():
    print("Generating synthetic multimodal spatiotemporal dataset "
          "(shape mirrors [N, 14, 15, 15, C] from Section 3.3) ...")
    X_full, y, day_index = data.build_dataset(n_samples=900, n_days=1095, grid_h=45, grid_w=45, seed=SEED)
    print(f"  X shape: {X_full.shape}   positive rate: {y.mean():.3f}")

    met_channels = list(range(8))  # ERA5-Land only, ties to Table 1

    all_results = []
    env_results, env_extras = run_regime(
        "Environmental (no fire history)", data.ENV_CHANNELS, X_full, y, day_index, met_channels
    )
    op_results, op_extras = run_regime(
        "Operational (with fire history)", data.OPERATIONAL_CHANNELS, X_full, y, day_index, met_channels
    )
    all_results += env_results + op_results

    df = pd.DataFrame(all_results)
    df = df[["regime", "model", "PR-AUC", "F1", "Recall", "ROC-AUC", "best_threshold"]]
    for c in ["PR-AUC", "F1", "Recall", "ROC-AUC", "best_threshold"]:
        df[c] = df[c].astype(float).round(3)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/comparison_table.csv", index=False)
    print("\n" + "=" * 70)
    print("COMPARISON TABLE  (mirrors paper Table 3, held-out test period)")
    print("=" * 70)
    print(df.to_string(index=False))

    # ---- interpretability (Section 3.5) -------------------------------
    print("\nRunning SHAP analysis on the environmental-regime LightGBM model ...")
    interpret.shap_summary_for_lightgbm(
        env_extras["lgb_model"], env_extras["X_tab_test"][:300], env_extras["feat_names"],
        f"{OUT_DIR}/shap_importance.png",
    )

    print("Rendering attention heatmap for the environmental-regime Temporal Transformer ...")
    pos_idx = np.where(env_extras["y_test"] == 1)[0]
    sample_i = pos_idx[0] if len(pos_idx) else 0
    interpret.attention_heatmap(
        env_extras["transformer"], env_extras["X_test"][sample_i], DEVICE,
        f"{OUT_DIR}/attention_heatmap.png",
    )

    print(f"\nDone. Outputs saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
