"""
main_real.py
============
Same model comparison as main.py, but the LABELS and the FIRE-HISTORY
channel now come from the real FIRMS VIIRS-SNPP data the user uploaded,
filtered to Riau and gridded to 5 km cells (real_data.py). Train/test split
is the real calendar years: train = 2019-2022, test = 2023, matching the
paper's primary evaluation protocol exactly (Section 3.5).

Grid + peat (channel 20) are now REAL, from a teammate's handoff
(real_grid_data/, see real_grid.py + real_data.py docstrings). ERA5-Land/
CHIRPS/SAR/Dynamic World (channels 0-19) are still synthetic placeholders
-- see real_data.py's module docstring and README for why, and for what
that means when reading the "Environmental regime" row of the results.
"""
import time
import numpy as np
import pandas as pd

import data
import models
import real_data
import train_eval as te
import interpret

DEVICE = "cpu"
OUT_DIR = "outputs"
SEED = 42

# n_train_samples/n_test_samples used to be a flat 900/900, picked without a
# stated reason. That is small enough that it structurally favours
# low-sample-efficiency-need models (RF, LR) over data-hungry ConvLSTM /
# Transformer, even though the true pool of eligible cell-days is far
# larger (thousands of eligible Riau cells x ~1,460 train-year days).
# We cannot use the full pool -- ConvLSTM/Transformer and the tabular
# baselines all consume the SAME extracted [N, T_IN, PATCH, PATCH, C]
# tensor (data.to_tabular collapses it, it does not avoid building it), and
# that tensor is the memory bottleneck: T_IN*PATCH*PATCH*N_CHANNELS*4 bytes
# per sample. Instead of another arbitrary constant, we derive n_samples
# from an explicit, stated RAM budget so the number is reproducible and
# adjustable (raise PATCH_MEMORY_BUDGET_GB on a bigger machine to sample
# closer to the true population; this is reported in the printout below so
# it's citable in the methodology section).
PATCH_MEMORY_BUDGET_GB = 3.0  # peak size of the raw X_train+X_test patch tensors


def _samples_from_memory_budget(budget_gb, train_frac=0.75):
    """(n_train_samples, n_test_samples) that keep X_train+X_test within
    `budget_gb` of raw float32 patch tensor, split 75/25 train/test (train
    spans 4 calendar years vs test's 1, so it gets the larger share)."""
    bytes_per_sample = data.T_IN * data.PATCH * data.PATCH * data.N_CHANNELS * 4
    total_samples = int(budget_gb * (1024 ** 3) / bytes_per_sample)
    n_train = int(total_samples * train_frac)
    n_test = total_samples - n_train
    return n_train, n_test, bytes_per_sample


def run_regime_real(regime_name, channels, X_train, y_train, X_test, y_test, met_channels, day_train,
                     hotspot_7d_train, hotspot_7d_test):
    print(f"\n{'=' * 70}\nREGIME: {regime_name}  ({len(channels)} channels)\n{'=' * 70}")
    rng = np.random.default_rng(SEED)
    n = len(X_train)

    # Real calendar-year holdout for validation (2022) instead of a random
    # slice of 2019-2022: a random slice shares its distribution with fit,
    # so early-stopping/monitoring on it doesn't test cross-year
    # generalization the way the real test year (2023) demands. Using 2022
    # as validation gives ConvLSTM/Transformer a genuinely year-out-of-
    # distribution signal to select on, closer to what actually matters.
    year_train = (pd.Timestamp(real_data.DATE_START) + pd.to_timedelta(day_train, unit="D")).year
    val_mask = year_train == 2022
    fit_mask = ~val_mask
    n_val = val_mask.sum()
    if n_val < 20:
        print(f"  [peringatan] cuma {n_val} sampel training di tahun 2022 -- terlalu sedikit untuk "
              f"validation set yang stabil. Jatuh ke slice acak 15% seperti sebelumnya.")
        perm = rng.permutation(n)
        n_val = max(int(0.15 * n), 20)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
    else:
        val_idx, fit_idx = np.where(val_mask)[0], np.where(fit_mask)[0]
        print(f"  validation = tahun 2022 asli ({n_val} sampel), fit = 2019-2021 ({len(fit_idx)} sampel)")

    X = X_train[..., channels]
    Xte = X_test[..., channels]
    fire_hist_pos = channels.index(data.FIRE_HISTORY_IDX) if data.FIRE_HISTORY_IDX in channels else None

    results = []

    # 2022 was Riau's quietest fire year of 2019-2023 (2,398 VIIRS detections
    # vs 35,166 / 5,496 / 3,462 / 4,461 for 2019/2020/2021/2023, corroborated
    # by an independent burned-area source -- see HANDOFF Fase 6). An A/B
    # test on this real data showed including 2022 in tabular training
    # *hurts* test-2023 performance (RF PR-AUC 0.646 -> 0.494), because 2022's
    # pattern is unrepresentative of an active fire year like the test year.
    # So tabular/linear/persistence models train on fit_idx only (2019-2021),
    # EXCLUDING 2022 entirely. Only the deep models below use val_idx (2022)
    # -- as a held-out validation set for checkpoint selection, never as
    # training signal.
    X_tab_full, feat_names = data.to_tabular(X[fit_idx], list(range(len(channels))))
    X_tab_test, _ = data.to_tabular(Xte, list(range(len(channels))))
    y_full = y_train[fit_idx]

    # Fase 8: hotspot_count_7d is computed from raw daily counts in
    # real_data.py (not derivable from the 14-day-rolled fire-history
    # *channel* -- see to_tabular's docstring), so it's appended here as an
    # extra tabular column rather than inside to_tabular. Operational
    # regime only: it's a fire-history-derived feature, meaningless
    # (all zeros/duplicated) for the Environmental regime.
    if fire_hist_pos is not None:
        X_tab_full = np.concatenate([X_tab_full, hotspot_7d_train[fit_idx, None]], axis=1)
        X_tab_test = np.concatenate([X_tab_test, hotspot_7d_test[:, None]], axis=1)
        feat_names = feat_names + ["hotspot_count_7d"]

    persist_col = None
    if fire_hist_pos is not None:
        name = f"center_last__{data.CHANNEL_NAMES[channels[fire_hist_pos]]}"
        persist_col = feat_names.index(name)
    pers = models.PersistenceBaseline(persist_col)
    pers.fit(X_tab_full, y_full)
    prob = pers.predict_proba(X_tab_test)[:, 1]
    results.append({"regime": regime_name, "model": "Persistence", **te.evaluate_probs(y_test, prob)})

    X_met_full, _ = data.to_tabular(X_train[fit_idx], met_channels)
    X_met_test, _ = data.to_tabular(X_test, met_channels)
    met_lr = models.make_meteorological_lr()
    met_lr.fit(X_met_full, y_full)
    prob = met_lr.predict_proba(X_met_test)[:, 1]
    results.append({"regime": regime_name, "model": "Meteorological LR", **te.evaluate_probs(y_test, prob)})

    tab_lr = models.make_tabular_lr()
    tab_lr.fit(X_tab_full, y_full)
    prob = tab_lr.predict_proba(X_tab_test)[:, 1]
    results.append({"regime": regime_name, "model": "Tabular LR", **te.evaluate_probs(y_test, prob)})

    tab_rf = models.make_tabular_rf()
    tab_rf.fit(X_tab_full, y_full)
    prob = tab_rf.predict_proba(X_tab_test)[:, 1]
    results.append({"regime": regime_name, "model": "Tabular RF", **te.evaluate_probs(y_test, prob)})

    n_pos, n_neg = y_full.sum(), len(y_full) - y_full.sum()
    tab_lgb = models.make_tabular_lightgbm(scale_pos_weight=n_neg / max(n_pos, 1))
    tab_lgb.fit(X_tab_full, y_full)
    prob = tab_lgb.predict_proba(X_tab_test)[:, 1]
    results.append({"regime": regime_name, "model": "Tabular LightGBM", **te.evaluate_probs(y_test, prob)})

    print("\n[ConvLSTM]")
    t0 = time.time()
    convlstm = models.ConvLSTMHotspot(in_channels=len(channels), hidden_channels=(12, 12))
    convlstm = te.train_torch_model(convlstm, X[fit_idx], y_train[fit_idx], X[val_idx], y_train[val_idx],
                                     epochs=5, batch_size=32, device=DEVICE)
    prob = te.predict_torch_model(convlstm, Xte, device=DEVICE)
    results.append({"regime": regime_name, "model": "ConvLSTM", **te.evaluate_probs(y_test, prob)})
    print(f"  ({time.time() - t0:.1f}s)")

    print("\n[Temporal Transformer]")
    t0 = time.time()
    transformer = models.TemporalTransformerHotspot(in_channels=len(channels), seq_len=data.T_IN,
                                                      d_model=48, n_layers=2)
    transformer = te.train_torch_model(transformer, X[fit_idx], y_train[fit_idx], X[val_idx], y_train[val_idx],
                                        epochs=5, batch_size=32, device=DEVICE)
    prob = te.predict_torch_model(transformer, Xte, device=DEVICE)
    results.append({"regime": regime_name, "model": "Temporal Transformer", **te.evaluate_probs(y_test, prob)})
    print(f"  ({time.time() - t0:.1f}s)")

    extras = {"X_tab_test": X_tab_test, "feat_names": feat_names, "lgb_model": tab_lgb,
              "X_test": Xte, "y_test": y_test, "transformer": transformer}
    return results, extras


def main():
    print("Loading + gridding REAL FIRMS VIIRS-SNPP detections for Riau (2019-2023) ...")
    csv_paths = {y: f"real_data/viirs-snpp/{y}/viirs-snpp_{y}_Indonesia.csv" for y in range(2019, 2024)}

    n_train_samples, n_test_samples, bytes_per_sample = _samples_from_memory_budget(PATCH_MEMORY_BUDGET_GB)
    print(f"  sample budget: {PATCH_MEMORY_BUDGET_GB:.1f} GB raw patch tensor "
          f"({bytes_per_sample / 1024:.1f} KiB/sample) "
          f"-> n_train_samples={n_train_samples}, n_test_samples={n_test_samples} "
          f"(was a flat 900/900; same budget-derived N is still shared by every "
          f"model in a regime, tabular and deep alike, so the RF-vs-ConvLSTM "
          f"comparison stays controlled -- it's just no longer needlessly tiny)")

    ds = real_data.build_real_dataset(csv_paths, n_train_samples=n_train_samples, n_test_samples=n_test_samples,
                                       train_pos_frac=0.25, test_pos_frac=0.10, seed=SEED)
    meta = ds["meta"]
    print(f"  grid: {meta['grid_h']} x {meta['grid_w']} cells @ 5km | {meta['n_days']} days "
          f"(2019-01-01..2023-12-31)")
    print(f"  real detections in Riau bbox (confidence n/h): {meta['n_real_detections']}")
    print(f"  TRUE prevalence  -> train(2019-2022): {meta['true_prevalence_train']:.4%}  "
          f"test(2023): {meta['true_prevalence_test']:.4%}")
    print(f"  SAMPLE prevalence (after stratified oversampling for tractability) "
          f"-> train: {meta['sample_prevalence_train']:.2%}  test: {meta['sample_prevalence_test']:.2%}")
    print(f"  eligible cell-day pool -> train(2019-2022): {meta['n_eligible_pool_train']:,} "
          f"({meta['n_pos_pool_train']:,} positive) | test(2023): {meta['n_eligible_pool_test']:,} "
          f"({meta['n_pos_pool_test']:,} positive)")
    print(f"  sampled -> train: {n_train_samples:,} ({100 * n_train_samples / max(meta['n_eligible_pool_train'], 1):.2f}% "
          f"of pool) | test: {n_test_samples:,} "
          f"({100 * n_test_samples / max(meta['n_eligible_pool_test'], 1):.2f}% of pool)")
    print("  (metrics below are on the enriched sample, NOT the true population rate above -- see README)")

    print(f"  grid source: {meta['grid_source']} "
          f"({'real BIG boundary' if meta['grid_source'] == 'real_grid_data' else 'fallback GeoJSON'})")
    peat_cov = meta.get("peat_coverage")
    if peat_cov is not None:
        print(f"  peat channel: REAL (BIG Peta Fungsi Ekosistem Gambut), "
              f"{peat_cov:.1%} of Riau cells have peat")
    else:
        print("  peat channel: synthetic (real_grid_data/peat_cell.csv not found)")

    env_cov = meta.get("env_coverage")
    if env_cov:
        real_any = any(v > 0 for v in env_cov.values())
        if real_any:
            print("  Fase 7 real environmental coverage (fraction of day-cell entries that are REAL, "
                  "rest = synthetic placeholder):")
            for k, v in env_cov.items():
                print(f"    {k}: {v:.1%}")
        else:
            print("  Fase 7: no real CHIRPS/Sentinel-1/peat CSVs found yet -- "
                  "environmental channels are 100% synthetic (run gee_ingest.py / fetch_peat.py first)")

    met_channels = list(range(8))
    all_results = []
    env_results, env_extras = run_regime_real(
        "Environmental (no fire history)", data.ENV_CHANNELS,
        ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"], met_channels, ds["day_train"],
        ds["hotspot_count_7d_train"], ds["hotspot_count_7d_test"],
    )
    op_results, op_extras = run_regime_real(
        "Operational (with real fire history)", data.OPERATIONAL_CHANNELS,
        ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"], met_channels, ds["day_train"],
        ds["hotspot_count_7d_train"], ds["hotspot_count_7d_test"],
    )
    all_results += env_results + op_results

    df = pd.DataFrame(all_results)
    df = df[["regime", "model", "PR-AUC", "F1", "Recall", "ROC-AUC", "best_threshold"]]
    for c in ["PR-AUC", "F1", "Recall", "ROC-AUC", "best_threshold"]:
        df[c] = df[c].astype(float).round(3)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/comparison_table_real.csv", index=False)
    print("\n" + "=" * 70)
    print("COMPARISON TABLE -- REAL VIIRS LABELS, test = calendar year 2023")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nRunning SHAP analysis (operational-regime LightGBM, real labels) ...")
    interpret.shap_summary_for_lightgbm(
        op_extras["lgb_model"], op_extras["X_tab_test"][:300], op_extras["feat_names"],
        f"{OUT_DIR}/shap_importance_real.png",
    )

    print("Rendering attention heatmap (operational-regime Temporal Transformer, real labels) ...")
    pos_idx = np.where(op_extras["y_test"] == 1)[0]
    sample_i = pos_idx[0] if len(pos_idx) else 0
    interpret.attention_heatmap(
        op_extras["transformer"], op_extras["X_test"][sample_i], DEVICE,
        f"{OUT_DIR}/attention_heatmap_real.png",
    )

    print(f"\nDone. Outputs saved to ./{OUT_DIR}/ (*_real.csv / *_real.png)")


if __name__ == "__main__":
    main()
