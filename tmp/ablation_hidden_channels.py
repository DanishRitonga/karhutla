"""
ablation_hidden_channels.py
============================
Resolves the two open questions flagged in review before ConvLSTM's
`hidden_channels=(12, 12)` can be reported in the paper without looking like
a leftover prototype constant:

  1. Convergence: are the 4-5 epochs used in main.py / main_real.py enough
     for ConvLSTM to actually reach its best validation PR-AUC, or does
     training stop while it is still improving?
  2. Capacity: how does val PR-AUC change across hidden_channels in
     {12, 24, 32} (12 = current, 24 = models.py's own class default, 32 =
     closer to the Temporal Transformer's d_model=48), holding everything
     else -- regime, data, seed, optimizer -- fixed?

Design choices, and why:

- Reuses main_real.py's EXACT data loading (_samples_from_memory_budget +
  real_data.build_real_dataset) and run_regime_real's EXACT fit/val split
  (year 2022 = val, 2019-2021 = fit) so these numbers are directly
  comparable to outputs/comparison_table_real.csv. Nothing about the data
  pipeline is re-decided here.
- Only ConvLSTM is touched. Persistence/LR/RF/LightGBM/Transformer are not
  retrained or re-evaluated by this script.
- Evaluated ONLY on val_idx (year 2022) -- never on X_test/y_test (year
  2023). Selecting hidden_channels by peeking at the test set would be the
  same test-set leakage the val/test split exists to prevent; the whole
  point is to pick a config on 2022 and only then let 2023 judge it once.
- Early stopping is on val PR-AUC with a patience window, not a fixed
  epoch count, so a run that hasn't converged is visible in the output
  instead of silently reported as if it had.

Usage:
    python3 ablation_hidden_channels.py
        Real data. Requires (same as main_real.py):
          - grid_definition.py copied next to real_data.py (see README.md,
            "Cara pakai")
          - real_data/viirs-snpp/<year>/viirs-snpp_<year>_Indonesia.csv for
            2019-2023
          - real_grid_data/ optional (falls back to
            riau_boundary_fallback.geojson + synthetic peat if absent, same
            as main_real.py)

    python3 ablation_hidden_channels.py --dry-run
        Synthetic data.py generator instead of real_data. Only proves the
        ablation harness itself runs end to end (loop, early stopping,
        checkpointing, CSV output) -- the PR-AUC numbers it prints are NOT
        valid for the paper.

Other flags: --regimes environmental,operational  --hidden-grid 12,24,32
             --max-epochs 30  --patience 6  --patch-memory-budget-gb 3.0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import data
import models
import train_eval as te

SEED = 42
DEVICE = "cpu"
OUT_DIR = "outputs"


def _load_real(patch_memory_budget_gb):
    import real_data
    import main_real as mr

    csv_paths = {y: f"real_data/viirs-snpp/{y}/viirs-snpp_{y}_Indonesia.csv" for y in range(2019, 2024)}
    n_train_samples, n_test_samples, _ = mr._samples_from_memory_budget(patch_memory_budget_gb)
    ds = real_data.build_real_dataset(
        csv_paths, n_train_samples=n_train_samples, n_test_samples=n_test_samples,
        train_pos_frac=0.25, test_pos_frac=0.10, seed=SEED,
    )
    y_train, day_train = ds["y_train"], ds["day_train"]
    year_train = (pd.Timestamp(real_data.DATE_START) + pd.to_timedelta(day_train, unit="D")).year
    val_mask = year_train == 2022
    fit_mask = ~val_mask
    n_val = val_mask.sum()
    if n_val < 20:
        rng = np.random.default_rng(SEED)
        n = len(y_train)
        perm = rng.permutation(n)
        n_val = max(int(0.15 * n), 20)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        print(f"  [peringatan] cuma {int(val_mask.sum())} sampel training di tahun 2022 -- "
              f"jatuh ke slice acak 15% (sama seperti main_real.py).")
    else:
        val_idx, fit_idx = np.where(val_mask)[0], np.where(fit_mask)[0]
    print(f"  [real] fit = tahun 2019-2021 ({len(fit_idx)} sampel), "
          f"val = tahun 2022 asli ({len(val_idx)} sampel)")
    return ds["X_train"], y_train, fit_idx, val_idx


def _load_dry_run():
    print("  [dry-run] pakai data.py synthetic generator -- INI CUMA SMOKE-TEST HARNESS,")
    print("  [dry-run] angka PR-AUC di bawah TIDAK VALID buat paper.")
    X_full, y, day_index = data.build_dataset(n_samples=900, n_days=1095, grid_h=45, grid_w=45, seed=SEED)
    train_mask, _test_mask = data.temporal_split(day_index, test_frac=0.2)
    train_days = day_index[train_mask]
    val_cutoff = np.quantile(train_days, 0.85)
    val_mask = train_mask & (day_index > val_cutoff)
    fit_mask = train_mask & ~val_mask
    fit_idx = np.where(fit_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"  [dry-run] fit = {len(fit_idx)} sampel, val = {len(val_idx)} sampel")
    return X_full, y, fit_idx, val_idx


def train_with_curve(model, X_fit, y_fit, X_val, y_val, max_epochs, patience,
                      batch_size=32, lr=1e-3, device=DEVICE):
    """
    Same loss/optimizer/pos_weight as train_eval.train_torch_model. Two
    differences: (1) runs up to max_epochs with early stopping on val
    PR-AUC instead of a fixed small epoch count, (2) returns the full
    per-epoch val PR-AUC curve so convergence is visible, not just the
    final number.
    """
    model = model.to(device)
    Xt = torch.from_numpy(X_fit).permute(0, 1, 4, 2, 3).float()
    yt = torch.from_numpy(y_fit).float()
    Xv = torch.from_numpy(X_val).permute(0, 1, 4, 2, 3).float().to(device)
    yv = y_val

    n_pos = max(yt.sum().item(), 1)
    n_neg = max(len(yt) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n = len(Xt)
    curve = []
    best_pr_auc, best_epoch, best_state = -1.0, 0, None
    epochs_since_best = 0
    stopped_early = False

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            optim.zero_grad()
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_logits = val_out[0] if isinstance(val_out, tuple) else val_out
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_metrics = te.evaluate_probs(yv, val_prob)
        curve.append({"epoch": epoch + 1, "train_loss": epoch_loss, "val_PR_AUC": val_metrics["PR-AUC"]})

        improved = val_metrics["PR-AUC"] > best_pr_auc
        if improved:
            best_pr_auc = val_metrics["PR-AUC"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(f"    epoch {epoch + 1:>2}/{max_epochs}  train_loss={epoch_loss:.4f}  "
              f"val_PR-AUC={val_metrics['PR-AUC']:.3f}" + ("  <- best" if improved else ""))

        if epochs_since_best >= patience:
            stopped_early = True
            print(f"    early stop: val PR-AUC belum improve {patience} epoch berturut-turut "
                  f"(best={best_pr_auc:.3f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # "converged" here means training stopped BEFORE exhausting max_epochs,
    # i.e. val PR-AUC plateaued and early stopping fired -- not that best
    # epoch was early. If the loop runs all max_epochs with the best score
    # still on the last epoch, that's the undertraining signal.
    converged = stopped_early
    return model, curve, best_pr_auc, best_epoch, converged


def run_ablation(regime_name, channels, X_full, y, fit_idx, val_idx, hidden_grid, max_epochs, patience):
    X = X_full[..., channels]
    rows, curves = [], []
    for hc in hidden_grid:
        print(f"\n  -- {regime_name} | hidden_channels=({hc},{hc}) --")
        t0 = time.time()
        torch.manual_seed(SEED)
        convlstm = models.ConvLSTMHotspot(in_channels=len(channels), hidden_channels=(hc, hc))
        n_params = sum(p.numel() for p in convlstm.parameters())
        convlstm, curve, best_pr_auc, best_epoch, converged = train_with_curve(
            convlstm, X[fit_idx], y[fit_idx], X[val_idx], y[val_idx],
            max_epochs=max_epochs, patience=patience,
        )
        dt = time.time() - t0
        rows.append({
            "regime": regime_name, "hidden_channels": hc, "n_params": n_params,
            "best_val_PR_AUC": round(best_pr_auc, 4), "best_epoch": best_epoch,
            "epochs_run": len(curve), "converged_before_max_epochs": converged,
            "train_time_sec": round(dt, 1),
        })
        for c in curve:
            curves.append({"regime": regime_name, "hidden_channels": hc, **c})
        flag = "" if converged else "  [!] masih improving di epoch terakhir -- naikkan --max-epochs"
        print(f"    -> best_val_PR-AUC={best_pr_auc:.4f} @ epoch {best_epoch}/{len(curve)} "
              f"({n_params:,} params, {dt:.1f}s){flag}")
    return rows, curves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="Synthetic data.py generator, smoke-test only -- lihat docstring.")
    ap.add_argument("--regimes", default="environmental,operational")
    ap.add_argument("--hidden-grid", default="12,24,32")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--patch-memory-budget-gb", type=float, default=3.0)
    args = ap.parse_args()

    hidden_grid = [int(x) for x in args.hidden_grid.split(",")]
    regime_keys = args.regimes.split(",")

    if args.dry_run:
        X_full, y, fit_idx, val_idx = _load_dry_run()
    else:
        X_full, y, fit_idx, val_idx = _load_real(args.patch_memory_budget_gb)

    regime_map = {
        "environmental": ("Environmental (no fire history)", data.ENV_CHANNELS),
        "operational": ("Operational (with fire history)", data.OPERATIONAL_CHANNELS),
    }

    all_rows, all_curves = [], []
    for key in regime_keys:
        name, channels = regime_map[key]
        rows, curves = run_ablation(name, channels, X_full, y, fit_idx, val_idx,
                                     hidden_grid, args.max_epochs, args.patience)
        all_rows += rows
        all_curves += curves

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    curves_df = pd.DataFrame(all_curves)
    tag = "_dryrun" if args.dry_run else ""
    summary_path = f"{OUT_DIR}/ablation_hidden_channels{tag}.csv"
    curves_path = f"{OUT_DIR}/ablation_hidden_channels_curves{tag}.csv"
    summary.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    print("\n" + "=" * 78)
    title = "ABLATION SUMMARY"
    if args.dry_run:
        title += "  [DRY RUN -- synthetic data, NOT valid for the paper]"
    print(title)
    print("=" * 78)
    print(summary.to_string(index=False))

    if len(summary) and not summary["converged_before_max_epochs"].any():
        print("\n[!] Peringatan: TIDAK ADA konfigurasi yang early-stop dalam --max-epochs "
              f"({args.max_epochs}). Semua run mungkin masih undertrained -- perbandingan "
              "12 vs 24 vs 32 di atas belum tentu valid. Naikkan --max-epochs dan ulangi.")

    print(f"\nSaved: {summary_path}  (satu baris per regime x hidden_channels)")
    print(f"Saved: {curves_path}  (val PR-AUC per epoch, buat plot kurva konvergensi)")


if __name__ == "__main__":
    main()
