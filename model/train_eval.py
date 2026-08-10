"""
train_eval.py
=============
Training loop for the two PyTorch spatiotemporal models, and a shared
evaluation function. Metrics follow Section 3.5 exactly:

  "Performance is primarily evaluated using PR-AUC, Recall, and F1-score,
   while ROC-AUC is reported as a secondary metric. Accuracy is not
   considered the primary evaluation metric because of the severe class
   imbalance."
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, recall_score,
)


def evaluate_probs(y_true, y_prob):
    """PR-AUC, best-F1 threshold, Recall @ that threshold, ROC-AUC (Sec 3.5)."""
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_i = np.nanargmax(f1s[:-1]) if len(f1s) > 1 else 0
    best_thr = thresholds[best_i] if len(thresholds) > 0 else 0.5

    y_pred = (y_prob >= best_thr).astype(int)
    return {
        "PR-AUC": pr_auc,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc,
        "best_threshold": float(best_thr),
    }


def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=6,
                       batch_size=32, lr=1e-3, device="cpu", verbose=True):
    """
    X_* are [N, T, H, W, C] numpy arrays (channel-last, matching the paper's
    tensor layout); converted here to torch's channel-first [N, T, C, H, W].
    Uses a pos_weight-adjusted BCE loss to handle the severe class imbalance
    called out in Section 3.1/3.5.
    """
    model = model.to(device)
    Xt = torch.from_numpy(X_train).permute(0, 1, 4, 2, 3).float()
    yt = torch.from_numpy(y_train).float()
    Xv = torch.from_numpy(X_val).permute(0, 1, 4, 2, 3).float().to(device)
    yv = y_val

    n_pos = max(yt.sum().item(), 1)
    n_neg = max(len(yt) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n = len(Xt)
    best_pr_auc = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_logits = val_out[0] if isinstance(val_out, tuple) else val_out
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_metrics = evaluate_probs(yv, val_prob)
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  train_loss={epoch_loss:.4f}  "
                  f"val_PR-AUC={val_metrics['PR-AUC']:.3f}  val_F1={val_metrics['F1']:.3f}")
        if val_metrics["PR-AUC"] > best_pr_auc:
            best_pr_auc = val_metrics["PR-AUC"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  -> checkpoint terbaik: epoch {best_epoch} (val_PR-AUC={best_pr_auc:.3f}), "
                  f"bobot model dikembalikan ke titik itu")
    return model


def predict_torch_model(model, X, device="cpu", batch_size=64):
    model.eval()
    Xt = torch.from_numpy(X).permute(0, 1, 4, 2, 3).float()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            xb = Xt[i:i + batch_size].to(device)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)
