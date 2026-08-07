"""
interpret.py
============
"To improve model interpretability, SHAP analysis is conducted for tabular
models, whereas attention visualization is employed for the spatiotemporal
architectures." (Section 3.5)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import torch


def shap_summary_for_lightgbm(model, X_sample, feature_names, out_path, max_display=15):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):  # binary classifiers sometimes return [neg, pos]
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([feature_names[i] for i in order][::-1], mean_abs[order][::-1], color="#c0392b")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("LightGBM feature importance (SHAP)\nTabular / operational regime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [feature_names[i] for i in order], mean_abs[order]


def attention_heatmap(model, X_one, device, out_path):
    """
    Runs a single sample through the Temporal Transformer and plots the
    last-layer self-attention matrix over the 14 input days -- this is the
    "attention visualization ... for the spatiotemporal architectures" step
    from Section 3.5, showing which antecedent days the model weighted most
    heavily when raising or lowering 7-day hotspot risk.
    """
    model.eval()
    x = torch.from_numpy(X_one[None]).permute(0, 1, 4, 2, 3).float().to(device)
    with torch.no_grad():
        logits, attn = model(x, return_attn=True)
    prob = torch.sigmoid(logits).item()
    attn_map = attn[0].cpu().numpy()  # [T, T]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(attn_map, cmap="inferno", aspect="auto")
    ax.set_xlabel("attended-to day (t-13 .. t)")
    ax.set_ylabel("query day (t-13 .. t)")
    ax.set_title(f"Temporal Transformer self-attention\npredicted 7-day risk = {prob:.2f}")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return prob, attn_map
