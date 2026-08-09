"""
models.py
=========
Implements every model in Table 3 of the paper:

  Category         Model                Purpose
  Naive             Persistence          Reference baseline
  Meteorological    Logistic Regression  Simple fire-danger baseline
  Tabular           LR, RF, LightGBM     Machine learning baseline
  Spatiotemporal    ConvLSTM             Main model
  Spatiotemporal    Temporal Transformer Main model

The two "main models" are implemented in PyTorch and consume the raw
[B, T=14, H=15, W=15, C] tensor (channel-masked to the environmental or
operational regime, see data.py). The tabular / meteorological baselines
consume the flattened feature matrix produced by data.to_tabular().
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


# ----------------------------------------------------------------------
# 1. Naive persistence baseline
# ----------------------------------------------------------------------
class PersistenceBaseline:
    """
    Predicts positive if the recent hotspot-history channel (fire_history,
    operational regime only) shows recent activity. This is the "reference
    baseline" of Table 3 -- it exists to show how much of a model's skill is
    just hotspot persistence rather than genuine environmental prediction,
    which is exactly the confound the paper's environmental/operational
    split is designed to isolate (Section 3.1 / Abstract).
    """

    def __init__(self, fire_history_channel_pos_in_tabular: int | None):
        self.col = fire_history_channel_pos_in_tabular

    def fit(self, X_tab, y):
        return self  # no fitting needed

    def predict_proba(self, X_tab):
        if self.col is None:
            # environmental regime has no fire-history channel -> persistence
            # is undefined; fall back to the base rate (uninformative)
            p = np.full(len(X_tab), 0.5, dtype=np.float32)
        else:
            raw = X_tab[:, self.col]
            p = 1 / (1 + np.exp(-(raw - raw.mean()) / (raw.std() + 1e-6)))
        return np.stack([1 - p, p], axis=1)


# ----------------------------------------------------------------------
# 2. Meteorological logistic-regression baseline (uses only ERA5 met vars)
# ----------------------------------------------------------------------
def make_meteorological_lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced")


# ----------------------------------------------------------------------
# 3. Tabular ML baselines
# ----------------------------------------------------------------------
def make_tabular_lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def make_tabular_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0,
    )


def make_tabular_lightgbm(scale_pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        num_leaves=31, scale_pos_weight=scale_pos_weight,
        subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1,
    )


# ----------------------------------------------------------------------
# 4. ConvLSTM (main spatiotemporal model)
# ----------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """Standard convolutional LSTM cell (Shi et al. 2015, cited as [13])."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, h, w, device):
        z = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return z, z.clone()


class ConvLSTMHotspot(nn.Module):
    """
    Stacked ConvLSTM -> global average pool -> MLP head -> single logit
    (binary cell-day hotspot risk, Section 3.1).

    Input:  [B, T, C, H, W]
    Output: [B] logits
    """

    def __init__(self, in_channels, hidden_channels=(64, 32), kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_channels
        for hc in hidden_channels:
            layers.append(ConvLSTMCell(prev, hc, kernel_size))
            prev = hc
        self.cells = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        device = x.device
        states = [cell.init_state(B, H, W, device) for cell in self.cells]
        for t in range(T):
            inp = x[:, t]
            for li, cell in enumerate(self.cells):
                h, c = states[li]
                h, c = cell(inp, h, c)
                states[li] = (h, c)
                inp = h
        last_h = states[-1][0]                   # [B, hidden, H, W]
        pooled = last_h.mean(dim=(2, 3))          # global average pool
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


# ----------------------------------------------------------------------
# 5. Temporal Transformer (main spatiotemporal model)
# ----------------------------------------------------------------------
import torchvision.models as tv_models


class ResNetFrameEncoder(nn.Module):
    """Replace toy Conv2d->AvgPool with a ResNet-18 backbone."""

    def __init__(self, in_channels, d_model):
        super().__init__()
        self.backbone = tv_models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                         stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, d_model)

    def forward(self, x):
        return self.backbone(x)


class SelfAttnBlock(nn.Module):
    """Transformer encoder block implemented manually so attention weights
    are easy to retrieve for the attention-visualization interpretability
    step described in Section 3.5."""

    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_w = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_w  # attn_w: [B, T, T]


class TemporalTransformerHotspot(nn.Module):
    """
    Per-frame CNN encoder -> learned positional embedding -> a stack of
    self-attention blocks over the 14-day sequence -> mean-pooled
    classification head -> single logit.

    Input:  [B, T, C, H, W]
    Output: logits [B], plus last-layer attention weights [B, T, T] for
    interpretability (interpret.py).
    """

    def __init__(self, in_channels, seq_len=14, d_model=256, n_heads=4,
                 n_layers=2, dim_ff=512, dropout=0.2):
        super().__init__()
        self.frame_encoder = ResNetFrameEncoder(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [SelfAttnBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, return_attn=False):
        B, T, C, H, W = x.shape
        frames = x.reshape(B * T, C, H, W)
        z = self.frame_encoder(frames).reshape(B, T, -1)
        z = z + self.pos_embed[:, :T, :]
        attn_last = None
        for block in self.blocks:
            z, attn_last = block(z)
        pooled = self.dropout(z.mean(dim=1))
        logits = self.head(pooled).squeeze(-1)
        if return_attn:
            return logits, attn_last
        return logits
