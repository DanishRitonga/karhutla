"""
data.py
=======
Synthetic data generator that mimics the multimodal spatiotemporal tensor
described in "Predicting Hotspot Emergence in Riau Province" (Section 3.2-3.3).

Real satellite products (FIRMS VIIRS, ERA5-Land, CHIRPS v3, Sentinel-1 SAR,
Google Dynamic World, peat map) are NOT fetched here -- there is no network
access to Earth-observation archives in this environment. Instead we build a
spatially- and temporally-correlated synthetic field that reproduces the
*shape* and *semantics* of the real tensor so the model architectures in
models.py can be built, trained, and sanity-checked end to end:

    Tensor shape: [N, T=14, H=15, W=15, C]
    C = 21 environmental channels + 1 fire-history channel (operational regime)

Channel layout (index -> name), grouped exactly as in Table 1/2 of the paper:

  0-7   ERA5-Land daily aggregates : t2m, d2m, u10, v10, swvl1, swvl2, ssr, tp
  8     CHIRPS v3 (SAT) rainfall   : chirps_precip
  9-11  Sentinel-1 SAR             : vv_db, vh_db, sar_available_mask
  12-19 Dynamic World land cover   : water, trees, grass, flooded_veg,
                                     crops, shrub_scrub, built, bare
  20    Peatland map (static)      : peat_depth_m
  21    Fire history (operational only) : hotspot_count_lag

Channels 0-20 (21 channels) = ENVIRONMENTAL_CHANNELS regime.
Channel 21 (+ all of the above)  = OPERATIONAL_CHANNELS regime (22 channels),
matching the paper's "C ~= 22" and the environmental-vs-operational split in
Section 3.2 ("environmental regime excludes fire history ... operational
regime includes fire history and establishes a practical performance ceiling").

Label rule: a cell-day is positive if a synthetic fire-risk score integrated
over the 7-day target window exceeds a threshold at least twice (k=2
persistence rule, Section 3.1/Table 2), reproducing the severe class
imbalance the paper calls out.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

PATCH = 15          # 15x15 spatial patch (paper: convolutional patch size)
T_IN = 14            # 14 antecedent days
HORIZON = 7           # 7-day forecast window
CENTER = PATCH // 2

CHANNEL_NAMES = [
    "t2m", "d2m", "u10", "v10", "swvl1", "swvl2", "ssr", "tp",   # ERA5-Land (8)
    "chirps_precip",                                              # CHIRPS (1)
    "sar_vv", "sar_vh", "sar_available",                          # Sentinel-1 (3)
    "dw_water", "dw_trees", "dw_grass", "dw_flooded_veg",
    "dw_crops", "dw_shrub_scrub", "dw_built", "dw_bare",          # Dynamic World (8)
    "peat_depth",                                                 # Peat map (1)
    "hotspot_count_lag",                                          # Fire history (1)
]
N_CHANNELS = len(CHANNEL_NAMES)                # 22
ENV_CHANNELS = list(range(0, N_CHANNELS - 1))  # 0..20  (21 channels)
OPERATIONAL_CHANNELS = list(range(0, N_CHANNELS))  # 0..21 (22 channels)
FIRE_HISTORY_IDX = N_CHANNELS - 1


def _smooth_field(rng, h, w, sigma=3.0):
    """Spatially correlated random field via Gaussian smoothing of white noise."""
    raw = rng.standard_normal((h, w))
    return gaussian_filter(raw, sigma=sigma, mode="wrap")


def _ar1_series(rng, n_days, phi=0.85, scale=1.0):
    """Temporally correlated AR(1) series (mimics slow-varying weather)."""
    noise = rng.standard_normal(n_days) * scale
    series = np.zeros(n_days)
    series[0] = noise[0]
    for t in range(1, n_days):
        series[t] = phi * series[t - 1] + noise[t]
    return series


def generate_riau_fields(n_days=760, grid_h=60, grid_w=60, seed=42, include_seasonal=True):
    """
    Build full spatiotemporal raster fields [n_days, grid_h, grid_w, N_CHANNELS]
    plus a static peat-depth map and static land-cover fractions, with a
    physically-motivated latent fire-risk field used only to generate labels
    (it is NOT one of the model input channels, mirroring a real early-warning
    setting where the model never sees the ground truth risk process itself).

    n_days=760 ~= 2 dry-season-heavy years, kept short for tractable CPU
    training in this demo; the paper's real design uses 2019-2023.

    include_seasonal=False drops the annual dry/wet sinusoid and ENSO-like
    modulation entirely (pure AR(1)/spatial noise instead). Used by
    real_data.py: when the channels are paired with REAL VIIRS labels, a
    fixed calendar-phase sinusoid risks accidentally overlapping with Riau's
    real dry-season peak (~day-of-year 240-270) and injecting a coincidental,
    non-physical "signal" into what is supposed to be an uninformative
    placeholder. See README for the empirical check that motivated this.
    """
    rng = np.random.default_rng(seed)

    # --- static layers ---------------------------------------------------
    peat_depth = np.clip(_smooth_field(rng, grid_h, grid_w, sigma=6.0) * 0.8 + 1.5, 0.0, 4.0)
    dw_static_logits = np.stack(
        [_smooth_field(rng, grid_h, grid_w, sigma=4.0) for _ in range(8)], axis=-1
    )  # 8 Dynamic World classes, static base map perturbed daily below
    peat_prone = (peat_depth > 1.8).astype(np.float32)  # used to bias risk

    fields = np.zeros((n_days, grid_h, grid_w, N_CHANNELS), dtype=np.float32)
    risk_latent = np.zeros((n_days, grid_h, grid_w), dtype=np.float32)

    # seasonal (annual) dry/wet cycle -> ENSO-like modulation (paper Sec.1)
    day_idx = np.arange(n_days)
    if include_seasonal:
        seasonal = np.sin(2 * np.pi * day_idx / 365.0 - np.pi / 2)  # peak dryness mid-cycle
        enso_bump = 0.6 * np.sin(2 * np.pi * day_idx / 900.0)
    else:
        # calendar-decoupled: slow AR(1) drift instead of a fixed-phase annual
        # cycle, so there is no day-of-year periodicity to coincidentally
        # align with real fire seasonality
        seasonal = _ar1_series(rng, n_days, phi=0.98, scale=0.10)
        enso_bump = np.zeros(n_days)

    # per-pixel AR(1) driver for soil moisture anomaly (spatially varying phase)
    spatial_phase = _smooth_field(rng, grid_h, grid_w, sigma=8.0)

    for t in range(n_days):
        dryness = seasonal[t] + enso_bump[t]
        # meteorology: hotter/drier when 'dryness' is high
        t2m = 26 + 4 * dryness + _smooth_field(rng, grid_h, grid_w, sigma=5) * 1.2
        d2m = t2m - (6 + 3 * dryness) + _smooth_field(rng, grid_h, grid_w, sigma=5) * 0.8
        u10 = _smooth_field(rng, grid_h, grid_w, sigma=5) * 1.5
        v10 = _smooth_field(rng, grid_h, grid_w, sigma=5) * 1.5
        swvl1 = np.clip(0.35 - 0.12 * dryness + 0.05 * spatial_phase
                         + _smooth_field(rng, grid_h, grid_w, sigma=4) * 0.03, 0.02, 0.55)
        swvl2 = np.clip(swvl1 * 0.9 + 0.02 * _smooth_field(rng, grid_h, grid_w, sigma=4), 0.02, 0.55)
        ssr = np.clip(180 + 60 * dryness + _smooth_field(rng, grid_h, grid_w, sigma=5) * 15, 0, 320)
        tp = np.clip(6 - 5 * dryness + _smooth_field(rng, grid_h, grid_w, sigma=5) * 2, 0, None)
        chirps = np.clip(tp * 0.9 + _smooth_field(rng, grid_h, grid_w, sigma=4) * 0.8, 0, None)

        sar_vv = -9 - 4 * (swvl1 - 0.2) + _smooth_field(rng, grid_h, grid_w, sigma=4) * 0.6
        sar_vh = sar_vv - 6 + _smooth_field(rng, grid_h, grid_w, sigma=4) * 0.5
        sar_avail = (rng.random((grid_h, grid_w)) > 0.15).astype(np.float32)  # ~6-12 day revisit gaps

        dw_noise = dw_static_logits + _smooth_field(rng, grid_h, grid_w, sigma=6.0)[..., None] * 0.05
        dw_probs = np.exp(dw_noise) / np.exp(dw_noise).sum(axis=-1, keepdims=True)

        fields[t, ..., 0] = t2m
        fields[t, ..., 1] = d2m
        fields[t, ..., 2] = u10
        fields[t, ..., 3] = v10
        fields[t, ..., 4] = swvl1
        fields[t, ..., 5] = swvl2
        fields[t, ..., 6] = ssr
        fields[t, ..., 7] = tp
        fields[t, ..., 8] = chirps
        fields[t, ..., 9] = sar_vv
        fields[t, ..., 10] = sar_vh
        fields[t, ..., 11] = sar_avail
        fields[t, ..., 12:20] = dw_probs
        fields[t, ..., 20] = peat_depth

        # latent risk: dry soil + heat + peat + shrub/crop cover (proxy for
        # slash-and-burn adjacent land use) raise risk; rainfall suppresses it
        shrub_crop = dw_probs[..., 4] + dw_probs[..., 5]  # crops + shrub_scrub
        risk_latent[t] = (
            2.6 * (0.35 - swvl1)
            + 0.05 * (t2m - 26)
            + 1.1 * peat_prone
            + 1.3 * shrub_crop
            - 0.15 * chirps
        )

    # smooth risk slightly in time (fire buildup is not instantaneous)
    risk_latent = uniform_filter1d(risk_latent, size=3, axis=0, mode="nearest")

    return fields, risk_latent, peat_depth


def _daily_events(risk_latent, rng):
    """
    Bernoulli draw per (day, row, col) representing a "true" VIIRS-like
    hotspot detection event, driven by the latent risk score. Both the
    future-window label and the past-window fire-history channel are
    derived from this SAME event process, so the fire-history channel is a
    genuine (causal, past-only) proxy for the same phenomenon the label
    describes -- exactly the persistence confound the paper's environmental
    vs. operational split is designed to isolate (Abstract, Section 3.1).
    """
    prob_daily = 1 / (1 + np.exp(-(risk_latent - 3.2)))  # logistic squashing, tuned for imbalance
    return rng.random(risk_latent.shape) < prob_daily


def _labels_from_draws(draws, k=2, horizon=HORIZON):
    """Positive if >=k events occur in the *future* window (t+1..t+horizon)."""
    n_days = draws.shape[0]
    labels = np.zeros(draws.shape, dtype=np.int64)
    for t in range(n_days - horizon):
        window_count = draws[t + 1: t + 1 + horizon].sum(axis=0)
        labels[t] = (window_count >= k).astype(np.int64)
    return labels


def _fire_history_from_draws(draws, window=14):
    """
    Causal rolling count of events in the *past* `window` days (including
    day t) -- this becomes the hotspot_count_lag channel used only in the
    operational regime. Strictly backward-looking, so no future information
    leaks into it (the paper's leakage checklist, Section 3.5).
    """
    n_days = draws.shape[0]
    cum = np.concatenate([np.zeros((1,) + draws.shape[1:]), np.cumsum(draws, axis=0)], axis=0)
    start = np.clip(np.arange(n_days) - window + 1, 0, None)
    end = np.arange(n_days) + 1
    return (cum[end] - cum[start]).astype(np.float32)


def build_dataset(n_samples=3000, n_days=760, grid_h=60, grid_w=60, seed=42):
    """
    Sample N valid (day, row, col) cell-day instances, extract the
    [T_IN, PATCH, PATCH, C] tensor and label for each, and return everything
    needed for both the tabular baselines and the spatiotemporal models.

    Returns
    -------
    X : float32 array [N, T_IN, PATCH, PATCH, N_CHANNELS]
    y : int64  array [N]
    day_index : int array [N]   (used for the temporal train/test split)
    """
    rng = np.random.default_rng(seed)
    fields, risk_latent, _ = generate_riau_fields(n_days, grid_h, grid_w, seed=seed)

    draws = _daily_events(risk_latent, rng)
    labels = _labels_from_draws(draws, k=2, horizon=HORIZON)
    fields[..., FIRE_HISTORY_IDX] = _fire_history_from_draws(draws, window=T_IN)

    valid_t = np.arange(T_IN, n_days - HORIZON)
    valid_r = np.arange(CENTER, grid_h - CENTER)
    valid_c = np.arange(CENTER, grid_w - CENTER)

    # oversample positive cell-days so the demo has enough positives to
    # learn from and evaluate, while keeping a realistic-ish imbalance
    flat_labels = labels[valid_t][:, valid_r][:, :, valid_c]
    pos_idx = np.argwhere(flat_labels == 1)
    neg_idx = np.argwhere(flat_labels == 0)
    n_pos = min(len(pos_idx), n_samples // 4)
    n_neg = n_samples - n_pos
    sel_pos = pos_idx[rng.choice(len(pos_idx), size=n_pos, replace=False)]
    sel_neg = neg_idx[rng.choice(len(neg_idx), size=min(n_neg, len(neg_idx)), replace=False)]
    sel = np.concatenate([sel_pos, sel_neg], axis=0)
    rng.shuffle(sel)

    N = len(sel)
    X = np.zeros((N, T_IN, PATCH, PATCH, N_CHANNELS), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    day_index = np.zeros(N, dtype=np.int64)

    for i, (ti, ri, ci) in enumerate(sel):
        t, r, c = valid_t[ti], valid_r[ri], valid_c[ci]
        X[i] = fields[t - T_IN + 1: t + 1, r - CENTER: r + CENTER + 1, c - CENTER: c + CENTER + 1, :]
        y[i] = labels[t, r, c]
        day_index[i] = t

    return X, y, day_index


def temporal_split(day_index, test_frac=0.2):
    """
    Reproduce the paper's strict temporal split (train on earlier days
    ~ 2019-2022, test on the latest days ~ 2023). Returns boolean masks.
    """
    cutoff = np.quantile(day_index, 1 - test_frac)
    train_mask = day_index <= cutoff
    test_mask = ~train_mask
    return train_mask, test_mask


def to_tabular(X, channels):
    """
    Collapse [N, T_IN, PATCH, PATCH, C] -> tabular feature matrix for the
    LR / RF / LightGBM baselines, using per-channel summary statistics
    (temporal mean/std/last-day-center-pixel + spatial mean of the patch),
    since gradient-boosted trees / logistic regression cannot consume raw
    spatiotemporal tensors directly (Table 3: "Tabular" baseline category).

    Fase 8: also appends a small, fixed set of physically-motivated derived
    features -- NOT a general feature-engineering expansion, deliberately
    scoped to avoid the "did it improve because of the features or the
    model?" confound raised in the design log. Only added when the
    relevant source channel is present in `channels`:

      rain_cum_7d / rain_cum_14d   sum of chirps_precip over the last 7 /
                                    all 14 antecedent days (patch-mean series)
      temp_mean_7d                 mean t2m over the last 7 antecedent days
      soilm_mean_7d                mean swvl1 over the last 7 antecedent days

    ConvLSTM/Transformer never see these -- they still get the raw 14-day
    window unmodified, per the design-log principle that engineered
    features are for models that can't learn temporal patterns themselves.
    (hotspot_count_7d is deliberately NOT computed here -- see
    real_data.py's fire_history_from_counts(window=7): the fire-history
    *channel* is already a 14-day rolling count, so a correct 7-day version
    needs the raw daily counts, not this already-aggregated tensor.)
    """
    Xc = X[..., channels]  # [N, T, H, W, C]
    center = Xc[:, :, CENTER, CENTER, :]           # [N, T, C] center-pixel series
    patch_mean = Xc.mean(axis=(2, 3))               # [N, T, C] spatial mean per day

    feats = np.concatenate([
        center.mean(axis=1), center.std(axis=1), center[:, -1, :],
        patch_mean.mean(axis=1), patch_mean.std(axis=1),
    ], axis=1)  # [N, 5*C]
    names = []
    for prefix in ["center_mean", "center_std", "center_last", "patch_mean_mean", "patch_mean_std"]:
        names += [f"{prefix}__{CHANNEL_NAMES[c]}" for c in channels]

    extra_feats, extra_names = [], []
    for pos, c in enumerate(channels):
        cname = CHANNEL_NAMES[c]
        series = patch_mean[:, :, pos]  # [N, T]
        if cname == "chirps_precip":
            extra_feats.append(series[:, -7:].sum(axis=1, keepdims=True))
            extra_names.append("rain_cum_7d")
            extra_feats.append(series.sum(axis=1, keepdims=True))
            extra_names.append("rain_cum_14d")
        elif cname == "t2m":
            extra_feats.append(series[:, -7:].mean(axis=1, keepdims=True))
            extra_names.append("temp_mean_7d")
        elif cname == "swvl1":
            extra_feats.append(series[:, -7:].mean(axis=1, keepdims=True))
            extra_names.append("soilm_mean_7d")
    if extra_feats:
        feats = np.concatenate([feats] + extra_feats, axis=1)
        names += extra_names

    return feats.astype(np.float32), names
