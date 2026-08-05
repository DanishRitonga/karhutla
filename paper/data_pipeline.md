# Data Processing Pipeline — Prediksi Hotspot Karhutla Riau

```mermaid
flowchart TB
    subgraph S0["Stage 0 — Build Grid"]
        A1["Riau admin boundary shapefile"] --> A2["Reproject to equal-area projection"]
        A2 --> A3["Create 5 km × 5 km fishnet"]
        A3 --> A4["Mask to Riau boundary"]
        A4 --> A5[("Grid: 84 cols × 90 rows<br/>~3,484 active cells")]
    end

    subgraph S1["Stage 1 — Resample Sources onto Grid"]
        subgraph LAB["1a. FIRMS VIIRS (LABEL)"]
            B1["Raw CSV: lat, lon, date, confidence, FRP"] --> B2["Filter conf ∈ {n, h}"]
            B2 --> B3["Spatial join → cell index"]
            B3 --> B4["GroupBy cell + date → count"]
            B4 --> B5["k=2 threshold over [t+1…t+7]"]
            B5 --> B6[("y: labels[cell, date]")]
        end

        subgraph ERA5["1b. ERA5-Land (9 km, hourly)"]
            C1["NetCDF: t2m, d2m, u10, v10, swvl1, swvl2, tp, ssr"] --> C2["Bilinear interpolate 9 km → 5 km"]
            C2 --> C3["Aggregate to daily: mean / sum"]
            C3 --> C4[("8 channels")]
        end

        subgraph CHIRPS["1c. CHIRPS v3 SAT (5 km, daily)"]
            D1["GeoTIFF: daily rainfall mm"] --> D2["Nearest-neighbor to grid"]
            D2 --> D3[("1 channel")]
        end

        subgraph S1G["1d. Sentinel-1 SAR (10 m, 6–12 day)"]
            E1["GEE: COPERNICUS/S1_GRD"] --> E2["Speckle filter + VV/VH compute"]
            E2 --> E3["Median resample 10 m → 5 km"]
            E3 --> E4["Gap fill ≤ 8 days + availability mask"]
            E4 --> E5[("2+1 channels")]
        end

        subgraph DW["1e. Dynamic World (10 m, Sentinel-2 derived)"]
            F1["GEE: GOOGLE/DYNAMICWORLD/V1"] --> F2["9 class probabilities per overpass"]
            F2 --> F3["Mean probability per cell per day"]
            F3 --> F4["Cloud-masked days → NaN"]
            F4 --> F5["Gap fill per O4"]
            F5 --> F6[("9 channels")]
        end

        subgraph PEAT["1f. Peta Gambut (static)"]
            G1["Shapefile: peat_thick ranges"] --> G2["Parse midpoint depth, drain==0"]
            G2 --> G3["Rasterize to 5 km grid"]
            G3 --> G4["Broadcast to all dates"]
            G4 --> G5[("1 channel")]
        end
    end

    A5 --> B2
    A5 --> C2
    A5 --> D2
    A5 --> E2
    A5 --> F2
    A5 --> G3

    B6 --> H1
    C4 --> H1
    D3 --> H1
    E5 --> H1
    F6 --> H1
    G5 --> H1

    subgraph S2["Stage 2 — Assemble Tensor"]
        H1[("Unified table: cell_idx, date, 22 channels")] --> H2["For each target day t:<br/>extract sliding window"]
        H2 --> H3["X[t] = features[t-13…t, 15×15 patch]<br/>y[t] = label[t+1…t+7] via k=2"]
        H3 --> H4[("Tensor: [N_samples, 14, 15, 15, C≈22]")]
    end

    subgraph S3["Stage 3 — Split"]
        H4 --> I1["train: t ∈ 2019–2022<br/>test:  t ∈ 2023"]
        I1 --> I2["(seasonal scope narrowed per O3)"]
    end

    subgraph S4["Stage 4 — Impute + Normalize"]
        I2 --> J1["Impute gaps (per channel, after split)"]
        J1 --> J2["Normalize: μ, σ from train only<br/>apply to train, test"]
    end

    subgraph S5["Stage 5 — Model"]
        J2 --> K1["Tabular: aggregate → feature vector → LightGBM"]
        J2 --> K2["Spatiotemporal: raw tensor → ConvLSTM / Temporal Transformer"]
    end

    B6 -. "y only, never X<br/>(environmental regime)" .-> H3

    style S0 fill:#f5f0ff,stroke:#7c3aed,color:#1a1a2e
    style S1 fill:#f5f0ff,stroke:#7c3aed,color:#1a1a2e
    style S2 fill:#f0f7ff,stroke:#2563eb,color:#1a1a2e
    style S3 fill:#f0f7ff,stroke:#2563eb,color:#1a1a2e
    style S4 fill:#f0f7ff,stroke:#2563eb,color:#1a1a2e
    style S5 fill:#fff0f5,stroke:#e94560,color:#1a1a2e
    style LAB fill:#e8f4f8,stroke:#0ea5e9,color:#1a1a2e
    style ERA5 fill:#e8f4f8,stroke:#0ea5e9,color:#1a1a2e
    style CHIRPS fill:#e8f4f8,stroke:#0ea5e9,color:#1a1a2e
    style S1G fill:#e8f4f8,stroke:#0ea5e9,color:#1a1a2e
    style DW fill:#e8f4f8,stroke:#0ea5e9,color:#1a1a2e
    style PEAT fill:#e8f4f8,stroke:#0ea5e9,color:#1a1a2e
```

## Channel count breakdown

| Source | Channels | Type |
|---|---|---|
| ERA5-Land | 8 (t2m, d2m, u10, v10, swvl1, swvl2, tp, ssr) | Dinamis harian |
| CHIRPS v3 SAT | 1 (precip) | Dinamis harian |
| Sentinel-1 | 2 (VV, VH) + 1 (availability mask) | Dinamis jarang |
| Dynamic World | 9 (water, trees, grass, flooded_veg, crops, shrub, built, bare, snow_ice) | Dinamis temporal, rawan cloud |
| Peat depth | 1 (midpoint meters) | Statik |
| **Total C** | **≈22** | |

## Anti-leakage rules (hard)

| Rule | Check |
|---|---|
| L1 | Temporal split: test always in future of train |
| L2 | No overlap between input [t-13…t] and target [t+1…t+7] |
| L4 | Impute AFTER split |
| L5 | Normalize with train μ/σ only |
| L8 | Tune inside train period (CV), test 2023 touched only for final report |
