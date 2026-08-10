# Bug: Seasonal 1:1 matching fails when val/test year range is a single year

**Commit:** 61f5566 | **Date:** 2026-08-09

## Symptom

```sh
python model/train.py --regime env --balance seasonal --n-train 20000 --n-val 5000 --n-test 20000 --epochs 30
```

```
seasonal 1:1 — matched 20000 pos (total 40000 samples, 0.8 % of pop)  # train OK
seasonal 1:1 — matched 0 pos (total 0 samples, 0.0 % of pop)           # val: 0
seasonal 1:1 — matched 0 pos (total 0 samples, 0.0 % of pop)           # test: 0
samples: train=20000 val=0 test=0

ValueError: Found array with 0 sample(s) (shape=(0, 58)) while a minimum of 1 is required by LogisticRegression.
```

## Root Cause

Default year splits:
```
--train 2019 2022   (4 years)
--val   2023 2023   (1 year)
--test  2023 2023   (1 year)
```

`_sample_seasonal()` filters eligible cells by `yr_mask` first, then for each positive tries to find a negative from a **different year** (`if y == target_yr: continue`). With val/test constrained to a single year (2023), every negative candidate is from the same year as the positive — zero matches.

## Proposed Fixes

### Option A (quick): Fall back to random sampling for val/test
In `train.py`, when `--balance seasonal`, train uses `_sample_seasonal()` but val/test use `_sample_cell_days()` (random). This is reasonable: the seasonal constraint is a training choice, not an evaluation requirement.

### Option B: Relax year ranges
Default val/test to multi-year ranges, e.g. `--val 2022 2023 --test 2023 2023`, but this leaks training years into validation.

### Option C: In `_sample_seasonal`, don't restrict by year at all
Remove `yr_mask` and match across all years. Then filter the output indices by year range after matching. This is harder because a positive in year X might be matched with a negative in year X (same year), which defeats the purpose.

**Recommendation:** Option A — seasonal matching is a **training data construction** technique. Val/test should remain random to avoid introducing selection bias into evaluation.

## Workaround

Pass multi-year val ranges manually:
```sh
python model/train.py --regime env --balance seasonal --val 2022 2023 --n-val 5000 ...
```
