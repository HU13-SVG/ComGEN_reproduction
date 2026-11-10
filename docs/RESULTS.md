# Final Results (stage5_full)

The table below is generated from the released per-image test scores.  The
`results.json` files are the verbatim run records; the numbers here are
rounded to four decimals.

| Category | Paper | Reproduced | Gap (pp) | Logical | Structural | Test n | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Breakfast box | 87.2 | 76.9976 | −10.2024 | 84.2783 | 70.2832 | 275 | 100 |
| Juice bottle | 99.1 | 87.8787 | −11.2213 | 92.0587 | 81.5641 | 330 | 89 |
| Pushpins | 92.9 | 57.7646 | −35.1354 | 55.7175 | 60.0644 | 310 | 10 |
| Screw bag | 79.8 | 65.2481 | −14.5519 | 60.2309 | 73.6305 | 341 | 92 |
| Splicing connectors | 95.2 | 59.1109 | −36.0891 | 61.5235 | 56.0455 | 312 | 83 |
| **Mean** | **91.2** | **69.4000** | **−21.8000** | — | — | **1,568** | — |

## How to read the comparison

* The paper prints 91.2 as the five-category mean.  The arithmetic mean of its
  five rounded category values is 90.84 (0.36 pp lower).  We report both and do
  not alter either.
* The reproduced mean is the equal-weighted mean of the five category AUROCs,
  computed on all official test images (good, logical, structural), following
  the paper's Table II all-anomaly protocol.  The logical/structural columns
  are diagnostic splits of the same scores, not separate model selections.
* The paper's pixel-level AU-sPRO (mean 71.8) is **not** reported here; see
  [`METRICS.md`](METRICS.md).
* The gaps are largest for pushpins (−35.14 pp) and splicing connectors
  (−36.09 pp).  The best pushpins checkpoint is epoch 10, so this category is
  sensitive to the validation-loss selection; the paper does not publish its
  Stage-III schedule, so this gap cannot be attributed to the synthesis module
  alone.
* The validation split is grouped by nearest reference, so the validation
  fraction varies with the number of reference groups.  Per-category
  synthetic train/validation counts are 912/88, 885/115, 877/123, 970/30 and
  534/466 for breakfast, juice, pushpins, screw bag and splicing; the
  splicing split is deliberately conservative because it has few reference
  groups.  No test image is in either side.

## Independent audit

`results/summary/audit_report.json` was produced by `summarize.py` and
checks, among other things:

* no train/validation index overlap;
* no train or validation path overlap with the test set;
* no synthetic source-group overlap between train and validation;
* no duplicated test paths, and label/type consistency;
* the pair-manifest SHA-256 recorded in `protocol.json` matches;
* no normal image is its own nearest reference;
* stored and independently recomputed AUROC agree for every category and
  subtype.

The audit reports `complete: true` and no overlaps.  `audit_summary.csv` is
the same information in table form.

## Reproduction command

The final run is identified by `results/stage5_full/*/protocol.json`:

```text
masks  = .../stage2_pairs
output = .../stage5_full
epochs = 100
batch  = 16
lr     = 1e-4
seed   = 42
evaluate = true
```

`results/stage2_pairs/*/protocol.json` records the generation provenance,
and `pairs_1000.json` records every selected pair, prompt group, seed,
reference and mask path.
