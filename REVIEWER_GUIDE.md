# Reviewer Guide

## Claim

This archive is a clean-room, paper-guided re-implementation of ComGEN
(ICRA 2025) on MVTec LOCO AD: multi-component textual inversion (Stage I),
prompt modification + low-density sampling (Stage II), reference-neighbour
association + residual masks (Stage III), and the CSDA detector (Stage IV).
It reports **image-level AUROC**; it is not a bit-exact reproduction of the
authors' code and it does not report the paper's pixel-level AU-sPRO.

## Where to check paper adherence

* [`docs/PROTOCOL.md`](docs/PROTOCOL.md) lists only settings the paper states
  explicitly, then separately lists every value the paper leaves open.
* [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) records prompts, `d_max`, CFG,
  attention alignment, mask normalisation, detector schedule and validation
  choices that the paper does not publish.
* [`docs/METRICS.md`](docs/METRICS.md) defines the reported image-level metric
  and explains why sPRO is not reported.
* [`docs/PROVENANCE.md`](docs/PROVENANCE.md) lists run-time vs released source
  hashes and the redistributed/omitted assets.
* Code map:
  * `train_mcl_v6.py` — Stage I, 6,100 epochs, batch 16, LR 10⁻⁵;
  * `generate_pairs_v6.py` — Stage II+III (2,000-candidate pool is our choice;
    the paper fixes 1,000 retained images);
  * `train_eval_csda_v6.py` — Stage IV (100 epochs is our choice);
  * `v6_vendor/` — SD-1.5, prompt, attention and SSIM utilities.

## Result evidence

* `results/stage5_full_v6/<category>/result.json` — image, logical and
  structural AUROC, best epoch, test count.
* `results/stage5_full_v6/<category>/test_scores.csv` — per-image scores.
* `results/stage5_full_v6/<category>/protocol.json` — args, assumptions,
  run-time source/model/manifest hashes, `test_used_for_selection=false`.
* `results/stage5_full_v6/<category>/split.json` and `history.json` — exact
  train/validation split and 100-epoch trajectory.
* `results/summary/audit_report.json` — independent check of AUROC, split
  overlap, self-reference exclusion and pair-manifest hash.
* `results/SHA256SUMS` — integrity of the released evidence.

Run:

```bash
python tools/verify_results.py          # recompute the released table
python summarize_v6.py                  # heavier split/provenance audit
```

`verify_results.py` uses only the Python standard library and also checks the
logical/structural diagnostic splits against `result.json`.

## Numbers to read carefully

* Reproduced mean: **69.40%** (equal-weighted five-category mean).
* The paper prints **91.2%**; the arithmetic mean of its five rounded
  category values is **90.84%** (a 0.36 pp difference).  We report both and do
  not adjust either.
* The reproduced values are on all official test images (good, logical,
  structural), following the paper's Table II all-anomaly protocol.
  Logical/structural columns are diagnostic splits of the same scores.
* The paper's pixel-level AU-sPRO (mean 71.8) is **not** reproduced.  No
  substitute pixel metric is claimed; see `docs/METRICS.md`.
* In the controlled classification comparison the original paper reports
  **52.2%**; the accompanying manuscript's harness reports **52.4%**, but that
  harness is not in this archive and 52.4% is not independently verifiable
  here.

## Reproducibility and provenance boundary

* The 2,000-candidate pools and 1,000 generated pairs per category (≈9.5 GB)
  are excluded.  `pairs_1000.json` records every selected pair, prompt group,
  seed, reference and mask path; `pairs_preview.jpg` shows fixed-position
  normal/generated/mask examples.
* Detector checkpoints (`best.pth`) are included, but retraining the detector
  requires regenerating the pair bank.
* The run-time hashes in `protocol.json` are retained unaltered.  The vendor
  algorithm modules match the released files; the three top-level drivers were
  cleaned after the run, so their released hashes differ.  The exact table is
  in `docs/PROVENANCE.md`.
* The pair-manifest hash matches the released `pairs_1000.json`; the audit
  re-checks it.
