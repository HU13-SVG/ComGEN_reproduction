# Metrics: What Is Reported and What Is Not

## Scope

This archive covers the MVTec LOCO AD downstream image-level result only.  It
does not reproduce the paper's IS / IC-LPIPS generation metrics (Table I), the
controlled ResNet-34 classification comparison, the Table III ablation, the
diesel-engine experiments or the MVTecAD experiments.  It reports one result:
image-level AUROC over good vs all anomalies.

## Image-level AUROC (reported)

* **Test images.**  All official MVTec LOCO `test/good` (label 0),
  `test/logical_anomalies` and `test/structural_anomalies` (label 1), exactly
  as the paper's Table II all-anomaly protocol.  Total 1,568 images.
* **Per-category counts (good / logical / structural).**
  breakfast 102/83/90, juice 94/142/94, pushpins 138/91/81,
  screw bag 122/137/82, splicing 119/108/85.
* **Score.**  Maximum of the sigmoid anomaly map.  The paper does not publish
  its image-level pooling rule; this is our choice.
* **Aggregation.**  Per-category AUROC with average ranks for ties, then an
  equal-weighted mean over the five categories.
* **Diagnostic splits.**  The `logical_anomalies_auroc` and
  `structural_anomalies_auroc` fields compare good images with one anomaly
  subtype at a time.  They are derived from the same scores and are not
  separate model selections.
* **Checkpoint selection.**  Lowest validation loss on a synthetic holdout
  grouped by nearest normal (approximately 10% of reference groups), plus
  official `validation/good`.  No test label
  is used for training or selection; the test set is evaluated once after
  training.

`python tools/verify_results.py` recomputes the image-level table and both
diagnostic subtype AUROCs from the released per-image scores using only the
Python standard library.

## Comparison with the paper

The paper's Table II reports image-level AUROC per category
(87.2, 99.1, 92.9, 79.8, 95.2) and a printed mean of 91.2.  The five rounded
values average to 90.84, a 0.36 pp difference; the archive reports both the
printed 91.2 and the arithmetic mean of the released values, without adjusting
either.

## Pixel-level AU-sPRO (not reported)

The paper's second Table II metric is pixel-level AU-sPRO (mean 71.8).  This
archive does **not** report it.  AU-sPRO requires:

1. the MVTec LOCO test images and pixel ground truth (not redistributed),
2. the official per-defect `defects_config.json` saturation thresholds (not
   redistributed), and
3. the detector checkpoint (included) plus per-pixel anomaly maps.

Without all of them, an "sPRO" value would not be the paper's metric.  We do
not substitute pixel AUROC, Dice, or any other metric.  Reviewers who
download MVTec LOCO and regenerate the pair bank can run the official MVTec
LOCO evaluator; the archive does not claim the paper's 71.8 sPRO value.

## What the released numbers support

* The image-level ranking of good versus all anomalies on the official test
  split, with the released per-image scores and detector checkpoints.
* A documented reproduction gap of 21.8 pp to the printed 91.2 mean, with the
  unpublished/assumed Stage-II/III/IV choices listed in
  [`PROTOCOL.md`](PROTOCOL.md) and [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

They do **not** support a claim that ComGEN's pixel-localization performance
has been reproduced.
