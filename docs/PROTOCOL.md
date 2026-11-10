# ComGEN  Reproduction Protocol

This document records the protocol used for the released `stage5_full`
results.  It separates settings that the paper makes explicit from choices
that the paper leaves unspecified.

## Scope and metric

* Paper: Tong et al., "Component-aware Unsupervised Logical Anomaly Generation
  for Industrial Anomaly Detection", ICRA 2025.
* Benchmark: MVTec LOCO AD, five categories.
* Metric: image-level AUROC over `test/good` versus all test anomalies
  (`logical_anomalies` + `structural_anomalies`), following the original paper.
* The paper prints a mean of **91.2%** in Table II.  The five rounded
  per-category values (87.2, 99.1, 92.9, 79.8, 95.2) average to **90.84%**;
  the reproduction reports both numbers rather than adjusting either.
* The reproduced mean is the equal-weighted mean of the five category AUROCs.

## Paper-specified settings

Only settings that the paper states explicitly are listed here.  Every numeric
value that the paper leaves open is in "Settings the paper does not fully
specify" below.

| Module | Paper requirement | Implementation |
|---|---|---|
| MCL | Learn multiple components jointly; one 768-parameter learnable embedding per noun component; only those noun embeddings are trainable; text encoder and diffusion network frozen (Sec. III-B, Eq. 4) | `train_mcl.py`: `NounEmbeddings`; all base text/UNet/VAE parameters frozen |
| Prompts | Adjectives, conjunctions and prepositions organise the components; CLIP-derived templates add diversity (Sec. III-B) | `vendor/config.py`, `vendor/utils.py` |
| Training data and duration | One-third of the normal images per category; 6,100 epochs; constant LR 10⁻⁵; batch 16 (Sec. IV-A) | `train_mcl.py`: `--epochs 6100`, `lr=1e-5`, batch 16 |
| Augmentation | Random crop and rotation (Sec. IV-A) | implemented; exact magnitudes are our choice (below) |
| PM | Delete the related token(s) to simulate a missing semantic component (Sec. III-B, Fig. 2-I) | `generate_pairs.py`: unmodified prompt + one deletion per component |
| Generation | DDIM with T = 50; 1,000 images per category (Sec. IV-A) | `generate_pairs.py`: DDIM 50 steps; 1,000 retained (candidate pool is below) |
| RNA | Normal reference library; multi-scale features; find the closest normal reference (Sec. III-C, Eq. 5) | frozen ResNet-18 blocks 1–3; one whole normal image per pair |
| Features | ResNet-18 blocks 1/2/3; 64×64×64, 128×32×32, 256×16×16 (Sec. IV-A) | `train_eval_csda.py`: `Encoder` |
| RM | Average all 16×16 cross-attention layers and heads; component maps at t = 0; 3×3 Gaussian smoothing (Sec. III-C, Eq. 7) | `generate_pairs.py`: `BatchAttention`, `_gaussian3` |
| Normal reconstruction | Deterministic DDIM forward encoding and reverse reconstruction (Sec. III-C, Eqs. 8–9) | `Pipeline.reconstruct`; step count/guidance are below |
| Residual mask | Sum of absolute component-attention differences, then upsample (Sec. III-C, Eq. 10) | `residual = (A_norm - A_gen).abs().sum(1)`, bilinear to 512² |
| CSDA | Channel-mean difference maps multiplied across scales (Eq. 11); U-Net exchanges scale information; depthwise-separable downsampling; bilinear + 1×1 upsampling (Sec. III-D) | `train_eval_csda.py`: `CSDA` |
| Detection loss | Smooth L1 + 5 × Focal + Dice (Eq. 12; λ = 5 in Sec. IV-A) | `objective`: `smooth_l1 + 5*focal + dice` |

### 6,100 epochs vs 6,100 steps

The textual-inversion reference configuration uses `max_steps: 6100`.  ComGEN
explicitly says **6,100 epochs**.  This reproduction follows the paper: 6,100
epochs with batch 16 and `drop_last=True` over the selected one-third subset.
Each epoch has `floor(N_selected / 16) = 7` updates for every category, giving
**42,700 updates** in total.  The manifests record epochs, steps per epoch and
cumulative updates.

## Settings the paper does not fully specify

These are implementation assumptions, not published values:

* Prompts for the four categories other than breakfast box.  The paper gives the
  breakfast-box sentence as its worked example; the composed prompt follows
  that structure with the learned tokens.
* Component segmentation (which nouns are components) for those four
  categories, component initialisation words and the six CLIP-style templates.
* The one-third MCL subset selection: `random.Random(42).sample(all, round(N/3))`
  gives 117/112/124/120/120 images; the paper fixes the fraction but not the
  seed or the rounding rule.
* MCL optimizer details: Adam with betas (0.9, 0.999), no weight decay, and
  `floor(N_selected/16)=7` updates per epoch with `drop_last=True`.
* MCL augmentation magnitudes (resize 544, pad 32, rotation ±5°, crop 544 to
  512); the paper says only "random crop and rotation".
* Number of low-density candidates (2,000) and the per-prompt quota used to
  keep exactly 1,000.
* `d_max`: implemented as the per-prompt order statistic that keeps the
  quota; no numeric threshold is published.
* CFG scale 7.5.
* Eq. 6 calls `sum SSIM` a distance although SSIM is a similarity.  We use
  `sum(1 − SSIM)` and keep the smallest values.
* SSIM window 7×7, Gaussian σ = 1.5, per-reference channel range.
* RNA feature resize to 256×256 and ImageNet normalisation for every path.
* RNA reference selection: Eq. 5 writes a per-scale argmin over the memory
  bank; we select one whole normal image minimising the summed full-map L2
  distance across the three scales and use it for every scale.  This is a
  documented interpretation of an ambiguous equation.
* Whether the removed tokens are re-queried after generation.  Both the
  generated image and its reference are queried at t = 0 with the same full
  normal prompt.
* Normal reconstruction uses 50 DDIM inverse steps and 50 reverse steps with
  guidance 1; the paper gives the DDIM equations but not the step count or
  guidance value.
* Detector input 256×256; U-Net widths 64/128/256; depthwise-separable
  downsampling and bilinear + 1×1 upsampling; the paper gives the CSDA
  structure and Eq. 11 but not these widths.
* The detector is evaluated once after training; no test image participates in
  MCL, generation, pairing, thresholding, validation or checkpoint selection.
* Per-image min–max normalisation of the Eq. 10 residual to [0,1] for the
  training soft target; the raw residual is kept.
* Stage-III detector optimizer and schedule: Adam, 100 epochs, LR 10⁻⁴,
  batch 16.
* Focal γ = 2, no class weighting, continuous soft targets.
* Approximately 10% of synthetic reference groups held out (with all their
  pairs) plus the official `validation/good` images; the fraction varies with
  the number of groups.  Checkpoint selected by validation loss.
* Image-level score: maximum of the sigmoid anomaly map.

## Vendor utility vs released driver

`vendor/comgen_pipeline.py` is a shared utility and contains alternative
paths (for example a clean-latent normal reconstruction shortcut and
drop/duplicate prompt enumeration).  The released Stage-II/III driver
`generate_pairs.py` overrides the reconstruction with the true DDIM
inverse (Eqs. 8–9) and enumerates **one deletion per component** only.  The
unused alternatives in the vendor code are not part of the released run.  The
vendor and driver files are cleaned copies of the run-time sources; the
run-time hashes are retained in `protocol.json` and tabulated in
[`PROVENANCE.md`](PROVENANCE.md).  Read the driver, not the vendor utility,
for the executed protocol.

## Data isolation

* MCL and generation read only `train/good`.
* Test images, labels and real test masks are never read by the generator,
  the mask pipeline, the validation split or the checkpoint-selection code.
* The detector's validation split uses synthetic samples plus the official
  `validation/good`; the test set is evaluated once after training.
* `summarize.py` independently checks train/validation/test path overlap,
  source-group overlap, self-reference exclusion and the pair-manifest hash.
  The released audit reports no overlap and a matching hash.

## Quality checks

Generating 1,000 files is not, by itself, evidence of quality.  The released
archive contains:

* a preview contact sheet per category (`pairs_preview.jpg`);
* the reconstruction PSNR for every selected pair (`pairs_1000.json`);
* the per-prompt selected-distance thresholds (`thresholds.json`);
* the paths (but not the bytes) of the raw residual files in
  `pairs_1000.json`.

The 2,000-candidate pool, the raw residual arrays and the reconstructed
normal cache are part of the omitted generated bank (≈9.5 GB), not the
archive.  The preview sheets are sampled at fixed positions (a linear index
spacing over the selected pairs), not selected by visual quality.

## Provenance

`protocol.json` for each generation and detector run records:

* SHA-256 of the generation, MCL and detector sources;
* package versions;
* the SD-1.5 component file hashes;
* the ResNet-18 weights hash;
* the DDIM scheduler configuration;
* the prompt list, seeds, candidate counts and assumptions;
* the pair-manifest hash used by the detector.

The raw hash cache is included under `results/provenance/hash_cache/`.
Paths inside the JSON manifests are the absolute paths from the execution
machine; they are kept verbatim as run records.

**Source-hash boundary.**  `protocol.json` retains the run-time hashes of
the detector/generation sources.  The released files were cleaned after the
run (comments/docstrings removed, portable paths, a local AUROC helper), so
their hashes differ from the run-time values.  The recorded values are
retained unaltered; [`PROVENANCE.md`](PROVENANCE.md) lists every pair.  The pair-manifest SHA-256 **does** match the released
`pairs_1000.json` (checked by `summarize.py`).

## Final run

The released numbers come from `stage5_full`, whose detector protocol
points to `stage2_pairs`.  Earlier detector variants (`stage5_csda`,
`stage5_csda_prompt_aware_v5`, `stage5_csda_faithful_v4`) and the
`ComGEN_reproduction` directory are not used for the reported table.
