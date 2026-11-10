# Assumptions and Paper Ambiguities

ComGEN publicly specifies the main architecture and training schedule, but not
every implementation detail.  This file records the choices that are **not**
fixed by the paper.  They are implemented in code and serialized in the
per-run `protocol.json` files.

## Prompts and component definitions

Only the breakfast-box sentence is given verbatim in the paper.  The other
four prompts are reconstructed from the documented components.  Tokens are
enclosed in angle brackets.

| Category | Components (token; initialisation) | Base prompt |
|---|---|---|
| Breakfast box | box (`<bb-box>`; box), fruits (`<bb-fruits>`; fruit), cereals (`<bb-cereals>`; cereal), toppings (`<bb-toppings>`; topping) | a `<bb-box>` containing `<bb-fruits>` on the left and `<bb-cereals>` with `<bb-toppings>` on the right |
| Juice bottle | bottle (`<jb-bottle>`; bottle), liquid (`<jb-liquid>`; juice), label (`<jb-label>`; label), icon (`<jb-icon>`; cherry) | a `<jb-bottle>` filled with `<jb-liquid>` with a `<jb-label>` showing a `<jb-icon>` |
| Pushpins | grid (`<pp-grid>`; compartment), pushpins (`<pp-pin>`; pin) | a `<pp-grid>` with compartments each holding one `<pp-pin>` |
| Screw bag | bag (`<sb-bag>`; bag), screws (`<sb-screw>`; screw), nuts (`<sb-nut>`; nut), washers (`<sb-washer>`; washer) | a `<sb-bag>` containing `<sb-screw>` and `<sb-nut>` and `<sb-washer>` |
| Splicing connectors | connectors (`<sc-conn>`; connector), cable (`<sc-cable>`; wire), clamps (`<sc-clamp>`; terminal) | two `<sc-conn>` linked by a `<sc-cable>` with `<sc-clamp>` |

The implementation config is `vendor/config.py`; the breakfast-box
initialisation words were explicitly set to box/fruit/cereal/topping in
`train_mcl.py` to match the paper's example.  The six CLIP-style templates
are listed in the same vendor module.

## Stage I: MCL

* **One-third subset.** `random.Random(seed=42).sample(all_paths, round(N/3))`.
  Counts: 117 / 112 / 124 / 120 / 120 for breakfast / juice / pushpins /
  screw bag / splicing connectors.
* **Epochs and updates.** 6,100 epochs, batch 16, `drop_last=True`; 7 updates
  per epoch and 42,700 total updates.
* **Precision.** Text encoder and VAE in fp32; frozen UNet in bf16.
* **Augmentation.** Resize 544², reflect padding 32, random rotation ±5°,
  center crop 544², random crop 512².  The magnitudes are assumptions.
* **Initialisation.** Each new token is initialised from an existing word;
  no other text-encoder parameters are trainable.

## Stage II: prompt modification and low-density sampling

* **Candidate count.** 2,000 candidates per category; the paper specifies
  1,000 selected images but not the candidate pool size.
* **Prompt groups.** Unmodified prompt plus one deletion per component:
  5/5/3/5/4 groups for breakfast / juice / pushpins / screw bag / splicing
  connectors.  The container noun is also removable (Fig. 2-I).
* **Seeds.** Candidate *i* uses seed `42 + i`.
* **CFG.** Guidance scale 7.5.
* **Low-density distance.** `sum(1 − SSIM)` over the three ResNet-18 scales,
  with a 7×7 Gaussian window and σ = 1.5.  The paper writes `d < d_max` but
  does not publish `d_max`; we keep the smallest distances in each prompt
  group up to a fixed quota.
* **Quota.** `1000 // G`, with the remainder assigned to the first groups.
  The per-group threshold is the last retained distance; `thresholds.json`
  records it.

## Stage III: RNA and residual masks

* **Reference.** One whole normal image minimising the sum of full-map L2
  distances over the three scales (not a mean-square distance).  The paper's
  Eq. 5 writes the closest-reference argmin per scale; we instead choose one
  whole normal image shared by all three scales, so the reference is
  consistent across scales.  This is our interpretation of an ambiguous
  equation, not a literal per-scale argmin.
* **Memory-bank scope.** The RNA memory bank uses all `train/good` images of
  the category, not only the one-third MCL subset.  The paper says "given N
  reference normal images" but does not state whether N is the MCL subset or
  the full training set; the MCL subset restriction applies only to Stage-I
  embedding learning.
* **Normal reconstruction.** 50-step `DDIMInverseScheduler` + 50-step reverse
  DDIM, guidance 1.  The reconstruction PSNR is stored per pair.
* **Attention.** Average over all 16×16 cross-attention layers and heads,
  Gaussian-smoothed with a 3×3 kernel.
* **Token alignment.** Both generated and reference images are queried at
  t = 0 with the same full normal prompt, so component channels share token
  identity.  The paper does not specify whether removed tokens are re-queried.
* **Residual.** Sum of absolute attention differences over component channels,
  upsampled bilinearly to 512×512.  Per-image min–max normalisation to [0,1]
  is used for the soft training target.  The raw float residual path is
  recorded in `pairs_1000.json`; the raw arrays belong to the omitted
  generated bank, not this archive.  Eq. 10 does not specify this mapping.
* **SSIM ambiguity.** Eq. 6 calls the sum of SSIM a distance; we use
  `sum(1 − SSIM)` as a dissimilarity and keep the smallest values.

## Stage IV: CSDA detector

* **Optimizer/schedule.** Adam, LR 10⁻⁴, 100 epochs, batch 16.  The paper does
  not publish the Stage-III optimizer or schedule.
* **Architecture.** Eq. 11 raw channel-mean differences; gated features before
  projection; U-Net widths 64/128/256; depthwise-separable downsampling;
  bilinear + 1×1 upsampling.
* **Loss.** Smooth L1 + 5 × Focal + Dice; focal γ = 2, no class weighting,
  continuous soft targets.
* **Validation.** Approximately 10% of synthetic reference groups (with all
  pairs sharing those references) plus the official `validation/good`; the
  actual fraction varies with the number of groups.  Checkpoint selected by
  lowest validation loss.
* **Test use.** The test set is read once after training; no test label is
  used for training or model selection.
* **Image score.** Maximum of the sigmoid anomaly map.  The paper does not
  specify image-level pooling.

## Determinism

The code seeds Python, NumPy, PyTorch and CUDA (seed 42).  The MCL DataLoader
uses a seeded generator.  Exact bitwise reproduction is not guaranteed across
GPU models, CUDA/cuDNN versions, or different numbers of GPUs; the reported
run used four RTX A6000s.

## Pixel-level metrics

The paper's Table II also reports pixel-level AU-sPRO (mean 71.8).  This
archive reports image-level AUROC only.  AU-sPRO needs the MVTec test images,
pixel ground truth and the official `defects_config.json` saturation
thresholds, none of which is redistributed; the included detector checkpoint
is not enough by itself.  No substitute pixel metric is claimed.  See
[`METRICS.md`](METRICS.md).

## Provenance boundary

`protocol.json` records the run-time source hashes.  The released Python files
were cleaned after the run (comments/docstrings removed, portable paths and a
local AUROC helper), so their hashes differ from the run-time values.  The
run-time values are retained, not rewritten.  The pair-manifest hash matches
the released `pairs_1000.json`.  The full recorded/released hash table is
[`PROVENANCE.md`](PROVENANCE.md).
