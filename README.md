# ComGEN* — Paper-Guided Reproduction on MVTec LOCO

Paper-guided re-implementation of ComGEN (Tong et al., ICRA 2025):
multi-component textual inversion (MCL), prompt modification + low-density
sampling (PM/LS), reference-neighbour association + residual mapping (RNA/RM),
and the CSDA detector.

Result: **69.40** mean image-level AUROC on all 1,568 MVTec LOCO test images,
compared with the paper's printed **91.2**.  This archive reports image-level
AUROC only; pixel AU-sPRO, IS/IC-LPIPS, the ablation, the classification
comparison, diesel-engine and MVTecAD experiments are not reproduced.

## Result

| Category | Paper (Table II) | Reproduced | Test n | Best epoch |
|---|---:|---:|---:|---:|
| Breakfast box | 87.2 | 77.00 | 275 | 100 |
| Juice bottle | 99.1 | 87.88 | 330 | 89 |
| Pushpins | 92.9 | 57.76 | 310 | 10 |
| Screw bag | 79.8 | 65.25 | 341 | 92 |
| Splicing connectors | 95.2 | 59.11 | 312 | 83 |
| **Mean** | **91.2** | **69.40** | **1,568** | — |

The arithmetic mean of the paper's five rounded values is 90.84; the printed
mean is 91.2.  Both are reported, neither is adjusted.  Checkpoints are
selected by validation loss on a reference-grouped synthetic holdout plus
official `validation/good`; no test label is used for selection.

## Layout

```text
train_mcl_v6.py          Stage I
generate_pairs_v6.py     Stage II + III
train_eval_csda_v6.py    Stage IV detector
run_v6_full_pipeline.py  persistent 4-GPU scheduler
summarize_v6.py          split/path/pair-hash audit
v6_vendor/               cleaned SD-1.5/prompt/attention utilities
results/                 per-category results, scores, checkpoints, logs
docs/                    PROTOCOL, ASSUMPTIONS, METRICS, PROVENANCE
scripts/                 full pipeline and detector scripts
tools/verify_results.py  stdlib-only AUROC recomputation
```

## Install and assets

```bash
conda create -n comgen python=3.8 -y
conda activate comgen
pip install -r requirements.txt
```

Download MVTec LOCO as `data/mvtec_loco/<category>/{train,validation,test}/...`.
Stable Diffusion 1.5 and ResNet-18 ImageNet weights are **not** included; see
`models/README.md`.  The 2,000-candidate pool and 1,000 generated pairs per
category (≈9.5 GB) are not included either; `results/stage2_pairs_v6/*/pairs_1000.json`
records the selected pairs, and `pairs_preview.jpg` shows fixed-index samples.

## Run

```bash
cp environment.sh.example environment.sh
# edit paths
source environment.sh
bash scripts/run_full_pipeline.sh
```

The scheduler runs one category per GPU and resumes completed stages.  The
detector can be retrained only after the pair bank has been regenerated:

```bash
bash scripts/run_detector_4gpu.sh
```

## Verify

```bash
python tools/verify_results.py
python summarize_v6.py
```

`verify_results.py` uses only the standard library and recomputes the image,
logical and structural AUROCs from the released per-image scores.
`summarize_v6.py` re-checks split/path overlap, self-reference exclusion and
the pair-manifest hash.

## Scope and deviations

* The paper does not publish the four non-breakfast prompts, `d_max`, the CFG
  scale, the attention re-query rule, the detector schedule or the image-level
  pooling rule; our choices are in `docs/ASSUMPTIONS.md`.
* We generate 2,000 candidates and retain 1,000 per category; the paper fixes
  only the final 1,000 count.
* The source hashes in `protocol.json` are run-time values.  The released
  Python files are cleaned copies (comments/docstrings removed); the full
  recorded/released table is `docs/PROVENANCE.md`.
* Pixel-level AU-sPRO is not reported; MVTec pixel ground truth and
  `defects_config.json` are not redistributed.

Details: [`docs/PROTOCOL.md`](docs/PROTOCOL.md),
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md),
[`docs/METRICS.md`](docs/METRICS.md),
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Licence

Research use.  Stable Diffusion 1.5, ResNet-18, MVTec LOCO and the
textual-inversion components retain their upstream licences.
