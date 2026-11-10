# Provenance and Reproduction Boundary

## Run environment

From `results/stage2_pairs/*/protocol.json` and
`results/stage5_full/*/protocol.json`:

* Python 3.8 environment; PyTorch 2.4.1+cu121, torchvision 0.19.1,
  diffusers 0.36.0, transformers 4.46.3, numpy 1.24.4, Pillow 10.4.0.
* Four NVIDIA RTX A6000 GPUs; the scheduler records one category per GPU.
* Stable Diffusion 1.5 in diffusers format; ResNet-18 ImageNet weights.
* The run records are dated from 2026-10-03 onward in
  `results/logs/state.json`.

## Source hashes: recorded at run time vs released files

`protocol.json` stores the SHA-256 of the source files at run time.  The
released archive is a cleaned copy: comments/docstrings were removed and
portable paths/local helpers were adjusted after the run.  The table below
lists both values so a reviewer can see the difference.  The run-time hashes
are retained unaltered.

| Released file | Run-time SHA-256 in `protocol.json` | Released SHA-256 | Status |
|---|---|---|---|
| `vendor/comgen_pipeline.py` | `e1e3e1bad998eec85cc77e4b1836ec0578362cc08afd2a976ea418ea699e8155` | `a5b805493bf6ed1f795d8f6046440e5ad9245d22b3f14f0f9c6388215ff042a9` | cleaned copy |
| `vendor/config.py` | `e4f15050eaf63f04da6b76d09b384c804f726c3a1acc7d62b8608dd5bdb11473` | `6c4849a018c2810d65e9d9f7f00ff65776980da0c8770002360624b566dc8d7a` | cleaned copy |
| `vendor/dataset.py` | `188177e12e57ff428467a7b86e683512bed3c9c2105f50d888bf0d03e9ee0f98` | `d143530f017099813e94e132a9166d4e57e28635097af52efed48ee6866ec367` | cleaned copy |
| `vendor/low_density_filter.py` | `52b79b6e79c03d7b2cc6694e3f1039370159f6ee9a66388b4675c5b5dd4ae81d` | `f2f707e576779d53632dec3ebcfa27b0a2e18b8edc286561b7967a7444c37ba9` | cleaned copy |
| `vendor/utils.py` | `3e49e02a0e8316c24bd8feea79905fafbc7e0c522d7f4e2936fc2b7fd0ed9875` | `cf1a3299d4581a82d069f8dd04c4216d0bf2a8fb8b0c90669f0613079d8341ad` | cleaned copy |
| `provenance.py` | `680f47c3bd868771315dba8e000d2cc41962de097940304e784aaf1815991a63` | `46506f5e42fd5829ea531d5d732205fdd70d1f6faa263039412b2402890d01d9` | cleaned copy |
| `generate_pairs.py` | `ce4c0281fd9d1002bfd3cf96fd7fafb07f00d3a05c0ad8c07be667048fbe9cbf` | `289197e31233df10eeeaa8c12df3f31c43538a6c26b02b0a80ae9c3d82c706d1` | cleaned copy |
| `train_mcl.py` | `90a177007097c03504f2d4275c4ba02dff1bf9bd845d657df8404519b0c7f9c7` | `1355bdb4a7e68e0282abd24768f83cc3672051ed58cff6534500d78bad0fd836` | cleaned copy |
| `train_eval_csda.py` | `1d9745dfb17364c3093dd2f5b80ca0e14481e0d69903cdaa532587e9ae02bbe6` | `a47f68d7888efaee0ae176a1b5265b868c7a3186ffa369a4ec4c796c1a0e5df4` | cleaned copy |

The pair-manifest hash does match exactly:
`d2ade97a4601eac1e8e37f616367dccfefbb7e175db86e5794f59ac243b3b0af` for
`results/stage2_pairs/breakfast_box/pairs_1000.json` and the recorded value
in `results/stage5_full/breakfast_box/protocol.json`.

## Evidence integrity

`results/SHA256SUMS` covers the per-image test scores, `result.json`,
`history.json`, `split.json`, `protocol.json`, `best.pth` and the summary
audit files.  The pair manifests are covered by their own protocol-recorded
hash (above).

## Redistributed and omitted assets

| Item | Included? | Notes |
|---|---|---|
| MCL code and learned embeddings (`embeddings_final.pt`) | Yes | Stage I evidence |
| 1,000 selected pair manifests, thresholds, previews | Yes | Images/latents/masks themselves are omitted |
| 2,000-candidate pool, 1,000 generated images/masks, raw residuals, normal cache | No | ≈9.5 GB generated bank |
| Detector checkpoints (`best.pth`) | Yes | Selected by validation loss |
| Stable Diffusion 1.5 and ResNet-18 weights | No | Download/create locally, hashes recorded in protocol |
| MVTec LOCO images | No | MVTec licence |

The manifests contain the execution machine's absolute paths.  They are kept
verbatim as run records; they are not portable file references.  The detector
cannot be retrained from this archive alone without regenerating the pair
bank; `scripts/run_detector_4gpu.sh` is intended for that case and requires
the full pipeline output.

## What a reviewer can verify without the generated bank

* `python tools/verify_results.py` recomputes the image-level AUROC table from
  `test_scores.csv` using only the standard library.
* `python summarize.py` re-runs the split/path/self-reference/pair-hash
  audit (CPU-only) and writes a new audit directory; it does not overwrite the
  released report.
* `results/stage5_full/*/best.pth` is the selected checkpoint; the released
  per-image scores are its test evaluation.
* The 100-epoch detector histories and train/validation splits are included.

## What requires the generated bank or a retraining run

* Retraining the detector from scratch (needs the 1,000 generated pairs).
* Regenerating Stage II/III (needs SD-1.5 and the MCL embeddings).
* Any pixel-level metric; see `docs/METRICS.md`.
