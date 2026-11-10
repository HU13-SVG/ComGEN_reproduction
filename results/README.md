# Results

| Path | Contents |
|---|---|
| `stage1_embeddings/<category>/training_manifest.json` | MCL subset, epochs, steps/epoch, trainable parameters, prompts, augmentation |
| `stage1_embeddings/<category>/embeddings_final.pt` | Learned component embeddings |
| `stage2_pairs/<category>/pairs_1000.json` | 1,000 selected pairs: prompt group, seed, generated image, latent, mask, raw residual, nearest normal, reconstruction PSNR |
| `stage2_pairs/<category>/thresholds.json` | Per-prompt low-density distance thresholds |
| `stage2_pairs/<category>/protocol.json` | Generation provenance, prompts, assumptions, SD-1.5 hashes |
| `stage2_pairs/<category>/pairs_preview.jpg` | Fixed-position normal/generated/mask preview |
| `stage5_full/<category>/result.json` | Final image, logical and structural AUROC; best epoch; test count |
| `stage5_full/<category>/test_scores.csv` | Per-image test scores, type and label |
| `stage5_full/<category>/history.json` | 100-epoch training/validation history |
| `stage5_full/<category>/split.json` | Exact train/validation indices and references |
| `stage5_full/<category>/protocol.json` | Detector args, assumptions, source/model hashes, no-test-selection flag |
| `stage5_full/<category>/best.pth` | Detector checkpoint selected by validation loss |
| `summary/reproduction_summary.csv` | Final paper-vs-reproduction table |
| `summary/reproduction_summary.json` | Same table with exact values |
| `summary/audit_report.json` | Independent AUROC/split/pair-manifest audit |
| `summary/audit_summary.csv` | Audit table |
| `logs/` | MCL, generation and detector logs (plus scheduler state) |
| `provenance/hash_cache/` | Raw file-hash cache used during generation |
| `SHA256SUMS` | SHA-256 of the released result evidence and checkpoints |

`python tools/verify_results.py` recomputes the image-level AUROC table and
the logical/structural diagnostic splits from the released `test_scores.csv`
files using only the Python standard library.  `python summarize.py`
performs the heavier split/path/self-reference/pair-hash audit and writes a
new `audit_<timestamp>/` directory under `results/stage5_full/`.

The released `pairs_1000.json` records pair metadata, not image bytes.  The
2,000-candidate pool, 1,000 generated images/masks, raw residual arrays and
normal cache are the omitted ≈9.5 GB bank.  `pairs_preview.jpg` is a
fixed-index sample, not a quality-selected subset.  Paths inside the run JSON
files are the absolute paths from the execution machine; they are kept
verbatim as run records, not portable file references.

The detector checkpoints (`best.pth`) are included, but retraining requires
the omitted pair bank.  Pixel-level AU-sPRO is not reported; see
`../docs/METRICS.md`.  Run-time vs released source hashes are tabulated in
`../docs/PROVENANCE.md`.
