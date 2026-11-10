# Data

## MVTec LOCO AD

Download the dataset from
<https://www.mvtec.com/company/research/datasets/mvtec-loco> and arrange it as:

```text
data/mvtec_loco/
├── breakfast_box/
│   ├── train/good/*.png
│   ├── validation/good/*.png
│   └── test/{good,logical_anomalies,structural_anomalies}/*.png
├── juice_bottle/
├── pushpins/
├── screw_bag/
└── splicing_connectors/
```

Normal-image counts used by the paper protocol:

| Category | Normals | One-third MCL subset |
|---|---:|---:|
| Breakfast box | 351 | 117 |
| Juice bottle | 335 | 112 |
| Pushpins | 372 | 124 |
| Screw bag | 360 | 120 |
| Splicing connectors | 360 | 120 |
| **Total** | **1,778** | **593** |

MCL and generation read only `train/good`.  The detector uses `train/good`
(all normals), `validation/good` (official validation) and the synthetic
pairs.  The test split is evaluated once after training.

## Released manifests

The `results/stage2_pairs_v6/*/pairs_1000.json` files contain the absolute
paths from the execution machine (images, latents, masks, references).  They
are run records, not portable paths.  The selected pairs, prompt groups,
seeds, references, mask paths and reconstruction PSNRs are all recorded there.
The `results/stage2_pairs_v6/*/pairs_preview.jpg` images are fixed-position
normal / generated / mask montages.

MVTec LOCO images are not redistributed and remain subject to the MVTec
licence.
