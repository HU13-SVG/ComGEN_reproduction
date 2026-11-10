# Model Assets

These assets are **not** included; download or create them locally.

## Stable Diffusion 1.5

Use the diffusers-format snapshot of
[`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
and place it at:

```text
models/stable-diffusion-v1-5/
├── tokenizer/
├── text_encoder/
├── vae/
├── unet/
├── scheduler/
├── model_index.json
└── ...
```

The code loads `tokenizer`, `text_encoder`, `vae`, `unet` and `scheduler` with
`local_files_only=True`.  Set `SD15_PATH` to another path if needed.

The exact SD-1.5 component hashes used for the released generation are
recorded in `results/stage2_pairs/*/protocol.json` under
`provenance.base_model.files`.

## ResNet-18 ImageNet weights

The detector and the RNA/low-density filter use a ResNet-18 state dict.  Create
it with torchvision:

```bash
python -c "import torch, torchvision
torch.save(torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
).state_dict(), 'models/resnet18_imagenet1k_v1.pth')"
```

Place the file at `models/resnet18_imagenet1k_v1.pth`, or set
`RESNET18_WEIGHTS` / pass `--resnet18`.

The SHA-256 used in the released protocol is recorded in
`results/stage5_full/*/protocol.json` under `resnet_sha256`.  It is the hash
of the run-time serialized state-dict file.  Re-saving the same torchvision
weights with a different PyTorch version can change the serialized bytes
while the tensors are identical; compare tensors if the file hash differs.
