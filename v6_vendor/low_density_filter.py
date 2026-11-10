#!/usr/bin/env python3
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def _find_weight_file(path: str) -> str:
    import os
    import glob
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        patterns = ['*.safetensors', '*.pth', '*.pt', '*.bin', '*.ckpt']
        hits = []
        for pat in patterns:
            hits += glob.glob(os.path.join(path, pat))
            hits += glob.glob(os.path.join(path, '**', pat), recursive=True)
        hits = sorted(set(hits), key=lambda p: (p.count(os.sep), p))
        if not hits:
            raise FileNotFoundError(f"No weight file (.safetensors/.pth/.pt/.bin) found under '{path}'.")
        return hits[0]
    raise FileNotFoundError(f"ResNet-18 weights path does not exist: '{path}'.")

def _load_state_dict(weight_file: str) -> dict:
    if weight_file.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(weight_file)
    else:
        sd = torch.load(weight_file, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'model', 'model_state_dict', 'net'):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    cleaned = {}
    for (k, v) in sd.items():
        nk = k
        for pref in ('module.', 'model.', 'backbone.', 'encoder.'):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    return cleaned

def load_local_resnet18(weights_path: str):
    net = torchvision.models.resnet18(weights=None)
    sd = _load_state_dict(_find_weight_file(weights_path))
    (missing, unexpected) = net.load_state_dict(sd, strict=False)
    relevant_missing = [m for m in missing if not m.startswith('fc.')]
    if relevant_missing:
        print(f"[resnet18] WARNING: missing keys (non-fc): {relevant_missing[:8]}{(' ...' if len(relevant_missing) > 8 else '')}")
    return net

class ResNet18MultiScale(nn.Module):

    def __init__(self, device: str='cuda', weights_path: str=None):
        super().__init__()
        if weights_path:
            net = load_local_resnet18(weights_path)
        else:
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval().to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = (x - self.mean) / self.std
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]

def _gaussian_window(window_size: int, sigma: float, channels: int, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    win = g[:, None] @ g[None, :]
    return win.expand(channels, 1, window_size, window_size).contiguous()

def channelwise_ssim(a: torch.Tensor, b: torch.Tensor, window_size: int=7, sigma: float=1.5) -> torch.Tensor:
    c = a.shape[1]
    win = _gaussian_window(window_size, sigma, c, a.device)
    pad = window_size // 2

    def filt(t):
        return F.conv2d(t, win, padding=pad, groups=c)
    (mu_a, mu_b) = (filt(a), filt(b))
    (mu_a2, mu_b2, mu_ab) = (mu_a * mu_a, mu_b * mu_b, mu_a * mu_b)
    sig_a = filt(a * a) - mu_a2
    sig_b = filt(b * b) - mu_b2
    sig_ab = filt(a * b) - mu_ab
    L = (b.amax(dim=(2, 3), keepdim=True) - b.amin(dim=(2, 3), keepdim=True)).clamp_min(1e-06)
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    ssim_map = (2 * mu_ab + c1) * (2 * sig_ab + c2) / ((mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2))
    return ssim_map.mean(dim=(1, 2, 3))

class LowDensityFilter:

    def __init__(self, reference_imgs: torch.Tensor, device: str='cuda', resnet_weights: str=None):
        self.device = device
        self.extractor = ResNet18MultiScale(device=device, weights_path=resnet_weights)
        with torch.no_grad():
            self.ref_feats = self._batched_features(reference_imgs.to(device))

    def _batched_features(self, imgs, bs: int=32):
        feats = [[], [], []]
        for i in range(0, imgs.shape[0], bs):
            fs = self.extractor(imgs[i:i + bs])
            for j in range(3):
                feats[j].append(fs[j].cpu())
        return [torch.cat(f, dim=0) for f in feats]

    @torch.no_grad()
    def score(self, cand_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cand_imgs = cand_imgs.to(self.device)
        cf = self.extractor(cand_imgs)
        B = cand_imgs.shape[0]
        N = self.ref_feats[0].shape[0]
        dist = torch.zeros(B, N, device=self.device)
        for j in range(3):
            cfj = cf[j].reshape(B, -1)
            rfj = self.ref_feats[j].reshape(N, -1).to(self.device)
            dist += torch.cdist(cfj, rfj, p=2)
        nearest = dist.argmin(dim=1)
        dissim = torch.zeros(B, device=self.device)
        for j in range(3):
            ref_sel = self.ref_feats[j][nearest.cpu()].to(self.device)
            ssim = channelwise_ssim(cf[j], ref_sel)
            dissim += 1.0 - ssim
        dissim = torch.nan_to_num(dissim, nan=1000000.0, posinf=1000000.0, neginf=1000000.0)
        return (dissim.cpu(), nearest.cpu())

    @torch.no_grad()
    def keep_mask(self, cand_imgs: torch.Tensor, min_dissim: float=0.15, max_dissim: float=1.5) -> torch.Tensor:
        (dissim, _) = self.score(cand_imgs)
        return (dissim >= min_dissim) & (dissim <= max_dissim)
