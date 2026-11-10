#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
ROOT = Path(__file__).resolve().parent
CATEGORIES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
PAPER = dict(zip(CATEGORIES, [87.2, 99.1, 92.9, 79.8, 95.2]))
DATA = os.environ.get('MVTEC_LOCO_ROOT', str(ROOT / 'data' / 'mvtec_loco'))
WEIGHTS = os.environ.get('RESNET18_WEIGHTS', str(ROOT / 'models' / 'resnet18_imagenet1k_v1.pth'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def roc_auc_score(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError('labels and scores must be 1-D and equal length')
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        raise ValueError('AUROC requires both classes')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = np.flatnonzero(scores == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))

class Encoder(nn.Module):

    def __init__(self, weights):
        super().__init__()
        net = models.resnet18(weights=None)
        net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleList([net.layer1, net.layer2, net.layer3])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        x = self.stem((x - self.mean) / self.std)
        result = []
        for block in self.blocks:
            x = block(x)
            result.append(x)
        return result

def load_rgb(path):
    with Image.open(path) as im:
        return transforms.functional.to_tensor(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))

@torch.no_grad()
def extract(encoder, paths, batch=32):
    parts = [[], [], []]
    for start in range(0, len(paths), batch):
        x = torch.stack([load_rgb(p) for p in paths[start:start + batch]]).cuda()
        for (dest, feature) in zip(parts, encoder(x)):
            dest.append(feature.half())
    return [torch.cat(p) for p in parts]

@torch.no_grad()
def nearest_indices(features, memory, excluded=None, batch=16):
    bank_flat = [b.flatten(1).float() for b in memory]
    bank_norms = [b.square().sum(1)[None] for b in bank_flat]
    result = []
    for start in range(0, len(features[0]), batch):
        total = None
        for (q, bank, norm) in zip(features, bank_flat, bank_norms):
            q = q[start:start + batch].flatten(1).float()
            d2 = q.square().sum(1)[:, None] + norm - 2 * (q @ bank.T)
            d = d2.clamp_min(0).sqrt()
            total = d if total is None else total + d
        if excluded is not None:
            ex = excluded[start:start + batch]
            valid = ex >= 0
            total[torch.arange(len(ex), device=total.device)[valid], ex[valid]] = float('inf')
        result.append(total.argmin(1))
    return torch.cat(result)

class SepConv(nn.Sequential):

    def __init__(self, ci, co, stride=1):
        super().__init__(nn.Conv2d(ci, ci, 3, stride, 1, groups=ci, bias=False), nn.BatchNorm2d(ci), nn.ReLU(inplace=True), nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class CSDA(nn.Module):

    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList([SepConv(128, 64), SepConv(256, 128), SepConv(512, 256)])
        (self.down12, self.down23) = (SepConv(64, 128, 2), SepConv(128, 256, 2))
        self.mid = SepConv(256, 256)
        (self.up3, self.up2) = (nn.Conv2d(256, 128, 1), nn.Conv2d(128, 64, 1))
        (self.dec2, self.dec1) = (SepConv(256, 128), SepConv(128, 64))
        self.out = nn.Sequential(SepConv(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, features, differences):
        masks = [d.mean(1, keepdim=True) for d in differences]
        m3 = masks[2]
        m2 = masks[1] * F.interpolate(m3, masks[1].shape[-2:], mode='bilinear', align_corners=False)
        m1 = masks[0] * F.interpolate(m2, masks[0].shape[-2:], mode='bilinear', align_corners=False)
        cs = [project(torch.cat([f, d], 1) * m) for (project, f, d, m) in zip(self.project, features, differences, [m1, m2, m3])]
        (c1, c2, c3) = cs
        e2 = c2 + self.down12(c1)
        e3 = c3 + self.down23(e2)
        z = self.mid(e3)
        z = self.dec2(torch.cat([e2, F.interpolate(self.up3(z), e2.shape[-2:], mode='bilinear', align_corners=False)], 1))
        z = self.dec1(torch.cat([c1, F.interpolate(self.up2(z), c1.shape[-2:], mode='bilinear', align_corners=False)], 1))
        return self.out(F.interpolate(z, size=(256, 256), mode='bilinear', align_corners=False))

def objective(logits, targets):
    prob = logits.sigmoid()
    positive = -targets * (1 - prob).square() * F.logsigmoid(logits)
    negative = -(1 - targets) * prob.square() * F.logsigmoid(-logits)
    focal = (positive + negative).mean()
    dice = 1 - (2 * (prob * targets).sum((1, 2, 3)) + 1) / ((prob + targets).sum((1, 2, 3)) + 1)
    return F.smooth_l1_loss(prob, targets) + 5 * focal + dice.mean()

def select_features(features, indices, reference_ids, memory):
    fs = [f[indices].float() for f in features]
    rs = [m[reference_ids[indices]].float() for m in memory]
    return (fs, [(f - r).abs() for (f, r) in zip(fs, rs)])

def epoch(net, features, masks, reference_ids, memory, indices, batch, optimizer=None):
    net.train(optimizer is not None)
    if optimizer is not None:
        indices = indices[torch.randperm(len(indices), device=indices.device)]
    total = 0.0
    scores = []
    for start in range(0, len(indices), batch):
        ix = indices[start:start + batch]
        (fs, ds) = select_features(features, ix, reference_ids, memory)
        with torch.set_grad_enabled(optimizer is not None):
            logits = net(fs, ds)
            loss = objective(logits, masks[ix].float())
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite detector loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * len(ix)
        scores.extend(logits.detach().sigmoid().flatten(1).amax(1).cpu().tolist())
    return (total / len(indices), scores)

def main(a):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output) / a.category
    out.mkdir(parents=True, exist_ok=True)
    pair_path = Path(a.masks) / a.category / 'pairs_1000.json'
    pairs = json.loads(pair_path.read_text())
    if len(pairs) != 1000 and (not a.smoke):
        raise ValueError('The paper run requires exactly 1000 synthetic image/mask pairs')
    if a.smoke:
        pairs = pairs[:32]
    train_normals = sorted((Path(a.data_root) / a.category / 'train/good').glob('*.png'))
    val_normals = sorted((Path(a.data_root) / a.category / 'validation/good').glob('*.png'))
    if not val_normals or len(train_normals) < 2:
        raise ValueError('Need train normals and independent official validation normals')
    if a.smoke:
        (train_normals, val_normals) = (train_normals[:16], val_normals[:8])
    paths = [Path(r['image']) for r in pairs] + train_normals + val_normals
    for path in paths:
        if 'test' in path.parts:
            raise ValueError('Test data in training/validation inputs')
    assumptions = dict(base_diffusion='SD1.5 (generation provenance in pairs manifest)', optimizer='Adam', lr=a.lr, detector_epochs=a.epochs, batch=a.batch_size, image_score='max sigmoid anomaly map', resize='bilinear 256x256 for every path', normal_training='all train/good, own reference excluded', validation='10% synthetic, grouped by source reference, plus official validation/good', focal='gamma 2, alpha unweighted, continuous soft target', csda='Eq.11 raw channel-mean differences; gated C before projection', decoder='UNet widths 64,128,256; depthwise down; bilinear+1x1 up', reference='one whole normal image, minimum sum of full-map L2 norms')
    manifest = dict(args=vars(a), assumptions=assumptions, pair_manifest_sha256=digest(pair_path), source_sha256=digest(__file__), resnet_sha256=digest(a.resnet18), test_used_for_selection=False, paper_target=PAPER[a.category], train_normals=[str(p) for p in train_normals], val_normals=[str(p) for p in val_normals])
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    encoder = Encoder(a.resnet18).cuda()
    memory = extract(encoder, train_normals)
    features = extract(encoder, paths)
    (ns, nt, nv) = (len(pairs), len(train_normals), len(val_normals))
    excluded = torch.full((len(paths),), -1, device='cuda', dtype=torch.long)
    excluded[ns:ns + nt] = torch.arange(nt, device='cuda')
    references = nearest_indices(features, memory, excluded)
    if torch.any(references[ns:ns + nt] == torch.arange(nt, device='cuda')):
        raise AssertionError('Normal self-reference was not excluded')
    masks = []
    for row in pairs:
        with Image.open(row['mask']) as im:
            mask = transforms.functional.to_tensor(im.convert('L').resize((256, 256), Image.Resampling.BILINEAR))
        if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError('Segmentation targets must be finite and in [0,1]')
        masks.append(mask)
    masks = torch.cat([torch.stack(masks), torch.zeros(nt + nv, 1, 256, 256)]).half().cuda()
    groups = sorted({r['closest_normal'] for r in pairs})
    random.Random(a.seed).shuffle(groups)
    val_groups = set(groups[:max(1, round(len(groups) * 0.1))])
    syn_train = [i for (i, r) in enumerate(pairs) if r['closest_normal'] not in val_groups]
    syn_val = [i for (i, r) in enumerate(pairs) if r['closest_normal'] in val_groups]
    if not syn_train or not syn_val:
        raise ValueError('Need at least two independent synthetic reference groups')
    train_ix = torch.tensor(syn_train + list(range(ns, ns + nt)), device='cuda')
    val_ix = torch.tensor(syn_val + list(range(ns + nt, ns + nt + nv)), device='cuda')
    (out / 'split.json').write_text(json.dumps(dict(train_indices=train_ix.tolist(), validation_indices=val_ix.tolist(), paths=[str(p) for p in paths], references=references.tolist()), indent=2))
    print(f'{a.category}: cached {len(paths)} examples; train={len(train_ix)} val={len(val_ix)}; self-reference excluded', flush=True)
    net = CSDA().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    history = []
    best = float('inf')
    start = time.time()
    for ep in range(1, a.epochs + 1):
        (train_loss, _) = epoch(net, features, masks, references, memory, train_ix, a.batch_size, optimizer)
        (val_loss, val_scores) = epoch(net, features, masks, references, memory, val_ix, a.batch_size)
        proxy_auc = roc_auc_score([1] * len(syn_val) + [0] * nv, val_scores) * 100
        row = dict(epoch=ep, train_loss=train_loss, validation_loss=val_loss, synthetic_validation_auroc=proxy_auc)
        history.append(row)
        if val_loss < best:
            best = val_loss
            torch.save(dict(model=net.state_dict(), epoch=ep, validation_loss=best, protocol=manifest), out / 'best.pth')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'{a.category}: epoch={ep}/{a.epochs} train={train_loss:.5f} val={val_loss:.5f} synthetic_val_AUC={proxy_auc:.2f}', flush=True)
    checkpoint = torch.load(out / 'best.pth', map_location='cuda', weights_only=False)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    result = dict(category=a.category, best_epoch=checkpoint['epoch'], best_validation_loss=best, seconds=time.time() - start, test_evaluated=False, protocol=str(out / 'protocol.json'))
    if a.evaluate:
        del features, masks
        items = [(p, int(folder.name != 'good'), folder.name) for folder in sorted((Path(a.data_root) / a.category / 'test').iterdir()) if folder.is_dir() for p in sorted(folder.glob('*.png'))]
        scores = []
        with torch.no_grad():
            for s in range(0, len(items), a.batch_size):
                chunk = items[s:s + a.batch_size]
                fs = extract(encoder, [p for (p, _, _) in chunk], a.batch_size)
                refs = nearest_indices(fs, memory)
                fs = [f.float() for f in fs]
                ds = [(f - m[refs].float()).abs() for (f, m) in zip(fs, memory)]
                scores.extend(net(fs, ds).sigmoid().flatten(1).amax(1).cpu().tolist())
        labels = [label for (_, label, _) in items]
        result.update(image_auroc=roc_auc_score(labels, scores) * 100, test_evaluated=True, test_count=len(items), paper_image_auroc=PAPER[a.category])
        for typ in ['logical_anomalies', 'structural_anomalies']:
            subset = [i for (i, (_, _, kind)) in enumerate(items) if kind in ('good', typ)]
            result[typ + '_auroc'] = roc_auc_score([labels[i] for i in subset], [scores[i] for i in subset]) * 100
        with (out / 'test_scores.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['path', 'type', 'label', 'score'])
            writer.writerows(((str(p), typ, label, score) for ((p, label, typ), score) in zip(items, scores)))
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=CATEGORIES, required=True)
    parser.add_argument('--data_root', default=DATA)
    parser.add_argument('--resnet18', default=WEIGHTS)
    parser.add_argument('--masks', default=str(ROOT / 'stage2_pairs_v6'))
    parser.add_argument('--output', default=str(ROOT / 'stage5_full_v6'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
