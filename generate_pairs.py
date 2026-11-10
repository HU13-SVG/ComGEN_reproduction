#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from diffusers import DDIMInverseScheduler
ROOT = Path(_file_).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from config import get_category_config
from utils import compose_prompt
from comgen_pipeline import ComGENPipeline, _CaptureProcessor, _gaussian3
from low_density_filter import channelwise_ssim
from train_eval_csda import Encoder, extract, nearest_indices, WEIGHTS, DATA
from provenance import capture_sources, package_versions, base_model_manifest, file_sha256

class BatchAttention:

    def _init_(self):
        self.res = 16
        self.capturing = True
        self.sum = None
        self.count = 0

    def add(self, maps):
        self.sum = maps.float() if self.sum is None else self.sum + maps.float()
        self.count += 1

    def averaged(self):
        if not self.count:
            raise RuntimeError('No 16x16 cross-attention recorded')
        return self.sum / self.count

class Pipeline(ComGENPipeline):

    @torch.no_grad()
    def common_attention(self, latent, full_prompt):
        cond = self._enc(full_prompt).expand(len(latent), -1, -1)
        original = dict(self.unet.attn_processors)
        store = BatchAttention()
        processors = {name: _CaptureProcessor(store) if name.endswith('attn2.processor') else proc for (name, proc) in original.items()}
        self.unet.set_attn_processor(processors)
        try:
            self.unet(latent, 0, encoder_hidden_states=cond)
        finally:
            self.unet.set_attn_processor(dict(original))
        maps = store.averaged()
        positions = [self._tok_index(full_prompt, c['token']) for c in self.components]
        maps = maps[:, :, positions].permute(0, 2, 1).reshape(len(latent), len(positions), 16, 16)
        flat = maps.reshape(-1, 1, 16, 16)
        return _gaussian3(flat).reshape_as(maps)

    @torch.no_grad()
    def sample_batch(self, prompts, seeds):
        cond = self._enc(prompts)
        uncond = self._enc([''] * len(prompts))
        embeddings = torch.cat([uncond, cond])
        z = torch.cat([torch.randn((1, 4, 64, 64), generator=torch.Generator(device=self.device).manual_seed(seed), device=self.device, dtype=self.dtype) for seed in seeds])
        for t in self.scheduler.timesteps:
            (eu, ec) = self.unet(torch.cat([z, z]), t, encoder_hidden_states=embeddings).sample.chunk(2)
            z = self.scheduler.step(eu + self.guidance * (ec - eu), t, z).prev_sample
        return z

    @torch.no_grad()
    def reconstruct(self, image, full_prompt):
        z = self.encode(image * 2 - 1)
        inverse = DDIMInverseScheduler.from_config(self.scheduler.config)
        inverse.set_timesteps(self.steps, device=self.device)
        cond = self._enc(full_prompt)
        for t in inverse.timesteps:
            eps = self.unet(z, t, encoder_hidden_states=cond).sample
            z = inverse.step(eps, t, z).prev_sample
        for t in self.scheduler.timesteps:
            eps = self.unet(z, t, encoder_hidden_states=cond).sample
            z = self.scheduler.step(eps, t, z).prev_sample
        return z

def write_json(path, obj):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)

def main(a):
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output) / a.category
    for folder in ('images', 'latents', 'masks', 'raw_residuals', 'normal_cache'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    cfg = get_category_config(a.category)
    full = compose_prompt(cfg)
    specs = [('unmodified', full)] + [('remove_' + c['name'], full.replace(c['token'], '').strip()) for c in cfg['components']]
    provenance_paths = [Path(_file_), ROOT / 'train_mcl.py', ROOT / 'train_eval_csda.py', ROOT / 'provenance.py'] + sorted((ROOT / 'vendor').glob('*.py'))
    provenance = dict(source_sha256=capture_sources(provenance_paths), package_versions=package_versions(), base_model=base_model_manifest(a.model), resnet_sha256=file_sha256(a.resnet18), ddim_scheduler_config=json.loads((Path(a.model) / 'scheduler/scheduler_config.json').read_text()))
    protocol = dict(category=a.category, seed=a.seed, embeddings=str(Path(a.embeddings).resolve()), embeddings_sha256=hashlib.sha256(Path(a.embeddings).read_bytes()).hexdigest(), provenance=provenance, published=dict(ddim_steps=50, attention_resolution=16, gaussian_kernel=3, reference_scales=['64x64x64', '128x32x32', '256x16x16'], keep=a.keep), assumptions=dict(cfg=a.guidance, candidates=a.candidates, base_model='SD1.5', ls='sum(1-SSIM); smallest distances within fixed per-prompt quotas', d_max='per-prompt order statistic to keep exactly 1000; numeric threshold unpublished', prompt_policy='unmodified plus one deletion of every noun; surrounding words retained', inversion='DDIMInverseScheduler, guidance 1, 50 steps; reverse guidance 1', attention='common full prompt, no added noise, t=0 after reconstruction/generation', soft_targets='per-image min/max Eq.10 residual; raw float residual also preserved', ssim='7x7 Gaussian, sigma 1.5, per-channel reference range'), paper_ambiguities=['Eq.6 calls sum SSIM a distance although SSIM is a similarity', 'whether A0 is queried again with the full prompt is not specified', 'mapping raw attention residuals to segmentation target range is not specified'], prompts=specs, test_used=False)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Output protocol differs; choose a fresh output directory')
    write_json(protocol_path, protocol)
    pipe = Pipeline(a.model, a.embeddings, device='cuda', num_inference_steps=50, guidance_scale=a.guidance, resolution=512)
    candidate_path = out / 'candidates.json'
    rows = json.loads(candidate_path.read_text()) if candidate_path.exists() else []
    for start in range(len(rows), a.candidates, a.batch_size):
        indices = list(range(start, min(start + a.batch_size, a.candidates)))
        prompts = [specs[i % len(specs)][1] for i in indices]
        seeds = [a.seed + i for i in indices]
        zs = pipe.sample_batch(prompts, seeds)
        imgs = pipe.decode(zs).cpu()
        for (i, z, img, prompt, seed) in zip(indices, zs, imgs, prompts, seeds):
            ip = out / 'images' / f'{i:05d}.png'
            zp = out / 'latents' / f'{i:05d}.pt'
            to_pil_image(img).save(ip)
            torch.save(z[None].cpu(), zp)
            rows.append(dict(image=str(ip.resolve()), latent=str(zp.resolve()), prompt=prompt, seed=seed, spec=specs[i % len(specs)][0]))
        write_json(candidate_path, rows)
        if start == 0 or len(rows) % 100 == 0:
            print(f'{a.category}: candidates {len(rows)}/{a.candidates}', flush=True)
    normals = sorted((Path(a.data_root) / a.category / 'train/good').glob('*.png'))
    encoder = Encoder(a.resnet18).cuda()
    memory = extract(encoder, normals)
    selected_path = out / 'selected.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        scored = []
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            features = extract(encoder, [r['image'] for r in batch], 16)
            nearest = nearest_indices(features, memory)
            distances = sum((1 - channelwise_ssim(f.float(), bank[nearest].float()) for (f, bank) in zip(features, memory)))
            if not torch.isfinite(distances).all():
                raise RuntimeError('Non-finite SSIM distance')
            for (row, index, distance) in zip(batch, nearest.tolist(), distances.tolist()):
                scored.append(dict(row, closest_normal=str(normals[index].resolve()), distance=distance))
        selected = []
        thresholds = {}
        for (i, (spec, _)) in enumerate(specs):
            group = sorted([r for r in scored if r['spec'] == spec], key=lambda r: (r['distance'], r['seed']))
            quota = a.keep // len(specs) + int(i < a.keep % len(specs))
            if len(group) < quota:
                raise ValueError('Insufficient candidates for a prompt quota')
            selected.extend(group[:quota])
            thresholds[spec] = group[quota - 1]['distance'] if quota else None
        write_json(out / 'all_candidate_scores.json', scored)
        write_json(out / 'thresholds.json', thresholds)
        write_json(selected_path, selected)
    del encoder, memory
    torch.cuda.empty_cache()
    completed = []
    for (i, row) in enumerate(selected):
        cache_key = hashlib.sha256(row['closest_normal'].encode()).hexdigest()
        cache = out / 'normal_cache' / f'{cache_key}.pt'
        if cache.exists():
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            normal_attention = saved['attention'].cuda()
        else:
            with Image.open(row['closest_normal']) as im:
                x = to_tensor(im.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR))[None].cuda()
            z = pipe.reconstruct(x, full)
            normal_attention = pipe.common_attention(z, full)
            reconstruction = pipe.decode(z)
            mse = (reconstruction - x).square().mean().item()
            saved = dict(attention=normal_attention.cpu(), reconstruction_mse=mse, reconstruction_psnr=float(-10 * np.log10(max(mse, 1e-12))))
            torch.save(saved, cache)
        mask_path = out / 'masks' / f'{i:05d}.png'
        raw_path = out / 'raw_residuals' / f'{i:05d}.npy'
        if not (mask_path.exists() and raw_path.exists()):
            z = torch.load(row['latent'], map_location='cuda', weights_only=True).to(pipe.dtype)
            anomaly_attention = pipe.common_attention(z, full)
            residual = (normal_attention - anomaly_attention).abs().sum(1, keepdim=True)
            residual = F.interpolate(residual, (512, 512), mode='bilinear', align_corners=False)[0, 0]
            if not torch.isfinite(residual).all():
                raise RuntimeError('Non-finite residual map')
            np.save(raw_path, residual.cpu().numpy())
            target = (residual - residual.min()) / (residual.max() - residual.min() + 1e-06)
            to_pil_image(target.cpu()).save(mask_path)
        completed.append(dict(row, mask=str(mask_path.resolve()), raw_residual=str(raw_path.resolve()), reconstruction_psnr=saved['reconstruction_psnr']))
        if i == 0 or (i + 1) % 25 == 0:
            print(f'{a.category}: masks {i + 1}/{len(selected)}', flush=True)
    write_json(out / 'pairs_1000.json', completed)
    sheet = Image.new('RGB', (256 * 3, 280 * min(12, len(completed))), 'white')
    draw = ImageDraw.Draw(sheet)
    indices = np.linspace(0, len(completed) - 1, min(12, len(completed)), dtype=int)
    for (y, index) in enumerate(indices):
        row = completed[index]
        draw.text((5, y * 280 + 3), row['spec'] + ' | reference / generated / mask', fill='black')
        for (x, key) in enumerate(('closest_normal', 'image', 'mask')):
            with Image.open(row[key]) as im:
                sheet.paste(im.convert('RGB').resize((256, 256)), (x * 256, y * 280 + 24))
    sheet.save(out / 'pairs_preview.jpg')
    print(f'{a.category}: completed {len(completed)} paired images', flush=True)
if _name_ == '_main_':
    p = argparse.ArgumentParser()
    p.add_argument('--category', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--output', default=str(ROOT / 'stage2_pairs'))
    p.add_argument('--model', default=os.environ.get('SD15_PATH', str(ROOT / 'models' / 'stable-diffusion-v1-5')))
    p.add_argument('--data_root', default=DATA)
    p.add_argument('--resnet18', default=WEIGHTS)
    p.add_argument('--candidates', type=int, default=2000)
    p.add_argument('--keep', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--guidance', type=float, default=7.5)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
