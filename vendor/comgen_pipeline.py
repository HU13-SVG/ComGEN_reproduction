#!/usr/bin/env python3
import os
import json
import math
from typing import Optional, List, Dict
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from config import get_category_config
from dataset import list_normal_images
from utils import load_sd_components, load_learned_embeddings, compose_prompt, enumerate_anomaly_specs, seed_everything

class _AttnStore:

    def _init_(self, res=16):
        self.res = res
        self.capturing = False
        self.sum = None
        self.count = 0

    def reset(self):
        self.sum = None
        self.count = 0

    def add(self, probs):
        p = probs[0]
        self.sum = p if self.sum is None else self.sum + p
        self.count += 1

    def averaged(self):
        return self.sum / max(self.count, 1)

class _CaptureProcessor:

    def _init_(self, store):
        self.store = store

    def _call_(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        if getattr(attn, 'spatial_norm', None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            (b, c, h, w) = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if getattr(attn, 'group_norm', None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        is_cross = encoder_hidden_states is not None
        q = attn.to_q(hidden_states)
        ehs = encoder_hidden_states if is_cross else hidden_states
        if getattr(attn, 'norm_cross', None) and is_cross:
            ehs = attn.norm_encoder_hidden_states(ehs)
        k = attn.to_k(ehs)
        v = attn.to_v(ehs)
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)
        probs = attn.get_attention_scores(q, k, attention_mask)
        if is_cross and self.store.capturing:
            qn = probs.shape[1]
            if int(round(qn ** 0.5)) == self.store.res and int(round(qn ** 0.5)) ** 2 == qn:
                p = probs.reshape(-1, attn.heads, qn, probs.shape[-1]).mean(1)
                self.store.add(p.detach())
        out = torch.bmm(probs, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(1, 2).view(b, c, h, w)
        if getattr(attn, 'residual_connection', False):
            out = out + residual
        return out / attn.rescale_output_factor

def _gaussian3(x):
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype) / 16.0
    k = k.view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)

class ComGENPipeline:

    def _init_(self, model_id, embeddings_path, device='cuda', num_inference_steps=50, guidance_scale=7.5, resolution=512):
        from diffusers import DDIMScheduler
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        (self.device, self.dtype, self.res) = (device, dtype, resolution)
        (self.steps, self.guidance) = (num_inference_steps, guidance_scale)
        (tok, te, vae, unet, _) = load_sd_components(model_id, device, weight_dtype=dtype)
        self.components = load_learned_embeddings(tok, te, embeddings_path)
        te.to(device)
        vae.to(device=device, dtype=torch.float32)
        (self.tokenizer, self.text_encoder, self.vae, self.unet) = (tok, te, vae, unet)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler', local_files_only=True)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.alphas = self.scheduler.alphas_cumprod.to(device)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.store = _AttnStore(res=16)
        self._orig = None

    @torch.no_grad()
    def _enc(self, p):
        ids = self.tokenizer(p, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt').input_ids.to(self.device)
        return self.text_encoder(ids)[0].to(self.dtype)

    def _cfg(self, prompt):
        return torch.cat([self._enc(''), self._enc(prompt)], dim=0)

    def _tok_index(self, prompt, token_str):
        ids = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True).input_ids
        tid = self.tokenizer.convert_tokens_to_ids(token_str)
        return ids.index(tid) if tid in ids else 1

    @torch.no_grad()
    def encode(self, img_m11):
        z = self.vae.encode(img_m11.to(self.device, torch.float32)).latent_dist.mean
        return (z * self.vae.config.scaling_factor).to(self.dtype)

    @torch.no_grad()
    def decode(self, z):
        z = z.to(torch.float32) / self.vae.config.scaling_factor
        return (self.vae.decode(z).sample.float() / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def sample(self, prompt, seed=0):
        g = torch.Generator(device=self.device).manual_seed(seed)
        emb = self._cfg(prompt)
        lat_shape = (1, self.unet.config.in_channels, self.res // 8, self.res // 8)
        z = torch.randn(lat_shape, device=self.device, dtype=self.dtype, generator=g)
        z = z * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            eps = self.unet(torch.cat([z, z]), t, encoder_hidden_states=emb).sample
            (eu, ec) = eps.chunk(2)
            eps = eu + self.guidance * (ec - eu)
            z = self.scheduler.step(eps, t, z).prev_sample
        return z

    @torch.no_grad()
    def ddim_invert(self, z0, prompt):
        cond = self._enc(prompt)
        ts = list(self.scheduler.timesteps.flip(0))
        z = z0
        for i in range(len(ts)):
            t = ts[i]
            t_next = ts[i + 1] if i + 1 < len(ts) else ts[i]
            eps = self.unet(z, t, encoder_hidden_states=cond).sample
            (a_t, a_next) = (self.alphas[t], self.alphas[t_next])
            x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return z

    def _install(self):
        self._orig = self.unet.attn_processors
        self.unet.set_attn_processor({n: _CaptureProcessor(self.store) for n in self._orig})

    def _restore(self):
        if self._orig is not None:
            self.unet.set_attn_processor(self._orig)
            self._orig = None

    @torch.no_grad()
    def token_attention(self, z0, full_prompt, token_indices):
        cond = self._enc(full_prompt)
        t0 = self.scheduler.timesteps[-1]
        noise = torch.randn(z0.shape, device=self.device, dtype=self.dtype)
        zt = self.scheduler.add_noise(z0, noise, t0)
        self.store.reset()
        self.store.capturing = True
        self._install()
        try:
            _ = self.unet(zt, t0, encoder_hidden_states=cond).sample
        finally:
            self.store.capturing = False
            self._restore()
        avg = self.store.averaged().float()
        r = self.store.res
        return {ti: _gaussian3(avg[:, ti].reshape(1, 1, r, r)) for ti in token_indices}

    @torch.no_grad()
    def residual_mask(self, z0_anom, normal_img01, full_prompt):
        z0_norm = self.encode(normal_img01.unsqueeze(0) * 2 - 1)
        idxs = [self._tok_index(full_prompt, c['token']) for c in self.components]
        A_gen = self.token_attention(z0_anom, full_prompt, idxs)
        A_norm = self.token_attention(z0_norm, full_prompt, idxs)
        res = None
        for ti in idxs:
            d = (A_norm[ti] - A_gen[ti]).abs()
            res = d if res is None else res + d
        res = F.interpolate(res, size=(self.res, self.res), mode='bilinear', align_corners=False)[0, 0]
        res = res - res.min()
        return (res / (res.max() + 1e-06)).cpu()

def run_sample(category, embeddings_path, data_root, model_id, out_dir=None, n_per_anomaly=200, num_inference_steps=50, guidance_scale=7.5, resolution=512, device='cuda', include_missing=True, include_extra=True, only_specs=None, num_shards=1, shard_id=0, manifest_name='manifest.json', use_low_density_filter=True, resnet_path=None, keep_quantile=0.6, ref_resolution=256, max_ref_images=200, save_masks=True, mask_threshold=0.5, seed=0, **_ignore):
    seed_everything(seed)
    cfg = get_category_config(category)
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: '{embeddings_path}'. Train first.")
    if out_dir is None:
        out_dir = os.path.join('outputs', 'generated', category)
    os.makedirs(out_dir, exist_ok=True)
    pipe = ComGENPipeline(model_id, embeddings_path, device=device, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, resolution=resolution)
    files = list_normal_images(data_root, category, split='train')[:max_ref_images]
    if not files:
        raise FileNotFoundError(f"No normal images for '{category}'.")
    tf_hi = T.Compose([T.Resize(resolution), T.CenterCrop(resolution), T.ToTensor()])
    tf_lo = T.Compose([T.Resize(ref_resolution), T.CenterCrop(ref_resolution), T.ToTensor()])
    ref_hi = [tf_hi(Image.open(f).convert('RGB')) for f in files]
    ref_lo = torch.stack([tf_lo(Image.open(f).convert('RGB')) for f in files])
    from low_density_filter import LowDensityFilter
    ld = LowDensityFilter(ref_lo, device=device, resnet_weights=resnet_path)
    full_prompt = compose_prompt(cfg)
    to_pil = T.ToPILImage()
    specs = enumerate_anomaly_specs(cfg, include_missing, include_extra)
    if only_specs is not None:
        specs = [s for s in specs if s['name'] in set(only_specs)]
    if not specs:
        raise RuntimeError(f"No specs for '{category}' (only_specs={only_specs}).")
    base = n_per_anomaly // num_shards
    rem = n_per_anomaly % num_shards
    shard_count = base + (1 if shard_id < rem else 0)
    shard_start = shard_id * base + min(shard_id, rem)
    tag = '' if num_shards == 1 else f' [shard {shard_id + 1}/{num_shards}]'
    token_of = {c['name']: c['token'] for c in cfg['components']}
    manifest = []
    for spec in specs:
        adir = os.path.join(out_dir, spec['name'])
        os.makedirs(adir, exist_ok=True)
        mdir = os.path.join(out_dir, spec['name'] + '_mask')
        bdir = os.path.join(out_dir, spec['name'] + '_mask_bin')
        if save_masks:
            os.makedirs(mdir, exist_ok=True)
            os.makedirs(bdir, exist_ok=True)
        prompt_mod = compose_prompt(cfg, removed=spec['removed'], duplicated=spec['duplicated'])
        print(f"[{category}/{spec['name']}]{tag} PM prompt: {prompt_mod!r} (target {shard_count})")
        if shard_count == 0:
            continue
        if use_low_density_filter and keep_quantile < 0.999:
            budget = min(math.ceil(shard_count / max(keep_quantile, 0.05)), shard_count * 8)
        else:
            budget = shard_count
        pool = []
        pbar = tqdm(total=budget, desc=f"{category}:{spec['name']}{tag}")
        for j in range(budget):
            z0 = pipe.sample(prompt_mod, seed=seed + 9173 * shard_id + j)
            img = pipe.decode(z0)[0].cpu()
            if use_low_density_filter:
                lo = F.interpolate(img.unsqueeze(0), size=(ref_resolution, ref_resolution), mode='bilinear', align_corners=False)
                (dissim, nearest) = ld.score(lo)
                d = float(dissim[0])
                nidx = int(nearest[0])
            else:
                (d, nidx) = (0.0, 0)
            pool.append((d, z0.cpu(), img, nidx))
            if len(pool) > budget:
                pool.sort(key=lambda x: x[0])
                pool = pool[:budget]
            pbar.update(1)
        pbar.close()
        pool.sort(key=lambda x: x[0])
        chosen = pool[:shard_count]
        for (k, (d, z0_cpu, img, nidx)) in enumerate(tqdm(chosen, desc=f"{category}:{spec['name']}{tag} masks")):
            gidx = shard_start + k
            img_path = os.path.join(adir, f'{gidx:04d}.png')
            to_pil(img.clamp(0, 1)).save(img_path)
            if save_masks:
                z0 = z0_cpu.to(pipe.device, pipe.dtype)
                normal_img = ref_hi[nidx]
                ymask = pipe.residual_mask(z0, normal_img, full_prompt)
                to_pil(ymask.clamp(0, 1)).save(os.path.join(mdir, f'{gidx:04d}.png'))
                binm = (ymask >= mask_threshold).float()
                to_pil(binm).save(os.path.join(bdir, f'{gidx:04d}.png'))
            manifest.append({'anomaly': spec['name'], 'path': img_path, 'closest_normal': files[nidx], 'dissim': round(d, 4), 'prompt': prompt_mod, 'mode': 'sample'})
    mpath = os.path.join(out_dir, manifest_name)
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[{category}]{tag} generated {len(manifest)} image-mask pairs -> {mpath}')
    return out_dir
