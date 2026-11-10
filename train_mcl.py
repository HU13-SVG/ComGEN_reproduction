#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from diffusers import DDIMScheduler
ROOT = Path(_file_).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from config import get_category_config
from utils import load_sd_components, add_placeholder_tokens, compose_prompt, wrap_with_template

class NounEmbeddings(nn.Module):

    def _init_(self, base, ids):
        super()._init_()
        self.base = base.requires_grad_(False)
        self.register_buffer('ids', torch.tensor(ids, device=base.weight.device))
        self.vectors = nn.Parameter(base.weight[ids].detach().clone())

    def forward(self, input_ids):
        x = self.base(input_ids)
        for (i, token_id) in enumerate(self.ids):
            x = torch.where((input_ids == token_id)[..., None], self.vectors[i], x)
        return x

class NormalSubset(Dataset):

    def _init_(self, paths):
        self.paths = paths
        self.augment = T.Compose([T.Resize((544, 544)), T.Pad(32, padding_mode='reflect'), T.RandomRotation(5, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(544), T.RandomCrop(512), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def _len_(self):
        return len(self.paths)

    def _getitem_(self, index):
        with Image.open(self.paths[index]) as im:
            return self.augment(im.convert('RGB'))

@torch.no_grad()
def preview(tok, te, vae, unet, model, full_prompt, path):
    scheduler = DDIMScheduler.from_pretrained(model, subfolder='scheduler', local_files_only=True)
    scheduler.set_timesteps(50, device='cuda')
    ids = tok(['', full_prompt], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.cuda()
    cond = te(ids)[0].to(unet.dtype)
    z = torch.randn((1, 4, 64, 64), device='cuda', dtype=unet.dtype, generator=torch.Generator(device='cuda').manual_seed(421337))
    for t in scheduler.timesteps:
        (eu, ec) = unet(torch.cat([z, z]), t, encoder_hidden_states=cond).sample.chunk(2)
        z = scheduler.step(eu + 7.5 * (ec - eu), t, z).prev_sample
    image = (vae.decode(z.float() / vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
    to_pil_image(image[0].float().cpu()).save(path)

def main(a):
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output) / a.category
    out.mkdir(parents=True, exist_ok=True)
    all_paths = sorted((Path(a.data_root) / a.category / 'train/good').glob('*.png'))
    selected = sorted(random.Random(a.seed).sample(all_paths, max(1, round(len(all_paths) / 3))))
    cfg = copy.deepcopy(get_category_config(a.category))
    if a.category == 'breakfast_box':
        for (c, word) in zip(cfg['components'], ['box', 'fruit', 'cereal', 'topping']):
            c['init'] = word
    components = cfg['components']
    (tok, te, vae, unet, noise_scheduler) = load_sd_components(a.model, 'cuda', torch.bfloat16)
    ids = add_placeholder_tokens(tok, te, components)
    te.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    vae.to(dtype=torch.float32)
    embedding = NounEmbeddings(te.get_input_embeddings(), ids)
    te.set_input_embeddings(embedding)
    parameters = [p for p in te.parameters() if p.requires_grad]
    assert sum((p.numel() for p in parameters)) == len(components) * 768
    optimizer = torch.optim.Adam(parameters, lr=1e-05, betas=(0.9, 0.999), eps=1e-08)
    dataset = NormalSubset(selected)
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, generator=generator)
    full = compose_prompt(cfg)
    manifest = dict(category=a.category, seed=a.seed, all_train_count=len(all_paths), selected_count=len(selected), selected_files=[str(p) for p in selected], epochs=a.epochs, batch_size=16, lr=1e-05, steps_per_epoch=len(loader), resolution=512, base_prompt=full, components=components, optimizer='Adam, no weight decay', precision='fp32 text/VAE, bf16 frozen UNet', trainable_parameters=len(components) * 768, source_sha256=hashlib.sha256(Path(_file_).read_bytes()).hexdigest(), augmentation='resize 544 square; reflect pad 32; rotate +/-5 degrees; center crop 544; random crop 512', assumptions=['SD1.5 base', 'noun initialization words', 'augmentation magnitudes', 'six CLIP-style templates', 'floor full batches per epoch; drop_last=True'], test_used=False)
    (out / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))
    (epoch0, steps) = (0, 0)
    if a.resume and (out / 'resume.pt').exists():
        state = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        embedding.vectors.data.copy_(state['vectors'].cuda())
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'])
        generator.set_state(state['loader_rng'])
        (epoch0, steps) = (state['epoch'], state['steps'])
    initial = embedding.vectors.detach().clone()
    started = time.time()
    for ep in range(epoch0 + 1, a.epochs + 1):
        total = 0.0
        for x in loader:
            x = x.cuda(non_blocking=True)
            with torch.no_grad():
                z = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                z = z.to(unet.dtype)
            noise = torch.randn_like(z)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(x),), device='cuda')
            noisy = noise_scheduler.add_noise(z, noise, t)
            tids = tok([wrap_with_template(full) for _ in range(len(x))], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.cuda()
            prediction = unet(noisy, t, encoder_hidden_states=te(tids)[0].to(unet.dtype)).sample
            target = noise if noise_scheduler.config.prediction_type == 'epsilon' else noise_scheduler.get_velocity(z, noise, t)
            loss = (prediction.float() - target.float()).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(embedding.vectors.grad).all():
                raise RuntimeError('Non-finite MCL loss/gradient; refusing to save invalid embeddings')
            grad_norm = embedding.vectors.grad.norm().item()
            optimizer.step()
            total += loss.item()
            steps += 1
        if ep == epoch0 + 1 or ep % 10 == 0:
            delta = (embedding.vectors - initial).norm().item()
            print(f'{a.category} epoch={ep}/{a.epochs} updates={steps} loss={total / len(loader):.6f} grad={grad_norm:.6f} delta={delta:.5f} elapsed_h={(time.time() - started) / 3600:.3f}', flush=True)
        if ep in (1, 100, 1000, 3000, a.epochs):
            preview(tok, te, vae, unet, a.model, full, out / f'normal_epoch_{ep:04d}.png')
        if ep % 100 == 0 or ep == a.epochs:
            learned = {c['token']: v.detach().cpu().clone() for (c, v) in zip(components, embedding.vectors)}
            torch.save(dict(embeddings=learned, components=components), out / f'embeddings_epoch_{ep:04d}.pt')
            state = dict(vectors=embedding.vectors.detach().cpu(), optimizer=optimizer.state_dict(), epoch=ep, steps=steps, random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), loader_rng=generator.get_state())
            torch.save(state, out / 'resume.tmp.pt')
            (out / 'resume.tmp.pt').replace(out / 'resume.pt')
    torch.save(dict(embeddings={c['token']: v.detach().cpu().clone() for (c, v) in zip(components, embedding.vectors)}, components=components), out / 'embeddings_final.pt')
if _name_ == '_main_':
    p = argparse.ArgumentParser()
    p.add_argument('--category', required=True)
    p.add_argument('--output', default=str(ROOT / 'stage1_embeddings'))
    p.add_argument('--data_root', default=os.environ.get('MVTEC_LOCO_ROOT', str(ROOT / 'data' / 'mvtec_loco')))
    p.add_argument('--model', default=os.environ.get('SD15_PATH', str(ROOT / 'models' / 'stable-diffusion-v1-5')))
    p.add_argument('--epochs', type=int, default=6100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    main(p.parse_args())
