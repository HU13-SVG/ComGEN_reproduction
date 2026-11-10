#!/usr/bin/env python3
import os
import random
from typing import List, Dict, Sequence, Optional
import torch
from config import TEMPLATES

def load_sd_components(model_id: str, device: str, weight_dtype=torch.float32, clip_path: str=None, local_files_only: bool=True):
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    lfo = {'local_files_only': local_files_only}
    if clip_path:
        tokenizer = CLIPTokenizer.from_pretrained(clip_path, **lfo)
        text_encoder = CLIPTextModel.from_pretrained(clip_path, **lfo)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer', **lfo)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder', **lfo)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', **lfo)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet', **lfo)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler', **lfo)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device)
    return (tokenizer, text_encoder, vae, unet, noise_scheduler)

def add_placeholder_tokens(tokenizer, text_encoder, components: List[Dict]) -> List[int]:
    tokens = [c['token'] for c in components]
    num_added = tokenizer.add_tokens(tokens)
    if num_added != len(tokens):
        raise ValueError(f'Some placeholder tokens already exist in the tokenizer. Use a fresh tokenizer or unique tokens. Requested {len(tokens)}, added {num_added}.')
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    embeds = text_encoder.get_input_embeddings().weight.data
    for (tok_id, comp) in zip(token_ids, components):
        init_ids = tokenizer.encode(comp['init'], add_special_tokens=False)
        if len(init_ids) == 0:
            raise ValueError(f"Initialiser word '{comp['init']}' tokenised to nothing.")
        embeds[tok_id] = embeds[init_ids].mean(dim=0).clone()
    return token_ids

def build_no_update_mask(tokenizer, placeholder_token_ids: Sequence[int]) -> torch.Tensor:
    mask = torch.ones((len(tokenizer),), dtype=torch.bool)
    for tid in placeholder_token_ids:
        mask[tid] = False
    return mask

def save_learned_embeddings(path: str, tokenizer, text_encoder, placeholder_token_ids: Sequence[int], components: List[Dict]):
    embeds = text_encoder.get_input_embeddings().weight.data
    learned = {comp['token']: embeds[tid].detach().cpu().clone() for (comp, tid) in zip(components, placeholder_token_ids)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'embeddings': learned, 'components': components}, path)

def load_learned_embeddings(tokenizer, text_encoder, path: str) -> List[Dict]:
    state = torch.load(path, map_location='cpu', weights_only=False)
    learned: Dict[str, torch.Tensor] = state['embeddings']
    components: List[Dict] = state['components']
    tokens = list(learned.keys())
    num_added = tokenizer.add_tokens(tokens)
    if num_added != len(tokens):
        raise ValueError('Placeholder tokens already present in tokenizer at load time.')
    text_encoder.resize_token_embeddings(len(tokenizer))
    embeds = text_encoder.get_input_embeddings().weight.data
    for (tok, vec) in learned.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        embeds[tid] = vec.to(embeds.dtype)
    return components

def compose_prompt(category_cfg: dict, removed: Optional[Sequence[str]]=None, duplicated: Optional[Sequence[str]]=None) -> str:
    removed = set(removed or [])
    duplicated = set(duplicated or [])
    token_of = {c['name']: c['token'] for c in category_cfg['components']}
    pieces = []
    for (name, frag) in category_cfg['fragments']:
        if name in removed:
            continue
        tok = token_of[name]
        pieces.append(frag.format(t=tok))
        if name in duplicated:
            pieces.append(frag.format(t=tok))
    return ' '.join(pieces).strip()

def wrap_with_template(prompt_core: str, rng: Optional[random.Random]=None) -> str:
    tpl = (rng or random).choice(TEMPLATES)
    return tpl.format(prompt_core)

def enumerate_anomaly_specs(category_cfg: dict, include_missing: bool=True, include_extra: bool=True) -> List[Dict]:
    specs = []
    for c in category_cfg['components']:
        if not c.get('removable', False):
            continue
        if include_missing:
            specs.append({'name': f"missing_{c['name']}", 'removed': [c['name']], 'duplicated': []})
        if include_extra:
            specs.append({'name': f"extra_{c['name']}", 'removed': [], 'duplicated': [c['name']]})
    return specs

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
