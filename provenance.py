#!/usr/bin/env python3
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
ROOT = Path(_file_).resolve().parent
DEFAULT_CACHE = ROOT / 'hash_cache'

def _stat(path):
    s = path.stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}

def file_sha256(path, cache_dir=DEFAULT_CACHE):
    path = Path(path).resolve(strict=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (hashlib.sha256(os.fsencode(path)).hexdigest() + '.json')
    before = _stat(path)
    try:
        cached = json.loads(cache_path.read_text())
        if cached.get('path') == str(path) and cached.get('stat') == before:
            digest = cached['sha256']
            if len(digest) == 64 and all((c in '0123456789abcdef' for c in digest)):
                return digest
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if before != _stat(path):
        raise RuntimeError(f'File changed while computing provenance: {path}')
    record = {'path': str(path), 'stat': before, 'sha256': digest.hexdigest()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=cache_dir, prefix=cache_path.stem + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record['sha256']

def capture_sources(paths):
    result = {}
    for value in paths:
        path = Path(value)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda : stream.read(1024 * 1024), b''):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result

def package_versions():
    packages = {'torch': 'torch', 'torchvision': 'torchvision', 'diffusers': 'diffusers', 'transformers': 'transformers', 'numpy': 'numpy', 'PIL': 'Pillow'}
    versions = {}
    for (name, distribution) in packages.items():
        try:
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions

def base_model_manifest(model_path, cache_dir=DEFAULT_CACHE):
    root = Path(model_path).resolve(strict=True)
    selected = set()
    components = {}
    for component in ('text_encoder', 'unet', 'vae'):
        weights = sorted((p for p in (root / component).rglob('*') if p.is_file() and p.suffix in {'.safetensors', '.bin', '.pt', '.pth'}))
        if not weights:
            raise FileNotFoundError(f'No model weights found for {component}: {root}')
        selected.update(weights)
        components[component] = [str(p.relative_to(root)) for p in weights]
    selected.update((p for p in root.rglob('*.json') if p.is_file()))
    tokenizer = [p for p in (root / 'tokenizer').rglob('*') if p.is_file()]
    if not tokenizer:
        raise FileNotFoundError(f"No tokenizer assets found: {root / 'tokenizer'}")
    selected.update(tokenizer)
    files = {}
    for path in sorted(selected):
        before = _stat(path)
        digest = file_sha256(path, cache_dir)
        if before != _stat(path):
            raise RuntimeError(f'File changed while collecting manifest: {path}')
        files[str(path.relative_to(root))] = dict(before, sha256=digest)
    return {'model_path': str(root), 'algorithm': 'sha256', 'components': components, 'files': files}
