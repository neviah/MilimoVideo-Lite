from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from huggingface_hub import hf_hub_url
from safetensors import safe_open

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    key: str
    repo_id: str
    filename: str
    out_rel_path: str
    sha256: Optional[str] = None
    quant: Optional[str] = None
    required_for_modes: tuple[str, ...] = ("low", "cpu")


DEFAULT_MANIFEST: List[ModelSpec] = [
    ModelSpec(
        key="ltx2_gguf_q4_m",
        repo_id=os.environ.get("MILIMO_LTX2_GGUF_Q4_REPO", "unsloth/LTX-2-GGUF"),
        filename=os.environ.get("MILIMO_LTX2_GGUF_Q4_FILE", "ltx-2-19b-dev-Q4_K_M.gguf"),
        out_rel_path="ltx2/ltx-2-19b-dev-Q4_K_M.gguf",
        sha256=os.environ.get("MILIMO_LTX2_GGUF_Q4_SHA256") or None,
        quant="Q4_M",
    ),
    ModelSpec(
        key="ltx2_gguf_q6_k",
        repo_id=os.environ.get("MILIMO_LTX2_GGUF_Q6_REPO", "unsloth/LTX-2-GGUF"),
        filename=os.environ.get("MILIMO_LTX2_GGUF_Q6_FILE", "ltx-2-19b-dev-Q6_K.gguf"),
        out_rel_path="ltx2/ltx-2-19b-dev-Q6_K.gguf",
        sha256=os.environ.get("MILIMO_LTX2_GGUF_Q6_SHA256") or None,
        quant="Q6_K",
    ),
    ModelSpec(
        key="flux2_klein_q4",
        repo_id=os.environ.get("MILIMO_FLUX2_Q4_REPO", "unsloth/FLUX.2-klein-9B-GGUF"),
        filename=os.environ.get("MILIMO_FLUX2_Q4_FILE", "flux-2-klein-9b-Q4_K_M.gguf"),
        out_rel_path="flux2/flux-2-klein-9b-Q4_K_M.gguf",
        sha256=os.environ.get("MILIMO_FLUX2_Q4_SHA256") or None,
        quant="4bit",
    ),
    ModelSpec(
        key="flux2_klein_q8",
        repo_id=os.environ.get("MILIMO_FLUX2_Q8_REPO", "unsloth/FLUX.2-klein-9B-GGUF"),
        filename=os.environ.get("MILIMO_FLUX2_Q8_FILE", "flux-2-klein-9b-Q8_0.gguf"),
        out_rel_path="flux2/flux-2-klein-9b-Q8_0.gguf",
        sha256=os.environ.get("MILIMO_FLUX2_Q8_SHA256") or None,
        quant="8bit",
    ),
    ModelSpec(
        key="flux2_ae_native",
        repo_id=os.environ.get("MILIMO_FLUX2_AE_REPO", "dci05049/flux2-klein-9b"),
        filename=os.environ.get("MILIMO_FLUX2_AE_FILE", "flux2-vae.safetensors"),
        out_rel_path="flux2/ae.safetensors",
        sha256=os.environ.get("MILIMO_FLUX2_AE_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="sam3_quant",
        repo_id=os.environ.get("MILIMO_SAM3_REPO", "1038lab/sam3"),
        filename=os.environ.get("MILIMO_SAM3_FILE", "sam3.pt"),
        out_rel_path="sam3/sam3.pt",
        sha256=os.environ.get("MILIMO_SAM3_SHA256") or None,
        quant="8bit",
    ),
    ModelSpec(
        key="gemma3_config",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_CONFIG_FILE", "config.json"),
        out_rel_path="text_encoders/gemma3/config.json",
        sha256=os.environ.get("MILIMO_GEMMA3_CONFIG_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_model_index",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_INDEX_FILE", "model.safetensors.index.json"),
        out_rel_path="text_encoders/gemma3/model.safetensors.index.json",
        sha256=os.environ.get("MILIMO_GEMMA3_INDEX_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_model_shard_1",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_SHARD1_FILE", "model-00001-of-00002.safetensors"),
        out_rel_path="text_encoders/gemma3/model-00001-of-00002.safetensors",
        sha256=os.environ.get("MILIMO_GEMMA3_SHARD1_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_model_shard_2",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_SHARD2_FILE", "model-00002-of-00002.safetensors"),
        out_rel_path="text_encoders/gemma3/model-00002-of-00002.safetensors",
        sha256=os.environ.get("MILIMO_GEMMA3_SHARD2_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_tokenizer",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_TOKENIZER_FILE", "tokenizer.model"),
        out_rel_path="text_encoders/gemma3/tokenizer.model",
        sha256=os.environ.get("MILIMO_GEMMA3_TOKENIZER_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_tokenizer_json",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_TOKENIZER_JSON_FILE", "tokenizer.json"),
        out_rel_path="text_encoders/gemma3/tokenizer.json",
        sha256=os.environ.get("MILIMO_GEMMA3_TOKENIZER_JSON_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_tokenizer_config",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_TOKENIZER_CONFIG_FILE", "tokenizer_config.json"),
        out_rel_path="text_encoders/gemma3/tokenizer_config.json",
        sha256=os.environ.get("MILIMO_GEMMA3_TOKENIZER_CONFIG_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_special_tokens",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_SPECIAL_TOKENS_FILE", "special_tokens_map.json"),
        out_rel_path="text_encoders/gemma3/special_tokens_map.json",
        sha256=os.environ.get("MILIMO_GEMMA3_SPECIAL_TOKENS_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
    ModelSpec(
        key="gemma3_preprocessor",
        repo_id=os.environ.get("MILIMO_GEMMA3_REPO", "unsloth/gemma-3-4b-it"),
        filename=os.environ.get("MILIMO_GEMMA3_PREPROCESSOR_FILE", "preprocessor_config.json"),
        out_rel_path="text_encoders/gemma3/preprocessor_config.json",
        sha256=os.environ.get("MILIMO_GEMMA3_PREPROCESSOR_SHA256") or None,
        quant="bf16",
        required_for_modes=("high", "low", "cpu"),
    ),
]


def _token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h: Dict[str, str] = {"User-Agent": "MilimoVideo-Lite/1.0"}
    t = _token()
    if t:
        h["Authorization"] = f"Bearer {t}"
    if extra:
        h.update(extra)
    return h


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: Optional[str]) -> bool:
    if not expected:
        return True
    actual = _sha256(path)
    return actual.lower() == expected.lower()


def _is_flux2_ae_compatible(path: Path) -> bool:
    """Validate Flux2 Klein AE architecture to avoid loading incompatible VAE files."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            needed = ("encoder.quant_conv.weight", "encoder.conv_out.weight", "decoder.conv_in.weight")
            if not all(k in f.keys() for k in needed):
                return False
            enc_shape = tuple(f.get_tensor("encoder.conv_out.weight").shape)
            dec_shape = tuple(f.get_tensor("decoder.conv_in.weight").shape)
            return enc_shape[:2] == (64, 512) and dec_shape[:2] == (512, 32)
    except Exception:
        return False


def _remote_size(url: str) -> Optional[int]:
    try:
        r = requests.head(url, headers=_headers(), timeout=30, allow_redirects=True)
        if r.ok and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        return None
    return None


def _human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _download_with_resume(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = _headers({"Range": f"bytes={existing}-"} if existing else None)
    total_remote = _remote_size(url)

    with requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"Download failed for {url}: HTTP {r.status_code}")

        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0

        downloaded = existing
        last_log = time.time()

        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_log >= 2.0:
                    if total_remote:
                        pct = min(100.0, (downloaded / float(total_remote)) * 100.0)
                        logger.info("download progress %.1f%% (%s/%s)", pct, _human_mb(downloaded), _human_mb(total_remote))
                    else:
                        logger.info("downloaded %s", _human_mb(downloaded))
                    last_log = now

    tmp.replace(out_path)


def list_manifest() -> List[ModelSpec]:
    return list(DEFAULT_MANIFEST)


def _iter_needed(mode: str) -> Iterable[ModelSpec]:
    for spec in DEFAULT_MANIFEST:
        if mode in spec.required_for_modes or "low" in spec.required_for_modes:
            yield spec


def ensure_models(models_root: str, mode: str = "low") -> Dict[str, str]:
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)

    resolved: Dict[str, str] = {}
    manifest_json: Dict[str, Dict[str, str]] = {}

    for spec in _iter_needed(mode):
        out_path = root / spec.out_rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and _verify_checksum(out_path, spec.sha256):
            if spec.key == "flux2_ae_native" and not _is_flux2_ae_compatible(out_path):
                logger.warning("Incompatible Flux2 AE detected at %s. Re-downloading compatible checkpoint.", out_path)
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                part = out_path.with_suffix(out_path.suffix + ".part")
                try:
                    part.unlink(missing_ok=True)
                except Exception:
                    pass
                # Continue to download branch below.
            else:
                logger.info("Model present: %s", out_path)
                resolved[spec.key] = str(out_path)
                manifest_json[spec.key] = {
                    "path": str(out_path),
                    "repo_id": spec.repo_id,
                    "filename": spec.filename,
                    "quant": spec.quant or "",
                    "status": "present",
                }
                continue

        url = hf_hub_url(spec.repo_id, spec.filename)
        try:
            logger.info("Downloading %s (%s) from %s", spec.key, spec.quant or "", url)
            _download_with_resume(url, out_path)

            if not _verify_checksum(out_path, spec.sha256):
                raise RuntimeError(f"Checksum mismatch for {out_path}")

            logger.info("Model ready: %s", out_path)
            resolved[spec.key] = str(out_path)
            manifest_json[spec.key] = {
                "path": str(out_path),
                "repo_id": spec.repo_id,
                "filename": spec.filename,
                "quant": spec.quant or "",
                "status": "downloaded",
            }
        except Exception as exc:
            logger.warning("Model download failed for %s: %s", spec.key, exc)
            manifest_json[spec.key] = {
                "path": str(out_path),
                "repo_id": spec.repo_id,
                "filename": spec.filename,
                "quant": spec.quant or "",
                "status": "failed",
                "error": str(exc),
            }

    idx_path = root / "milimovideo_lite_manifest.json"
    idx_path.write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")
    logger.info("Wrote model manifest: %s", idx_path)

    return resolved


def select_quantized_model(resolved: Dict[str, str], family: str, prefer: str) -> Optional[str]:
    prefer = prefer.lower()
    keys = [k for k in resolved.keys() if k.startswith(family)]
    if not keys:
        return None

    # Prefer explicit quant target then fallback to first available.
    for key in keys:
        if prefer in key.lower():
            return resolved[key]
    return resolved[keys[0]]
