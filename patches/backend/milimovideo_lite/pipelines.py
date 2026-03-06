from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .interfaces import (
    ExecutionPlan,
    ImagePipeline,
    SegmentationPipeline,
    TemporalWindow,
    TileSpec,
    VideoPipeline,
)
from .model_manager import select_quantized_model
from .vram import get_total_vram_gb

logger = logging.getLogger(__name__)


def _choose_chunk_size(vram_gb: float) -> int:
    if vram_gb >= 10.0:
        return 8
    if vram_gb >= 7.0:
        return 6
    return 4


def _choose_tile_grid(vram_gb: float) -> str:
    if vram_gb >= 8.0:
        return "2x2"
    return "3x3"


def _build_temporal_windows(total_frames: int, chunk_size: int, overlap: int = 1) -> List[TemporalWindow]:
    if total_frames <= 0:
        return []
    windows: List[TemporalWindow] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < total_frames:
        end = min(total_frames, start + chunk_size)
        windows.append(
            TemporalWindow(
                start=start,
                end=end,
                overlap_left=overlap if start > 0 else 0,
                overlap_right=overlap if end < total_frames else 0,
            )
        )
        if end >= total_frames:
            break
        start += step
    return windows


def _build_tiles(width: int, height: int, grid: str) -> List[TileSpec]:
    rows, cols = (2, 2) if grid == "2x2" else (3, 3)
    tiles: List[TileSpec] = []
    for r in range(rows):
        for c in range(cols):
            y0 = int((r * height) / rows)
            y1 = int(((r + 1) * height) / rows)
            x0 = int((c * width) / cols)
            x1 = int(((c + 1) * width) / cols)
            tiles.append(TileSpec(row=r, col=c, y0=y0, y1=y1, x0=x0, x1=x1))
    return tiles


def stitch_temporal_chunks(chunks: List[np.ndarray], overlap: int = 1) -> np.ndarray:
    """Blend-overlap stitch for list of [F,H,W,C] chunks."""
    if not chunks:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    if len(chunks) == 1:
        return chunks[0]

    out = chunks[0]
    for nxt in chunks[1:]:
        if overlap <= 0:
            out = np.concatenate([out, nxt], axis=0)
            continue

        ol = min(overlap, out.shape[0], nxt.shape[0])
        if ol <= 0:
            out = np.concatenate([out, nxt], axis=0)
            continue

        head = out[:-ol]
        tail = nxt[ol:]

        a = out[-ol:].astype(np.float32)
        b = nxt[:ol].astype(np.float32)
        weights = np.linspace(0.0, 1.0, ol, dtype=np.float32).reshape(ol, 1, 1, 1)
        blend = (a * (1.0 - weights) + b * weights).astype(out.dtype)

        out = np.concatenate([head, blend, tail], axis=0)
    return out


def stitch_latent_tiles(tiles: List[np.ndarray], width: int, height: int, grid: str) -> np.ndarray:
    rows, cols = (2, 2) if grid == "2x2" else (3, 3)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    norm = np.zeros((height, width, 1), dtype=np.float32)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = int((r * height) / rows)
            y1 = int(((r + 1) * height) / rows)
            x0 = int((c * width) / cols)
            x1 = int(((c + 1) * width) / cols)
            if idx >= len(tiles):
                break
            tile = tiles[idx].astype(np.float32)
            idx += 1
            tile = tile[: (y1 - y0), : (x1 - x0)]
            canvas[y0:y1, x0:x1] += tile
            norm[y0:y1, x0:x1] += 1.0

    norm[norm == 0] = 1.0
    out = (canvas / norm).clip(0, 255).astype(np.uint8)
    return out


@dataclass
class LowVRAMLTX2Backend:
    resolved_models: Dict[str, str] = field(default_factory=dict)
    quant: str = "Q4_M"
    gguf_path: Optional[str] = None
    model: Any = None
    device: str = "cpu"

    def load(self) -> None:
        quant = self.quant.lower()
        selected = select_quantized_model(self.resolved_models, "ltx2_gguf", quant)
        if not selected:
            logger.warning("No GGUF LTX-2 model found in resolved index; using planner-only mode")
            self.gguf_path = None
            self.model = None
            self.device = "cpu"
            return
        self.gguf_path = selected

        # llama.cpp-style loading path for GGUF artifacts.
        try:
            from llama_cpp import Llama

            n_gpu_layers = int(torch.cuda.is_available()) * int(
                float(os.environ.get("MILIMO_LTX_GPU_LAYERS", "24"))
            )
            self.model = Llama(
                model_path=self.gguf_path,
                n_ctx=int(os.environ.get("MILIMO_LTX_N_CTX", "4096")),
                n_gpu_layers=n_gpu_layers,
                offload_kqv=True,
                logits_all=False,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            self.device = "cuda" if torch.cuda.is_available() and n_gpu_layers > 0 else "cpu"
            logger.info("Loaded GGUF LTX backend via llama.cpp (%s)", self.gguf_path)
        except Exception as exc:
            logger.warning("llama.cpp GGUF load unavailable, trying Unsloth fallback: %s", exc)
            try:
                from unsloth import FastLanguageModel

                self.model, _ = FastLanguageModel.from_pretrained(
                    model_name=self.gguf_path,
                    max_seq_length=int(os.environ.get("MILIMO_LTX_N_CTX", "4096")),
                    load_in_4bit=True,
                )
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loaded GGUF LTX backend via Unsloth (%s)", self.gguf_path)
            except Exception as unsloth_exc:
                self.model = None
                self.device = "cpu"
                logger.warning("GGUF runtime load fallback (planner-only mode): %s", unsloth_exc)

    def attention_overrides(self) -> Dict[str, Any]:
        flash_ok = bool(torch.cuda.is_available())
        return {
            "enable_memory_efficient_attention": True,
            "enable_flash_attention_fallback": flash_ok,
            "enable_paged_attention": True,
            "enable_kv_cache_compression": True,
        }

    def to_plan(self, settings: Dict[str, Any], mode: str) -> ExecutionPlan:
        vram_gb = get_total_vram_gb() or 0.0
        width = int(settings.get("width", 768))
        height = int(settings.get("height", 512))
        frames = int(settings.get("num_frames", 121))
        chunk_size = int(settings.get("temporal_chunk_size", _choose_chunk_size(vram_gb)))
        grid = settings.get("latent_tile", _choose_tile_grid(vram_gb))
        overlap = int(settings.get("chunk_overlap", 1))

        plan = ExecutionPlan(
            mode=mode,
            quantization=self.quant,
            cpu_offload=True,
            temporal_chunk_size=chunk_size,
            temporal_windows=_build_temporal_windows(frames, chunk_size, overlap=overlap),
            tile_grid=grid,
            tiles=_build_tiles(width, height, grid),
            attention_backend="memory-efficient",
            kv_cache_compression=True,
            paged_attention=True,
            extra={
                "ltx_gguf_path": self.gguf_path,
                **self.attention_overrides(),
                "enable_unsloth_dynamic_quant": True,
                "chunk_overlap": overlap,
            },
        )
        return plan


@dataclass
class LowVRAMFluxBackend:
    resolved_models: Dict[str, str] = field(default_factory=dict)
    quant_bits: int = 4
    model_path: Optional[str] = None

    def load(self) -> None:
        prefer = "q4" if self.quant_bits == 4 else "q8"
        selected = select_quantized_model(self.resolved_models, "flux2_klein", prefer)
        if not selected:
            logger.warning("No quantized Flux model found in resolved index; using planner-only mode")
            self.model_path = None
            return
        self.model_path = selected
        logger.info("Selected Flux low-VRAM model: %s", self.model_path)

    def _xformers_enabled(self) -> bool:
        try:
            import xformers  # noqa: F401

            return True
        except Exception:
            return False

    def to_plan(self, settings: Dict[str, Any], mode: str) -> ExecutionPlan:
        vram_gb = get_total_vram_gb() or 0.0
        width = int(settings.get("width", 1024))
        height = int(settings.get("height", 1024))
        grid = settings.get("latent_tile", _choose_tile_grid(vram_gb))
        chunk_size = 1
        return ExecutionPlan(
            mode=mode,
            quantization=f"{self.quant_bits}bit",
            cpu_offload=True,
            temporal_chunk_size=chunk_size,
            temporal_windows=[TemporalWindow(start=0, end=1)],
            tile_grid=grid,
            tiles=_build_tiles(width, height, grid),
            attention_backend="xformers" if self._xformers_enabled() else "flash-fallback",
            kv_cache_compression=False,
            paged_attention=False,
            extra={
                "flux_model_path": self.model_path,
                "enable_xformers_attention": self._xformers_enabled(),
                "enable_flash_attention_fallback": True,
                "enable_tiled_vae": True,
                "enable_latent_tiling": True,
            },
        )


@dataclass
class LowVRAMSAMBackend:
    resolved_models: Dict[str, str] = field(default_factory=dict)
    model_path: Optional[str] = None
    use_cpu: bool = False

    def load(self) -> None:
        self.model_path = self.resolved_models.get("sam3_quant")
        if not self.model_path:
            logger.warning("No quantized SAM3 model found in resolved index; using planner-only mode")
            self.model_path = None
        vram = get_total_vram_gb() or 0.0
        self.use_cpu = (not torch.cuda.is_available()) or vram < 8.0
        logger.info("SAM low-VRAM backend path=%s cpu_fallback=%s", self.model_path, self.use_cpu)

    def to_plan(self, mode: str) -> ExecutionPlan:
        return ExecutionPlan(
            mode=mode,
            quantization="8bit",
            cpu_offload=self.use_cpu,
            temporal_chunk_size=1,
            temporal_windows=[TemporalWindow(start=0, end=1)],
            tile_grid="2x2",
            tiles=[],
            attention_backend="na",
            kv_cache_compression=False,
            paged_attention=False,
            extra={
                "sam_model_path": self.model_path,
                "sam_quant_bits": 8,
                "sam_device": "cpu" if self.use_cpu else "cuda",
            },
        )


@dataclass
class HighVRAMPipeline(VideoPipeline, ImagePipeline, SegmentationPipeline):
    resolved_models: Dict[str, str] = field(default_factory=dict)

    def plan_video(self, settings: Dict[str, Any]) -> ExecutionPlan:
        frames = int(settings.get("num_frames", 121))
        width = int(settings.get("width", 768))
        height = int(settings.get("height", 512))
        return ExecutionPlan(
            mode="high",
            quantization="fp16/bf16",
            cpu_offload=False,
            temporal_chunk_size=max(16, frames),
            temporal_windows=[TemporalWindow(start=0, end=frames)],
            tile_grid="1x1",
            tiles=_build_tiles(width, height, "2x2")[:1],
            attention_backend="flash",
            kv_cache_compression=False,
            paged_attention=False,
        )

    def generate_video(self, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": prompt, "settings": settings, "mode": "high", "plan": self.plan_video(settings)}

    def plan_image(self, settings: Dict[str, Any]) -> ExecutionPlan:
        width = int(settings.get("width", 1024))
        height = int(settings.get("height", 1024))
        return ExecutionPlan(
            mode="high",
            quantization="fp16/bf16",
            cpu_offload=False,
            temporal_chunk_size=1,
            temporal_windows=[TemporalWindow(start=0, end=1)],
            tile_grid="1x1",
            tiles=_build_tiles(width, height, "2x2")[:1],
            attention_backend="flash",
            kv_cache_compression=False,
            paged_attention=False,
        )

    def generate_image(self, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": prompt, "settings": settings, "mode": "high", "plan": self.plan_image(settings)}

    def plan_segmentation(self, settings: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        return ExecutionPlan(
            mode="high",
            quantization="fp16",
            cpu_offload=False,
            temporal_chunk_size=1,
            temporal_windows=[TemporalWindow(start=0, end=1)],
            tile_grid="1x1",
            tiles=[],
            attention_backend="na",
            kv_cache_compression=False,
            paged_attention=False,
        )

    def segment(self, frame: Any) -> Dict[str, Any]:
        return {"mode": "high", "frame": frame, "plan": self.plan_segmentation()}


@dataclass
class LowVRAMPipeline(VideoPipeline, ImagePipeline, SegmentationPipeline):
    resolved_models: Dict[str, str] = field(default_factory=dict)
    mode: str = "low"

    def _ltx_backend(self, settings: Dict[str, Any]) -> LowVRAMLTX2Backend:
        backend = LowVRAMLTX2Backend(
            resolved_models=self.resolved_models,
            quant=str(settings.get("ltx_quant", "Q4_M")),
        )
        backend.load()
        return backend

    def _flux_backend(self, settings: Dict[str, Any]) -> LowVRAMFluxBackend:
        backend = LowVRAMFluxBackend(
            resolved_models=self.resolved_models,
            quant_bits=int(settings.get("flux_quant_bits", 4)),
        )
        backend.load()
        return backend

    def _sam_backend(self) -> LowVRAMSAMBackend:
        backend = LowVRAMSAMBackend(resolved_models=self.resolved_models)
        backend.load()
        return backend

    def plan_video(self, settings: Dict[str, Any]) -> ExecutionPlan:
        return self._ltx_backend(settings).to_plan(settings, self.mode)

    def generate_video(self, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        plan = self.plan_video(settings)
        tuned = dict(settings)
        tuned.update(plan.extra)
        tuned["ltx_quant"] = plan.quantization
        tuned["enable_cpu_offload"] = plan.cpu_offload
        tuned["temporal_chunk_size"] = plan.temporal_chunk_size
        tuned["latent_tile"] = plan.tile_grid
        tuned["low_vram_temporal_windows"] = [w.__dict__ for w in plan.temporal_windows]
        tuned["low_vram_tiles"] = [t.__dict__ for t in plan.tiles]
        tuned["low_vram_chunk_overlap"] = int(plan.extra.get("chunk_overlap", 1))
        return {"prompt": prompt, "settings": tuned, "mode": self.mode, "plan": plan}

    def plan_image(self, settings: Dict[str, Any]) -> ExecutionPlan:
        return self._flux_backend(settings).to_plan(settings, self.mode)

    def generate_image(self, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        plan = self.plan_image(settings)
        tuned = dict(settings)
        tuned.update(plan.extra)
        tuned["flux_quant_bits"] = 4 if "4" in plan.quantization else 8
        tuned["enable_cpu_offload"] = plan.cpu_offload
        tuned["latent_tile"] = plan.tile_grid
        tuned["low_vram_tiles"] = [t.__dict__ for t in plan.tiles]
        return {"prompt": prompt, "settings": tuned, "mode": self.mode, "plan": plan}

    def plan_segmentation(self, settings: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        return self._sam_backend().to_plan(self.mode)

    def segment(self, frame: Any) -> Dict[str, Any]:
        plan = self.plan_segmentation()
        return {"mode": self.mode, "frame": frame, "plan": plan, **plan.extra}
