from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import config

from .model_manager import ensure_models
from .interfaces import PipelineRouter
from .pipelines import HighVRAMPipeline, LowVRAMPipeline
from .vram import get_total_vram_gb, get_vram_mode, resolve_runtime_mode

logger = logging.getLogger(__name__)


_BOOTSTRAPPED = False
_MODE_CACHE = None
_MODELS_CACHE: Dict[str, str] = {}


class LitePipelineRouter(PipelineRouter):
    def __init__(self, mode: str, resolved_models: Dict[str, str]):
        self._mode = mode
        self._resolved_models = resolved_models
        self._high = HighVRAMPipeline(resolved_models=resolved_models)
        low_mode = "cpu" if mode == "cpu" else "low"
        self._low = LowVRAMPipeline(resolved_models=resolved_models, mode=low_mode)

    def current_mode(self) -> str:
        return self._mode

    def video_pipeline(self):
        return self._high if self._mode == "high" else self._low

    def image_pipeline(self):
        return self._high if self._mode == "high" else self._low

    def segmentation_pipeline(self):
        return self._high if self._mode == "high" else self._low


def _pick_pipeline(mode: str):
    return LitePipelineRouter(mode, _MODELS_CACHE)


def bootstrap_lite_runtime() -> None:
    global _BOOTSTRAPPED, _MODE_CACHE, _MODELS_CACHE
    if _BOOTSTRAPPED:
        return

    configured = get_vram_mode()
    effective = resolve_runtime_mode()
    _MODE_CACHE = effective

    logger.info(
        "MilimoVideo-Lite runtime mode configured=%s effective=%s vram_gb=%s",
        configured,
        effective,
        get_total_vram_gb(),
    )

    models_root = os.path.join(config.BACKEND_DIR, "models")
    try:
        if os.environ.get("MILIMO_SKIP_MODEL_DOWNLOAD", "0") == "1":
            logger.warning("MILIMO_SKIP_MODEL_DOWNLOAD=1 set, skipping model download")
            _MODELS_CACHE = {}
        else:
            _MODELS_CACHE = ensure_models(models_root, mode=effective)
    except Exception as exc:
        logger.warning("Model auto-download encountered an issue: %s", exc)
        _MODELS_CACHE = {}

    _BOOTSTRAPPED = True


def get_router() -> LitePipelineRouter:
    bootstrap_lite_runtime()
    mode = _MODE_CACHE or resolve_runtime_mode()
    return _pick_pipeline(mode)


def describe_runtime() -> Dict[str, Any]:
    bootstrap_lite_runtime()
    mode = _MODE_CACHE or resolve_runtime_mode()
    return {
        "configured_mode": get_vram_mode(),
        "effective_mode": mode,
        "vram_gb": get_total_vram_gb(),
        "models": _MODELS_CACHE,
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def adjust_video_params_for_mode(params: Dict[str, Any]) -> Dict[str, Any]:
    bootstrap_lite_runtime()
    tuned = dict(params)

    router = get_router()
    mode = router.current_mode()
    if mode == "high":
        return tuned

    planner = router.video_pipeline()
    result = planner.generate_video(tuned.get("prompt", ""), tuned)
    tuned.update(result.get("settings", {}))

    strict_caps = _is_truthy(os.environ.get("MILIMO_LOWVRAM_STRICT_CAPS", "0"))

    default_steps = _safe_int(os.environ.get("MILIMO_LOWVRAM_VIDEO_DEFAULT_STEPS", 24), 24)
    default_width = _safe_int(os.environ.get("MILIMO_LOWVRAM_VIDEO_DEFAULT_WIDTH", 768), 768)
    default_height = _safe_int(os.environ.get("MILIMO_LOWVRAM_VIDEO_DEFAULT_HEIGHT", 432), 432)

    hard_max_steps = _safe_int(os.environ.get("MILIMO_LOWVRAM_VIDEO_HARD_MAX_STEPS", 28), 28)
    hard_max_width = _safe_int(os.environ.get("MILIMO_LOWVRAM_VIDEO_HARD_MAX_WIDTH", 960), 960)
    hard_max_height = _safe_int(os.environ.get("MILIMO_LOWVRAM_VIDEO_HARD_MAX_HEIGHT", 544), 544)

    if strict_caps:
        tuned["num_inference_steps"] = min(_safe_int(tuned.get("num_inference_steps", 40), 40), default_steps)
        tuned["width"] = min(_safe_int(tuned.get("width", 768), 768), default_width)
        tuned["height"] = min(_safe_int(tuned.get("height", 512), 512), default_height)
    else:
        tuned["num_inference_steps"] = min(_safe_int(tuned.get("num_inference_steps", default_steps), default_steps), hard_max_steps)
        tuned["width"] = min(_safe_int(tuned.get("width", default_width), default_width), hard_max_width)
        tuned["height"] = min(_safe_int(tuned.get("height", default_height), default_height), hard_max_height)

    # Temporal chunking and latent tiling are injected as planner metadata.
    chunk_size = _safe_int(tuned.get("temporal_chunk_size", 6), 6)
    frames = _safe_int(tuned.get("num_frames", 121), 121)
    tuned["num_frames"] = max(chunk_size, min(frames, 121))

    if mode == "cpu":
        tuned["device"] = "cpu"
        tuned["enable_cpu_offload"] = True
        tuned["width"] = min(tuned["width"], _safe_int(os.environ.get("MILIMO_CPU_VIDEO_MAX_WIDTH", 512), 512))
        tuned["height"] = min(tuned["height"], _safe_int(os.environ.get("MILIMO_CPU_VIDEO_MAX_HEIGHT", 320), 320))
        tuned["num_inference_steps"] = min(
            tuned["num_inference_steps"],
            _safe_int(os.environ.get("MILIMO_CPU_VIDEO_MAX_STEPS", 16), 16),
        )

    return tuned


def adjust_image_params_for_mode(params: Dict[str, Any]) -> Dict[str, Any]:
    bootstrap_lite_runtime()
    tuned = dict(params)

    router = get_router()
    mode = router.current_mode()
    if mode == "high":
        return tuned

    planner = router.image_pipeline()
    result = planner.generate_image(tuned.get("prompt", ""), tuned)
    tuned.update(result.get("settings", {}))
    strict_caps = _is_truthy(os.environ.get("MILIMO_LOWVRAM_STRICT_CAPS", "0"))

    default_steps = _safe_int(os.environ.get("MILIMO_LOWVRAM_IMAGE_DEFAULT_STEPS", 8), 8)
    default_width = _safe_int(os.environ.get("MILIMO_LOWVRAM_IMAGE_DEFAULT_WIDTH", 640), 640)
    default_height = _safe_int(os.environ.get("MILIMO_LOWVRAM_IMAGE_DEFAULT_HEIGHT", 640), 640)

    hard_max_steps = _safe_int(os.environ.get("MILIMO_LOWVRAM_IMAGE_HARD_MAX_STEPS", 20), 20)
    hard_max_width = _safe_int(os.environ.get("MILIMO_LOWVRAM_IMAGE_HARD_MAX_WIDTH", 1024), 1024)
    hard_max_height = _safe_int(os.environ.get("MILIMO_LOWVRAM_IMAGE_HARD_MAX_HEIGHT", 1024), 1024)

    if strict_caps:
        tuned["num_inference_steps"] = min(_safe_int(tuned.get("num_inference_steps", 25), 25), default_steps)
        tuned["width"] = min(_safe_int(tuned.get("width", 1024), 1024), default_width)
        tuned["height"] = min(_safe_int(tuned.get("height", 1024), 1024), default_height)
    else:
        tuned["num_inference_steps"] = min(_safe_int(tuned.get("num_inference_steps", default_steps), default_steps), hard_max_steps)
        tuned["width"] = min(_safe_int(tuned.get("width", default_width), default_width), hard_max_width)
        tuned["height"] = min(_safe_int(tuned.get("height", default_height), default_height), hard_max_height)
    tuned["enable_ae"] = True
    tuned["enable_true_cfg"] = False

    if mode == "cpu":
        tuned["width"] = min(tuned["width"], _safe_int(os.environ.get("MILIMO_CPU_IMAGE_MAX_WIDTH", 384), 384))
        tuned["height"] = min(tuned["height"], _safe_int(os.environ.get("MILIMO_CPU_IMAGE_MAX_HEIGHT", 384), 384))
        tuned["num_inference_steps"] = min(
            tuned["num_inference_steps"],
            _safe_int(os.environ.get("MILIMO_CPU_IMAGE_MAX_STEPS", 6), 6),
        )

    return tuned


def adjust_element_visual_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Tune element/character visual generation for low-VRAM systems."""
    bootstrap_lite_runtime()
    tuned = dict(params)

    router = get_router()
    mode = router.current_mode()
    if mode == "high":
        tuned.setdefault("num_inference_steps", _safe_int(os.environ.get("MILIMO_ELEMENT_DEFAULT_STEPS", 12), 12))
        tuned.setdefault("width", _safe_int(os.environ.get("MILIMO_ELEMENT_DEFAULT_WIDTH", 768), 768))
        tuned.setdefault("height", _safe_int(os.environ.get("MILIMO_ELEMENT_DEFAULT_HEIGHT", 768), 768))
        return tuned

    strict_caps = _is_truthy(os.environ.get("MILIMO_LOWVRAM_STRICT_CAPS", "0"))

    default_steps = _safe_int(os.environ.get("MILIMO_LOWVRAM_ELEMENT_DEFAULT_STEPS", 6), 6)
    default_width = _safe_int(os.environ.get("MILIMO_LOWVRAM_ELEMENT_DEFAULT_WIDTH", 512), 512)
    default_height = _safe_int(os.environ.get("MILIMO_LOWVRAM_ELEMENT_DEFAULT_HEIGHT", 512), 512)

    hard_max_steps = _safe_int(os.environ.get("MILIMO_LOWVRAM_ELEMENT_HARD_MAX_STEPS", 12), 12)
    hard_max_width = _safe_int(os.environ.get("MILIMO_LOWVRAM_ELEMENT_HARD_MAX_WIDTH", 768), 768)
    hard_max_height = _safe_int(os.environ.get("MILIMO_LOWVRAM_ELEMENT_HARD_MAX_HEIGHT", 768), 768)

    if strict_caps:
        tuned["num_inference_steps"] = min(_safe_int(tuned.get("num_inference_steps", 25), 25), default_steps)
        tuned["width"] = min(_safe_int(tuned.get("width", 1024), 1024), default_width)
        tuned["height"] = min(_safe_int(tuned.get("height", 1024), 1024), default_height)
    else:
        tuned["num_inference_steps"] = min(_safe_int(tuned.get("num_inference_steps", default_steps), default_steps), hard_max_steps)
        tuned["width"] = min(_safe_int(tuned.get("width", default_width), default_width), hard_max_width)
        tuned["height"] = min(_safe_int(tuned.get("height", default_height), default_height), hard_max_height)

    if mode == "cpu":
        tuned["width"] = min(tuned["width"], _safe_int(os.environ.get("MILIMO_CPU_ELEMENT_MAX_WIDTH", 384), 384))
        tuned["height"] = min(tuned["height"], _safe_int(os.environ.get("MILIMO_CPU_ELEMENT_MAX_HEIGHT", 384), 384))
        tuned["num_inference_steps"] = min(
            tuned["num_inference_steps"],
            _safe_int(os.environ.get("MILIMO_CPU_ELEMENT_MAX_STEPS", 4), 4),
        )

    return tuned


def get_sam_runtime_overrides() -> Dict[str, Any]:
    router = get_router()
    plan = router.segmentation_pipeline().plan_segmentation({})
    return dict(plan.extra)


def before_video_task(job_id: str, params: Dict[str, Any]) -> None:
    logger.info("MilimoVideo-Lite video task mode=%s job=%s", (_MODE_CACHE or resolve_runtime_mode()), job_id)
    windows = params.get("low_vram_temporal_windows")
    if windows:
        logger.info("Low-VRAM temporal windows=%s", len(windows))


def before_image_task(job_id: str, params: Dict[str, Any]) -> None:
    logger.info("MilimoVideo-Lite image task mode=%s job=%s", (_MODE_CACHE or resolve_runtime_mode()), job_id)
