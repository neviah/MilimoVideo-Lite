from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


VRAMMode = Literal["high", "low", "auto", "cpu"]


@dataclass
class TemporalWindow:
    start: int
    end: int
    overlap_left: int = 0
    overlap_right: int = 0


@dataclass
class TileSpec:
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class ExecutionPlan:
    mode: VRAMMode
    quantization: str
    cpu_offload: bool
    temporal_chunk_size: int
    temporal_windows: List[TemporalWindow] = field(default_factory=list)
    tile_grid: str = "2x2"
    tiles: List[TileSpec] = field(default_factory=list)
    attention_backend: str = "auto"
    kv_cache_compression: bool = True
    paged_attention: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class VideoPipeline(ABC):
    @abstractmethod
    def generate_video(self, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def plan_video(self, settings: Dict[str, Any]) -> ExecutionPlan:
        raise NotImplementedError


class ImagePipeline(ABC):
    @abstractmethod
    def generate_image(self, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def plan_image(self, settings: Dict[str, Any]) -> ExecutionPlan:
        raise NotImplementedError


class SegmentationPipeline(ABC):
    @abstractmethod
    def segment(self, frame: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def plan_segmentation(self, settings: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        raise NotImplementedError


class PipelineRouter(ABC):
    @abstractmethod
    def current_mode(self) -> VRAMMode:
        raise NotImplementedError

    @abstractmethod
    def video_pipeline(self) -> VideoPipeline:
        raise NotImplementedError

    @abstractmethod
    def image_pipeline(self) -> ImagePipeline:
        raise NotImplementedError

    @abstractmethod
    def segmentation_pipeline(self) -> SegmentationPipeline:
        raise NotImplementedError
