from __future__ import annotations

import json
import os
import time

import psutil
import torch

from .runtime import adjust_image_params_for_mode, adjust_video_params_for_mode, describe_runtime, get_router


def _mem_snapshot() -> dict:
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    gpu_alloc = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_alloc = int(torch.cuda.memory_allocated())
        gpu_reserved = int(torch.cuda.memory_reserved())
    return {
        "rss_mb": round(rss / (1024 * 1024), 2),
        "gpu_alloc_mb": round(gpu_alloc / (1024 * 1024), 2),
        "gpu_reserved_mb": round(gpu_reserved / (1024 * 1024), 2),
    }


def main() -> None:
    t0 = time.time()

    print("=== MilimoVideo-Lite Dry Run ===")
    runtime = describe_runtime()
    print("Runtime:")
    print(json.dumps(runtime, indent=2, default=str))

    router = get_router()
    print(f"Selected mode: {router.current_mode()}")

    base_video = {
        "prompt": "a cinematic test scene",
        "width": 768,
        "height": 512,
        "num_frames": 1,
        "num_inference_steps": 8,
    }
    base_image = {
        "prompt": "a cinematic keyframe",
        "width": 768,
        "height": 768,
        "num_inference_steps": 8,
    }

    tuned_video = adjust_video_params_for_mode(base_video)
    tuned_image = adjust_image_params_for_mode(base_image)

    print("Video plan fields:")
    print(json.dumps({
        "temporal_chunk_size": tuned_video.get("temporal_chunk_size"),
        "tile": tuned_video.get("latent_tile"),
        "windows": len(tuned_video.get("low_vram_temporal_windows", [])),
        "ltx_quant": tuned_video.get("ltx_quant"),
    }, indent=2))

    print("Image plan fields:")
    print(json.dumps({
        "flux_quant_bits": tuned_image.get("flux_quant_bits"),
        "tile": tuned_image.get("latent_tile"),
        "tiles": len(tuned_image.get("low_vram_tiles", [])),
        "xformers": tuned_image.get("enable_xformers_attention"),
    }, indent=2))

    # 1-frame dummy inference pass through planner interfaces.
    vp = router.video_pipeline()
    ip = router.image_pipeline()
    sp = router.segmentation_pipeline()

    _ = vp.generate_video("dummy", tuned_video)
    _ = ip.generate_image("dummy", tuned_image)
    _ = sp.segment(frame={"shape": [1, tuned_image["height"], tuned_image["width"], 3]})

    print("Memory:")
    print(json.dumps(_mem_snapshot(), indent=2))
    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
