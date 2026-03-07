# MilimoVideo-Lite

MilimoVideo-Lite is a Pinokio app wrapper for [mainza-ai/milimovideo](https://github.com/mainza-ai/milimovideo) that preserves the existing frontend UI and adds backend-side VRAM-aware behavior.

## What This App Does

- Clones `mainza-ai/milimovideo` into `sandbox/workspace/milimovideo`
- Creates and uses a sandboxed Python venv in `sandbox/venv`
- Installs backend Python dependencies and frontend Node dependencies
- Applies backend patch files only (frontend untouched)
- Starts backend API and existing frontend UI together
- Keeps all runtime data under `sandbox/` only
- Automatically checks and downloads required models into `backend/models` when backend starts

## Pinokio Files

- `pinokio.js`: Dynamic app menu (Install, Start, Update, logs)
- `install.js`: Full setup and patch application
- `update.js`: Git pull + reapply backend patches
- `start.js`: Starts backend and frontend via `scripts/start_milimovideo_lite.js`

## Backend Additions

Injected under cloned repo path:

- `backend/milimovideo_lite/vram.py`
- `backend/milimovideo_lite/model_manager.py`
- `backend/milimovideo_lite/interfaces.py`
- `backend/milimovideo_lite/pipelines.py`
- `backend/milimovideo_lite/runtime.py`
- `backend/milimovideo_lite/dry_run.py`

`scripts/apply_backend_patch.py` also patches:

- `backend/server.py` startup to bootstrap runtime/model checks
- `backend/tasks/video.py` to apply low-VRAM tuning before generation
- `backend/tasks/image.py` to apply low-VRAM tuning before generation
- `sam3/start_sam_server.py` to force CPU fallback under 8GB VRAM

## VRAM Modes

`get_vram_mode()` returns one of:

- `high`
- `low`
- `auto`

Set using environment variable:

```bash
MILIMO_VRAM_MODE=auto
```

`auto` resolution logic:

- VRAM `>= 16GB` -> high profile
- VRAM `6GB to <16GB` -> low profile
- VRAM `< 6GB` -> CPU fallback profile

## Low-VRAM Behavior

Low mode applies backend-only tuning while preserving endpoint contracts:

- LTX pathway:
  - GGUF backend class: `LowVRAMLTX2Backend`
  - quantization: `Q4_M` or `Q6_K`
  - llama.cpp-style GGUF loading with Unsloth fallback
  - memory-efficient / flash-fallback / paged attention flags
  - KV-cache compression flag
  - temporal chunk windows (default 4-8)
  - latent tiling (`2x2` or `3x3`) + tile stitching helpers
  - CPU offload flag
- Flux pathway:
  - backend class: `LowVRAMFluxBackend`
  - quantization: 4-bit or 8-bit
  - xFormers + flash-fallback flags
  - tiled VAE encode/decode
  - latent tiling and tile stitching support
  - CPU offload flag
  - low-VRAM auto caps for image generation defaults: `640x640`, `8` steps
  - low-VRAM auto caps for element/character visuals defaults: `512x512`, `6` steps
- SAM pathway:
  - backend class: `LowVRAMSAMBackend`
  - 8-bit quantized runtime intent
  - forced CPU fallback under 8GB VRAM

## Model Downloading

At backend startup, model manager:

- checks `backend/models`
- downloads missing files from HuggingFace with resume enabled
- verifies checksums when SHA256 env vars are provided
- logs progress and download status to backend logs

Model source defaults are configurable via environment variables:

- `MILIMO_LTX2_GGUF_Q4_REPO`, `MILIMO_LTX2_GGUF_Q4_FILE`, `MILIMO_LTX2_GGUF_Q4_SHA256`
- `MILIMO_LTX2_GGUF_Q6_REPO`, `MILIMO_LTX2_GGUF_Q6_FILE`, `MILIMO_LTX2_GGUF_Q6_SHA256`
- `MILIMO_FLUX2_Q4_REPO`, `MILIMO_FLUX2_Q4_FILE`, `MILIMO_FLUX2_Q4_SHA256`
- `MILIMO_FLUX2_Q8_REPO`, `MILIMO_FLUX2_Q8_FILE`, `MILIMO_FLUX2_Q8_SHA256`
- `MILIMO_FLUX2_SAFETENSORS_REPO`, `MILIMO_FLUX2_SAFETENSORS_FILE`, `MILIMO_FLUX2_SAFETENSORS_SHA256`
- `MILIMO_FLUX2_AE_REPO`, `MILIMO_FLUX2_AE_FILE`, `MILIMO_FLUX2_AE_SHA256`
- `MILIMO_SAM3_REPO`, `MILIMO_SAM3_FILE`, `MILIMO_SAM3_SHA256`
- `MILIMO_GEMMA3_REPO`
- `MILIMO_GEMMA3_CONFIG_FILE`, `MILIMO_GEMMA3_INDEX_FILE`
- `MILIMO_GEMMA3_SHARD1_FILE`, `MILIMO_GEMMA3_SHARD2_FILE`
- `MILIMO_GEMMA3_TOKENIZER_FILE`, `MILIMO_GEMMA3_TOKENIZER_JSON_FILE`, `MILIMO_GEMMA3_TOKENIZER_CONFIG_FILE`
- `MILIMO_GEMMA3_SPECIAL_TOKENS_FILE`, `MILIMO_GEMMA3_PREPROCESSOR_FILE`

Validated default public sources currently used:

- LTX2 GGUF: `unsloth/LTX-2-GGUF` (`Q4_K_M`, `Q6_K`)
- Flux2 Klein GGUF: `unsloth/FLUX.2-klein-9B-GGUF` (`Q4_K_M`, `Q8_0`)
- Flux2 Klein weights (required by current Flux backend): `dci05049/flux2-klein-9b` (`flux-2-klein-9b.safetensors`)
- Flux2 AE (native): `dci05049/flux2-klein-9b` (`flux2-vae.safetensors` saved as `backend/models/flux2/ae.safetensors`)
- SAM3 checkpoint: `1038lab/sam3` (`sam3.pt`)
- Gemma3 text encoder (LTX compatible): `unsloth/gemma-3-4b-it` (safetensors shards + tokenizer/processor files)

## RTX 30-Series Notes

- RTX 30-series cards are supported.
- FP8 transformer mode is now auto-disabled on CUDA GPUs with capability below 8.9 (includes most 30xx cards), preventing "FP8 not compatible" errors.
- Flux text encoder now falls back to non-FP8 Qwen automatically when Triton is unavailable on Windows CUDA setups.

## GPU Runtime Notes

- Install and update now run `python scripts/ensure_torch_cuda.py` to detect NVIDIA GPUs and switch PyTorch to a CUDA build when available.
- If CUDA wheels cannot be resolved for your environment, the app falls back to CPU mode and logs a warning.
- You can override tried CUDA indexes with `MILIMO_TORCH_CUDA_INDEXES` (comma-separated URLs).
- On Windows CUDA without Triton, Flux text encoder defaults to a non-FP8 compatible model (`Qwen/Qwen3-8B`) on CPU.
- Override behavior with `MILIMO_QWEN3_CUDA_NO_TRITON_PATH` and `MILIMO_QWEN3_CUDA_NO_TRITON_DEVICE`.
- Backend launch sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce allocator fragmentation.
- Override low-VRAM image caps with: `MILIMO_LOWVRAM_IMAGE_MAX_WIDTH`, `MILIMO_LOWVRAM_IMAGE_MAX_HEIGHT`, `MILIMO_LOWVRAM_IMAGE_MAX_STEPS`.
- Override low-VRAM element visual caps with: `MILIMO_LOWVRAM_ELEMENT_MAX_WIDTH`, `MILIMO_LOWVRAM_ELEMENT_MAX_HEIGHT`, `MILIMO_LOWVRAM_ELEMENT_MAX_STEPS`.

If you override with gated/private repos, set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

## Unified Pipeline Interface

The patch includes these abstractions:

- `VideoPipeline.generate_video(prompt, settings)`
- `ImagePipeline.generate_image(prompt, settings)`
- `SegmentationPipeline.segment(frame)`

Implementations:

- `HighVRAMPipeline` (keeps default behavior)
- `LowVRAMPipeline` (quantized/chunked/tiled/offload tuning)

Runtime selects automatically from VRAM mode and applies parameter routing internally while keeping all existing API endpoints unchanged.

## Dry-Run Validation

Use the validation helper after patching a clone:

```bash
python scripts/lite_dry_run.py sandbox/workspace/milimovideo
```

It prints:

- detected VRAM and selected mode
- resolved model index
- selected low/high pipeline routing
- 1-frame dummy planner inference
- process and CUDA memory stats

For quick validation without downloading large models:

```bash
set MILIMO_SKIP_MODEL_DOWNLOAD=1
python scripts/lite_dry_run.py sandbox/workspace/milimovideo
```
