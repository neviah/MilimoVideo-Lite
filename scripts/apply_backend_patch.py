import argparse
import os
import shutil
from pathlib import Path


def copy_patch_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def patch_once(path: Path, needle: str, inject: str) -> None:
    text = path.read_text(encoding="utf-8")
    if inject in text:
        return
    if needle not in text:
        raise RuntimeError(f"Could not patch {path}: needle not found")
    path.write_text(text.replace(needle, needle + inject), encoding="utf-8")


def remove_text(path: Path, target: str) -> None:
    text = path.read_text(encoding="utf-8")
    if target in text:
        path.write_text(text.replace(target, ""), encoding="utf-8")


def patch_video_task(backend_dir: Path) -> None:
    file_path = backend_dir / "tasks" / "video.py"
    patch_once(
        file_path,
        "logger = logging.getLogger(__name__)\n",
        "\nfrom milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task\n",
    )
    patch_once(
        file_path,
        "    update_job_db(job_id, \"processing\")\n",
        "    params = adjust_video_params_for_mode(params)\n",
    )
    patch_once(
        file_path,
        "    params = adjust_video_params_for_mode(params)\n",
        "    before_video_task(job_id, params)\n",
    )


def patch_image_task(backend_dir: Path) -> None:
    file_path = backend_dir / "tasks" / "image.py"
    patch_once(
        file_path,
        "logger = logging.getLogger(__name__)\n",
        "\nfrom milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task\n",
    )
    patch_once(
        file_path,
        "    update_job_db(job_id, \"processing\")\n",
        "    params = adjust_image_params_for_mode(params)\n",
    )
    patch_once(
        file_path,
        "    params = adjust_image_params_for_mode(params)\n",
        "    before_image_task(job_id, params)\n",
    )


def patch_flux_wrapper(backend_dir: Path) -> None:
    file_path = backend_dir / "models" / "flux_wrapper.py"
    if not file_path.exists():
        return

    needle = (
        "            if os.path.exists(qwen_tokenizer_path):\n"
        "                os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = qwen_tokenizer_path\n"
    )
    inject = (
        "            has_triton = False\n"
        "            try:\n"
        "                import triton  # type: ignore  # noqa: F401\n"
        "                has_triton = True\n"
        "            except Exception:\n"
        "                has_triton = False\n"
        "\n"
        "            if self.device == \"cuda\" and not has_triton:\n"
        "                # Windows CUDA often lacks Triton wheels; force non-FP8 text encoder.\n"
        "                cuda_qwen = os.environ.get(\"MILIMO_QWEN3_CUDA_NO_TRITON_PATH\", \"Qwen/Qwen3-8B\")\n"
        "                cuda_tok = os.environ.get(\"MILIMO_QWEN3_CUDA_NO_TRITON_TOKENIZER\", cuda_qwen)\n"
        "                os.environ[\"QWEN3_8B_PATH\"] = cuda_qwen\n"
        "                os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = cuda_tok\n"
        "                logger.warning(\"Triton not available on CUDA runtime. Using non-FP8 Qwen fallback.\")\n"

        "            if self.device == \"cpu\":\n"
        "                # CPU-safe fallback: avoid default FP8 checkpoint which requires GPU/XPU.\n"
        "                cpu_qwen = os.environ.get(\"MILIMO_QWEN3_CPU_PATH\", \"Qwen/Qwen3-8B\")\n"
        "                cpu_tok = os.environ.get(\"MILIMO_QWEN3_CPU_TOKENIZER\", cpu_qwen)\n"
        "                os.environ.setdefault(\"QWEN3_8B_PATH\", cpu_qwen)\n"
        "                os.environ.setdefault(\"QWEN3_8B_TOKENIZER_PATH\", cpu_tok)\n"
        "                logger.warning(\"CUDA not available for Flux text encoder. Using non-FP8 CPU fallback model.\")\n"
    )
    patch_once(file_path, needle, inject)

    # Add resilient retry chain around text encoder load for CPU fallback mode.
    load_line = "            self.text_encoder = load_text_encoder(model_name, device=self.device)\n"
    if load_line in file_path.read_text(encoding="utf-8"):
        retry_block = (
            "            try:\n"
            "                self.text_encoder = load_text_encoder(model_name, device=self.device)\n"
            "            except Exception as text_exc:\n"
            "                err_text = str(text_exc)\n"
            "                needs_non_fp8_fallback = (\n"
            "                    self.device == \"cpu\"\n"
            "                    or \"No module named 'triton'\" in err_text\n"
            "                    or \"finegrained_fp8\" in err_text\n"
            "                    or \"FP8\" in err_text\n"
            "                )\n"
            "                if not needs_non_fp8_fallback:\n"
            "                    raise\n"
            "                logger.warning(f\"Primary text encoder failed, trying non-FP8 fallback: {text_exc}\")\n"
            "                fallback_chain = [\n"
            "                    os.environ.get(\"MILIMO_QWEN3_CUDA_NO_TRITON_PATH\", \"Qwen/Qwen3-8B\"),\n"
            "                    os.environ.get(\"MILIMO_QWEN3_CPU_PATH\", \"Qwen/Qwen3-8B\"),\n"
            "                    \"Qwen/Qwen3-8B-Base\",\n"
            "                    \"Qwen/Qwen2.5-7B-Instruct\",\n"
            "                ]\n"
            "                loaded = False\n"
            "                for fb in fallback_chain:\n"
            "                    if not fb:\n"
            "                        continue\n"
            "                    os.environ[\"QWEN3_8B_PATH\"] = fb\n"
            "                    os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = fb\n"
            "                    logger.warning(f\"Retrying CPU text encoder with fallback model: {fb}\")\n"
            "                    try:\n"
            "                        self.text_encoder = load_text_encoder(model_name, device=self.device)\n"
            "                        loaded = True\n"
            "                        break\n"
            "                    except Exception as fb_exc:\n"
            "                        logger.warning(f\"CPU fallback model failed ({fb}): {fb_exc}\")\n"
            "                if not loaded:\n"
            "                    raise\n"
        )
        patch_once(file_path, load_line, retry_block)


def patch_server_startup(backend_dir: Path) -> None:
    file_path = backend_dir / "server.py"
    patch_once(
        file_path,
        "from events import event_manager\n",
        "from milimovideo_lite.runtime import bootstrap_lite_runtime, describe_runtime\n",
    )
    patch_once(
        file_path,
        "logger = logging.getLogger(__name__)\n",
        "logger.info(\"MilimoVideo-Lite runtime hook loaded\")\n",
    )
    patch_once(
        file_path,
        "    init_db()\n",
        "    bootstrap_lite_runtime()\n    logger.info(f\"MilimoVideo-Lite runtime: {describe_runtime()}\")\n",
    )

    # Cleanup for older buggy injections that referenced symbols before import.
    remove_text(file_path, "_ = (HighVRAMPipeline, LowVRAMPipeline)\n")
    remove_text(file_path, "from milimovideo_lite.pipelines import HighVRAMPipeline, LowVRAMPipeline\n")


def patch_sam_startup(project_root: Path) -> None:
    file_path = project_root / "sam3" / "start_sam_server.py"
    if not file_path.exists():
        return
    patch_once(
        file_path,
        "logger = logging.getLogger(\"SAM3_Server\")\n",
        "\ntry:\n    from milimovideo_lite.runtime import get_sam_runtime_overrides\nexcept Exception:\n    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"backend\"))\n    if backend_dir not in sys.path:\n        sys.path.append(backend_dir)\n    from milimovideo_lite.runtime import get_sam_runtime_overrides\n\ndef _sam_total_vram_gb() -> float:\n    if torch.cuda.is_available():\n        try:\n            return float(torch.cuda.get_device_properties(0).total_memory) / (1024.0 ** 3)\n        except Exception:\n            return 0.0\n    return 0.0\n",
    )
    patch_once(
        file_path,
        "        device = \"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\"\n",
        "        sam_overrides = get_sam_runtime_overrides()\n        if sam_overrides.get(\"sam_device\") == \"cpu\":\n            logger.warning(\"MilimoVideo-Lite forcing SAM3 CPU fallback\")\n            device = \"cpu\"\n        elif device == \"cuda\" and _sam_total_vram_gb() < 8.0:\n            logger.warning(\"VRAM < 8GB detected, forcing SAM3 to CPU fallback\")\n            device = \"cpu\"\n",
    )


def patch_model_engine(backend_dir: Path) -> None:
    file_path = backend_dir / "model_engine.py"
    if not file_path.exists():
        return

    patch_once(
        file_path,
        "            \"gemma_root\": os.path.join(models_dir, \"text_encoders\", \"gemma3\"),\n",
        "            \"gemma_root\": os.path.join(config.BACKEND_DIR, \"models\", \"text_encoders\", \"gemma3\"),\n",
    )

    fp8_block_old = (
        "            is_mps = (device == \"mps\")\n"
        "            fp8 = False if is_mps else True \n"
    )
    fp8_block_new = (
        "            is_mps = (device == \"mps\")\n"
        "            fp8 = False\n"
        "            if device == \"cuda\":\n"
        "                try:\n"
        "                    major, minor = torch.cuda.get_device_capability(0)\n"
        "                    # FP8 needs Ada/Hopper-class GPU support (RTX 40xx+ or better).\n"
        "                    fp8 = (major, minor) >= (8, 9)\n"
        "                    if not fp8:\n"
        "                        logger.warning(f\"Disabling FP8 transformer on unsupported GPU capability {major}.{minor}\")\n"
        "                except Exception as cap_exc:\n"
        "                    logger.warning(f\"Unable to detect CUDA capability for FP8 gate: {cap_exc}\")\n"
        "                    fp8 = False\n"
    )
    patch_once(file_path, fp8_block_old, fp8_block_new)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply MilimoVideo-Lite backend patch")
    parser.add_argument("project_root", help="Path to cloned milimovideo repo")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    project_root = Path(args.project_root).resolve()
    backend_dir = project_root / "backend"
    if not backend_dir.exists():
        raise RuntimeError(f"backend directory not found: {backend_dir}")

    patch_src = workspace_root / "patches" / "backend" / "milimovideo_lite"
    patch_dst = backend_dir / "milimovideo_lite"
    copy_patch_tree(patch_src, patch_dst)

    patch_video_task(backend_dir)
    patch_image_task(backend_dir)
    patch_flux_wrapper(backend_dir)
    patch_model_engine(backend_dir)
    patch_server_startup(backend_dir)
    patch_sam_startup(project_root)

    print("Applied MilimoVideo-Lite backend patches")


if __name__ == "__main__":
    main()
