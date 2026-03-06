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


def patch_server_startup(backend_dir: Path) -> None:
    file_path = backend_dir / "server.py"
    patch_once(
        file_path,
        "from events import event_manager\n",
        "from milimovideo_lite.runtime import bootstrap_lite_runtime, describe_runtime\nfrom milimovideo_lite.pipelines import HighVRAMPipeline, LowVRAMPipeline\n",
    )
    patch_once(
        file_path,
        "logger = logging.getLogger(__name__)\n",
        "logger.info(\"MilimoVideo-Lite runtime hook loaded\")\n_ = (HighVRAMPipeline, LowVRAMPipeline)\n",
    )
    patch_once(
        file_path,
        "    init_db()\n",
        "    bootstrap_lite_runtime()\n    logger.info(f\"MilimoVideo-Lite runtime: {describe_runtime()}\")\n",
    )


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
    patch_server_startup(backend_dir)
    patch_sam_startup(project_root)

    print("Applied MilimoVideo-Lite backend patches")


if __name__ == "__main__":
    main()
