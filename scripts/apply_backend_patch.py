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


def replace_region(path: Path, start_marker: str, end_marker: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Could not patch {path}: start marker not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"Could not patch {path}: end marker not found")
    new_text = text[:start] + replacement + text[end:]
    path.write_text(new_text, encoding="utf-8")


def remove_text(path: Path, target: str) -> None:
    text = path.read_text(encoding="utf-8")
    if target in text:
        path.write_text(text.replace(target, ""), encoding="utf-8")


def replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def patch_video_task(backend_dir: Path) -> None:
    file_path = backend_dir / "tasks" / "video.py"
    patch_once(
        file_path,
        "logger = logging.getLogger(__name__)\n",
        "\nfrom milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task, ensure_video_runtime_ready\n",
    )
    replace_text(
        file_path,
        "from milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task\n",
        "from milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task, ensure_video_runtime_ready\n",
    )
    replace_text(
        file_path,
        "from milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task, ensure_video_runtime_ready\n\nfrom milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task, ensure_video_runtime_ready\n",
        "from milimovideo_lite.runtime import adjust_video_params_for_mode, before_video_task, ensure_video_runtime_ready\n",
    )
    patch_once(
        file_path,
        "    update_job_db(job_id, \"processing\")\n",
        "    ensure_video_runtime_ready()\n",
    )
    patch_once(
        file_path,
        "    ensure_video_runtime_ready()\n",
        "    params = adjust_video_params_for_mode(params)\n",
    )
    patch_once(
        file_path,
        "    params = adjust_video_params_for_mode(params)\n",
        "    before_video_task(job_id, params)\n",
    )
    replace_text(file_path, "logger.info(f\"✓ Path exists: {resolved_abs}\")", "logger.info(f\"[OK] Path exists: {resolved_abs}\")")
    replace_text(file_path, "logger.warning(f\"✗ Raw path not found: {path} (resolved as {resolved_abs})\")", "logger.warning(f\"[MISSING] Raw path not found: {path} (resolved as {resolved_abs})\")")


def patch_image_task(backend_dir: Path) -> None:
    file_path = backend_dir / "tasks" / "image.py"

    patch_once(
        file_path,
        "logger = logging.getLogger(__name__)\n",
        "\nfrom milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task, ensure_image_runtime_ready\n",
    )
    replace_text(
        file_path,
        "from milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task\n",
        "from milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task, ensure_image_runtime_ready\n",
    )
    replace_text(
        file_path,
        "from milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task, ensure_image_runtime_ready\n\nfrom milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task, ensure_image_runtime_ready\n",
        "from milimovideo_lite.runtime import adjust_image_params_for_mode, before_image_task, ensure_image_runtime_ready\n",
    )
    patch_once(
        file_path,
        "    update_job_db(job_id, \"processing\")\n",
        "    ensure_image_runtime_ready()\n",
    )
    patch_once(
        file_path,
        "    ensure_image_runtime_ready()\n",
        "    params = adjust_image_params_for_mode(params)\n",
    )
    patch_once(
        file_path,
        "    params = adjust_image_params_for_mode(params)\n",
        "    before_image_task(job_id, params)\n",
    )
    run_flux_replacement = (
        "        def _run_flux():\n"
        "             def flux_callback(step, total):\n"
        "                 if job_id in active_jobs:\n"
        "                     try:\n"
        "                         progress_pct = (step / total) * 100\n"
        "                         msg = f\"Generating Image ({step}/{total})\"\n"
        "                         active_jobs[job_id][\"progress\"] = int(progress_pct)\n"
        "                         active_jobs[job_id][\"status_message\"] = msg\n"
        "                         if active_jobs[job_id].get(\"cancelled\", False):\n"
        "                             raise RuntimeError(f\"Job {job_id} cancelled by user.\")\n"
        "\n"
        "                         # Allow pure cancellation check without clearing status\n"
        "                         if step >= 0:\n"
        "                            # Must run in asyncio thread\n"
        "                            asyncio.run_coroutine_threadsafe(\n"
        "                                broadcast_progress(job_id, active_jobs[job_id][\"progress\"], \"processing\", active_jobs[job_id][\"status_message\"]),\n"
        "                                loop\n"
        "                            )\n"
        "                     except Exception as e:\n"
        "                         if \"cancelled\" in str(e).lower():\n"
        "                             raise  # Rethrow so outer block catches it\n"
        "                         logger.error(f\"Error in flux callback: {e}\")\n"
        "\n"
        "             flux_model_path = params.get(\"flux_model_path\")\n"
        "             prev_flux_model_path = os.environ.get(\"KLEIN_9B_MODEL_PATH\")\n"
        "             if isinstance(flux_model_path, str) and flux_model_path:\n"
        "                 os.environ[\"KLEIN_9B_MODEL_PATH\"] = flux_model_path\n"
        "                 logger.info(f\"Using low-VRAM Flux model path override: {flux_model_path}\")\n"
        "\n"
        "             try:\n"
        "                 img = flux_inpainter.generate_image(\n"
        "                     prompt=prompt,\n"
        "                     width=width,\n"
        "                     height=height,\n"
        "                     guidance=cfg_scale,\n"
        "                     num_inference_steps=steps,\n"
        "                     ip_adapter_images=element_images,\n"
        "                     callback=flux_callback,\n"
        "                     seed=seed,\n"
        "                     negative_prompt=negative_prompt,\n"
        "                     enable_ae=enable_ae,\n"
        "                     enable_true_cfg=enable_true_cfg\n"
        "                 )\n"
        "             finally:\n"
        "                 if prev_flux_model_path is None:\n"
        "                     os.environ.pop(\"KLEIN_9B_MODEL_PATH\", None)\n"
        "                 else:\n"
        "                     os.environ[\"KLEIN_9B_MODEL_PATH\"] = prev_flux_model_path\n"
        "\n"
        "             # Save\n"
        "             if not project_id:\n"
        "                 raise ValueError(\"project_id required\")\n"
        "\n"
        "             paths = get_project_output_paths(job_id, project_id)\n"
        "             out_path = paths[\"output_path\"].replace(\".mp4\", \".jpg\")\n"
        "             thumb_path = paths[\"thumbnail_path\"]\n"
        "\n"
        "             os.makedirs(os.path.dirname(out_path), exist_ok=True)\n"
        "             os.makedirs(os.path.dirname(thumb_path), exist_ok=True)\n"
        "\n"
        "             img.save(out_path, quality=95)\n"
        "             img.resize((round(width/4), round(height/4))).save(thumb_path)\n"
        "\n"
        "             web_url = f\"/projects/{project_id}/generated/{os.path.basename(out_path)}\"\n"
        "             web_thumb = f\"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}\"\n"
        "\n"
        "             return web_url, web_thumb\n"
        "\n"
        "        web_url, web_thumb = await loop.run_in_executor(None, _run_flux)\n"
    )
    replace_region(
        file_path,
        "        def _run_flux():\n",
        "        web_url, web_thumb = await loop.run_in_executor(None, _run_flux)\n",
        run_flux_replacement,
    )


def patch_element_manager(backend_dir: Path) -> None:
    file_path = backend_dir / "managers" / "element_manager.py"
    if not file_path.exists():
        return

    text = file_path.read_text(encoding="utf-8")
    if "from milimovideo_lite.runtime import adjust_element_visual_params\n" not in text and "from milimovideo_lite.runtime import adjust_element_visual_params, ensure_image_runtime_ready\n" not in text:
        if "from models import Element, Asset\n" in text:
            patch_once(
                file_path,
                "from models import Element, Asset\n",
                "from milimovideo_lite.runtime import adjust_element_visual_params, ensure_image_runtime_ready\n",
            )
        elif "from database import engine, Element, Project, Asset\n" in text:
            patch_once(
                file_path,
                "from database import engine, Element, Project, Asset\n",
                "from milimovideo_lite.runtime import adjust_element_visual_params, ensure_image_runtime_ready\n",
            )
        else:
            raise RuntimeError(f"Could not patch {file_path}: expected import anchor not found")

    replace_text(
        file_path,
        "from milimovideo_lite.runtime import adjust_element_visual_params\n",
        "from milimovideo_lite.runtime import adjust_element_visual_params, ensure_image_runtime_ready\n",
    )

    patch_once(
        file_path,
        "        logger.info(f\"Generating visual for Element {element.name} ({element.id})\")\n",
        "        ensure_image_runtime_ready()\n        visual_params = adjust_element_visual_params({\"width\": 1024, \"height\": 1024, \"num_inference_steps\": 25})\n",
    )

    replace_text(
        file_path,
        "                width=1024,\n",
        "                width=visual_params.get(\"width\", 1024),\n",
    )
    replace_text(
        file_path,
        "                height=1024,\n",
        "                height=visual_params.get(\"height\", 1024),\n",
    )
    patch_once(
        file_path,
        "                guidance=guidance,\n",
        "                num_inference_steps=visual_params.get(\"num_inference_steps\", 25),\n",
    )


def patch_flux_wrapper(backend_dir: Path) -> None:
    file_path = backend_dir / "models" / "flux_wrapper.py"
    if not file_path.exists():
        return

    load_model_gate_replacement = (
        "    def load_model(self, enable_ae=True):\n"
        "        # In Lite mode we keep a single loaded Flux model to avoid OOM from repeated reloads.\n"
        "        if self.model_loaded:\n"
        "            if self.last_ae_enable_request != enable_ae:\n"
        "                logger.info(\n"
        "                    f\"AE Mode Changed (requested {self.last_ae_enable_request} -> {enable_ae}). Keeping loaded model to avoid VRAM reload.\"\n"
        "                )\n"
        "                self.last_ae_enable_request = enable_ae\n"
        "            return\n"
        "\n"
        "        self.last_ae_enable_request = enable_ae\n"
        "        logger.info(f\"Loading Flux 2 (Klein) Model on {self.device}. Native AE: {enable_ae}\")\n"
        "\n"
    )
    replace_region(
        file_path,
        "    def load_model(self, enable_ae=True):\n",
        "        # Unload conflicting models (e.g., LTX) before loading Flux\n",
        load_model_gate_replacement,
    )

    replace_text(
        file_path,
        "            os.environ[\"KLEIN_9B_MODEL_PATH\"] = os.path.join(base_path, \"flux-2-klein-9b.safetensors\")\n",
        "            # Respect externally provided model path (from low-VRAM planner); otherwise default to safetensors.\n            os.environ.setdefault(\"KLEIN_9B_MODEL_PATH\", os.path.join(base_path, \"flux-2-klein-9b.safetensors\"))\n            logger.info(f\"Flux model path resolved to: {os.environ.get('KLEIN_9B_MODEL_PATH')}\")\n",
    )

    qwen_replacement = (
        "            # Set Qwen paths\n"
        "            qwen_path = os.path.join(base_path, \"text_encoder\")\n"
        "            qwen_tokenizer_path = os.path.join(base_path, \"tokenizer\")\n"
        "            if os.path.exists(qwen_path):\n"
        "                os.environ[\"QWEN3_8B_PATH\"] = qwen_path\n"
        "            if os.path.exists(qwen_tokenizer_path):\n"
        "                os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = qwen_tokenizer_path\n"
        "\n"
        "            has_triton = False\n"
        "            try:\n"
        "                import triton  # type: ignore  # noqa: F401\n"
        "                has_triton = True\n"
        "            except Exception:\n"
        "                has_triton = False\n"
        "\n"
        "            if self.device == \"cuda\" and not has_triton:\n"
        "                # Windows CUDA often lacks Triton wheels; force smaller non-FP8 text encoder.\n"
        "                # Hard-pin compatible encoder for Flux Klein-9B in no-Triton CUDA mode.\n"
        "                cuda_qwen = \"Qwen/Qwen3-8B\"\n"
        "                cuda_tok = \"Qwen/Qwen3-8B\"\n"
        "                text_device = os.environ.get(\"MILIMO_QWEN3_CUDA_NO_TRITON_DEVICE\", \"cpu\")\n"
        "                os.environ[\"QWEN3_8B_PATH\"] = cuda_qwen\n"
        "                os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = cuda_tok\n"
        "                os.environ[\"MILIMO_QWEN3_TEXT_ENCODER_DEVICE\"] = text_device\n"
        "                logger.warning(f\"Triton not available on CUDA runtime. Using non-FP8 Qwen fallback on {text_device}.\")\n"
        "\n"
        "            if self.device == \"cpu\":\n"
        "                cpu_qwen = os.environ.get(\"MILIMO_QWEN3_CPU_PATH\", \"Qwen/Qwen3-8B\")\n"
        "                cpu_tok = os.environ.get(\"MILIMO_QWEN3_CPU_TOKENIZER\", cpu_qwen)\n"
        "                os.environ.setdefault(\"QWEN3_8B_PATH\", cpu_qwen)\n"
        "                os.environ.setdefault(\"QWEN3_8B_TOKENIZER_PATH\", cpu_tok)\n"
        "                logger.warning(\"CUDA not available for Flux text encoder. Using non-FP8 CPU fallback model.\")\n"
        "\n"
    )
    replace_region(
        file_path,
        "            # Set Qwen paths\n",
        "            model_name = \"flux.2-klein-9b\"\n",
        qwen_replacement,
    )
    cuda_no_triton_block = (
        "            if self.device == \"cuda\" and not has_triton:\n"
        "                # Windows CUDA often lacks Triton wheels; force compatible non-FP8 text encoder.\n"
        "                cuda_qwen = \"Qwen/Qwen3-8B\"\n"
        "                cuda_tok = \"Qwen/Qwen3-8B\"\n"
        "                text_device = os.environ.get(\"MILIMO_QWEN3_CUDA_NO_TRITON_DEVICE\", \"cpu\")\n"
        "                os.environ[\"QWEN3_8B_PATH\"] = cuda_qwen\n"
        "                os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = cuda_tok\n"
        "                os.environ[\"MILIMO_QWEN3_TEXT_ENCODER_DEVICE\"] = text_device\n"
        "                logger.warning(f\"Triton not available on CUDA runtime. Using non-FP8 Qwen fallback on {text_device}.\")\n"
        "\n"
    )
    replace_region(
        file_path,
        "            if self.device == \"cuda\" and not has_triton:\n",
        "            if self.device == \"cpu\":\n",
        cuda_no_triton_block,
    )

    text_encoder_replacement = (
        "            logger.info(\"Loading Text Encoder...\")\n"
        "            try:\n"
        "                text_encoder_device = os.environ.get(\"MILIMO_QWEN3_TEXT_ENCODER_DEVICE\", self.device)\n"
        "                self.text_encoder = load_text_encoder(model_name, device=text_encoder_device)\n"
        "            except Exception as text_exc:\n"
        "                err_text = str(text_exc)\n"
        "                needs_non_fp8_fallback = (\n"
        "                    self.device == \"cpu\"\n"
        "                    or \"No module named 'triton'\" in err_text\n"
        "                    or \"finegrained_fp8\" in err_text\n"
        "                    or \"FP8\" in err_text\n"
        "                    or \"out of memory\" in err_text.lower()\n"
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
        "                text_encoder_device = os.environ.get(\"MILIMO_QWEN3_TEXT_ENCODER_DEVICE\", self.device)\n"
        "                loaded = False\n"
        "                for fb in fallback_chain:\n"
        "                    if not fb:\n"
        "                        continue\n"
        "                    os.environ[\"QWEN3_8B_PATH\"] = fb\n"
        "                    os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = fb\n"
        "                    logger.warning(f\"Retrying text encoder with fallback model: {fb} (device={text_encoder_device})\")\n"
        "                    try:\n"
        "                        self.text_encoder = load_text_encoder(model_name, device=text_encoder_device)\n"
        "                        loaded = True\n"
        "                        break\n"
        "                    except Exception as fb_exc:\n"
        "                        logger.warning(f\"Text-encoder fallback failed ({fb}): {fb_exc}\")\n"
        "                if not loaded:\n"
        "                    raise\n"
        "\n"
        "            # Flux2 Klein-9B expects text embedding width 12288 (3 * hidden_size 4096).\n"
        "            expected_ctx_width = 12288\n"
        "            ctx_width = 0\n"
        "            try:\n"
        "                probe_ctx = self.text_encoder([\"compatibility probe\"])\n"
        "                ctx_width = int(probe_ctx.shape[-1])\n"
        "                del probe_ctx\n"
        "            except Exception as probe_exc:\n"
        "                logger.warning(f\"Unable to probe text encoder width: {probe_exc}\")\n"
        "            if ctx_width and ctx_width != expected_ctx_width:\n"
        "                logger.warning(\n"
        "                    f\"Incompatible text encoder width={ctx_width} (expected {expected_ctx_width}); forcing Qwen/Qwen3-8B fallback\"\n"
        "                )\n"
        "                os.environ[\"QWEN3_8B_PATH\"] = \"Qwen/Qwen3-8B\"\n"
        "                os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = \"Qwen/Qwen3-8B\"\n"
        "                text_encoder_device = os.environ.get(\"MILIMO_QWEN3_TEXT_ENCODER_DEVICE\", self.device)\n"
        "                self.text_encoder = load_text_encoder(model_name, device=text_encoder_device)\n"
        "                try:\n"
        "                    probe_ctx = self.text_encoder([\"compatibility probe\"])\n"
        "                    ctx_width = int(probe_ctx.shape[-1])\n"
        "                    del probe_ctx\n"
        "                    logger.info(f\"Text encoder width after fallback: {ctx_width}\")\n"
        "                except Exception as probe_exc:\n"
        "                    logger.warning(f\"Unable to verify text encoder width after fallback: {probe_exc}\")\n"
        "\n"
    )
    replace_region(
        file_path,
        "            logger.info(\"Loading Text Encoder...\")\n",
        "            # Load AutoEncoder\n",
        text_encoder_replacement,
    )

    positive_ctx_replacement = (
        "                ctx = self.text_encoder([prompt]).to(self.dtype)\n"
        "                try:\n"
        "                    ctx_width = int(ctx.shape[-1])\n"
        "                except Exception:\n"
        "                    ctx_width = 0\n"
        "                if ctx_width and ctx_width != 12288:\n"
        "                    logger.warning(\n"
        "                        f\"Prompt embedding width {ctx_width} is incompatible with Flux Klein-9B; reloading Qwen/Qwen3-8B\"\n"
        "                    )\n"
        "                    from flux2.util import load_text_encoder\n"
        "                    os.environ[\"QWEN3_8B_PATH\"] = \"Qwen/Qwen3-8B\"\n"
        "                    os.environ[\"QWEN3_8B_TOKENIZER_PATH\"] = \"Qwen/Qwen3-8B\"\n"
        "                    text_encoder_device = os.environ.get(\"MILIMO_QWEN3_TEXT_ENCODER_DEVICE\", self.device)\n"
        "                    self.text_encoder = load_text_encoder(model_name, device=text_encoder_device)\n"
        "                    ctx = self.text_encoder([prompt]).to(self.dtype)\n"
        "                ctx = ctx.to(self.device)\n"
        "                ctx, ctx_ids = batched_prc_txt(ctx)\n"
    )
    replace_region(
        file_path,
        "                ctx = self.text_encoder([prompt]).to(self.dtype)\n",
        "                ctx, ctx_ids = batched_prc_txt(ctx)\n",
        positive_ctx_replacement,
    )
    replace_text(
        file_path,
        "                ctx, ctx_ids = batched_prc_txt(ctx)\n                ctx, ctx_ids = batched_prc_txt(ctx)\n",
        "                ctx, ctx_ids = batched_prc_txt(ctx)\n",
    )
    patch_once(
        file_path,
        "                    ctx_uncond = self.text_encoder([neg_txt]).to(self.dtype)\n",
        "                    ctx_uncond = ctx_uncond.to(self.device)\n",
    )

    ae_replacement = (
        "            # Load AutoEncoder\n"
        "            # Prefer local ae.safetensors whenever present, even when Native AE toggle is off.\n"
        "            loaded_native = False\n"
        "\n"
        "            if os.path.exists(ae_path_file):\n"
        "                if enable_ae:\n"
        "                    logger.info(f\"Loading Native AutoEncoder from {ae_path_file}...\")\n"
        "                    loaded_native = True\n"
        "                else:\n"
        "                    logger.info(f\"Using local AutoEncoder from {ae_path_file} (native conditioning disabled)\")\n"
        "                os.environ[\"AE_MODEL_PATH\"] = ae_path_file\n"
        "                self.ae = load_ae(model_name, device=self.device)\n"
        "                self.ae.eval()\n"
        "                try:\n"
        "                    self.ae = self.ae.to(device=self.device, dtype=self.dtype)\n"
        "                except Exception:\n"
        "                    self.ae = self.ae.to(self.device)\n"
        "\n"
        "            elif os.path.exists(ae_path_dir) and os.path.exists(os.path.join(ae_path_dir, \"config.json\")):\n"
        "                logger.warning(f\"Using Diffusers VAE fallback (Native Requested: {enable_ae})\")\n"
        "                self.ae = FluxAEWrapper(ae_path_dir, self.device, dtype=self.dtype)\n"
        "            else:\n"
        "                logger.warning(\"No local VAE found, trying HuggingFace download...\")\n"
        "                self.ae = load_ae(model_name, device=self.device)\n"
        "                self.ae.eval()\n"
        "                try:\n"
        "                    self.ae = self.ae.to(device=self.device, dtype=self.dtype)\n"
        "                except Exception:\n"
        "                    self.ae = self.ae.to(self.device)\n"
        "\n"
    )
    replace_region(
        file_path,
        "            # Load AutoEncoder\n",
        "            self.using_native_ae = loaded_native\n",
        ae_replacement,
    )

    latent_ref_replacement = (
        "                dummy_img = Image.new(\"RGB\", (W, H), (0, 0, 0))\n"
        "                img_tensor = torch.from_numpy(np.array(dummy_img)).float() / 127.5 - 1.0\n"
        "                img_tensor = rearrange(img_tensor, \"h w c -> 1 c h w\").to(self.device).to(self.dtype)\n"
        "                try:\n"
        "                    ae_dtype = next(self.ae.parameters()).dtype\n"
        "                    img_tensor = img_tensor.to(ae_dtype)\n"
        "                except Exception:\n"
        "                    pass\n"
        "                z_shape_ref = self.ae.encode(img_tensor)\n"
        "\n"
    )
    text = file_path.read_text(encoding="utf-8")
    if (
        "                dummy_img = Image.new(\"RGB\", (W, H), (0, 0, 0))\n" in text
        and "                # Use Generator for noise to respect seed properly on all devices\n" in text
    ):
        replace_region(
            file_path,
            "                dummy_img = Image.new(\"RGB\", (W, H), (0, 0, 0))\n",
            "                # Use Generator for noise to respect seed properly on all devices\n",
            latent_ref_replacement,
        )
    replace_text(
        file_path,
        "                z_shape_ref = self.ae.encode(img_tensor)\n                z_shape_ref = self.ae.encode(img_tensor)\n",
        "                z_shape_ref = self.ae.encode(img_tensor)\n",
    )
    replace_text(
        file_path,
        "                z_shape_ref = self.ae.encode(img_tensor)\n                z_shape_ref = self.ae.encode(img_tensor)\n                \n",
        "                z_shape_ref = self.ae.encode(img_tensor)\n\n",
    )


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


def patch_flux2_text_encoder_defaults(project_root: Path) -> None:
    file_path = project_root / "flux2" / "src" / "flux2" / "text_encoder.py"
    if not file_path.exists():
        return
    replace_text(
        file_path,
        "    model_spec = os.environ.get(f\"QWEN3_{variant}_PATH\", f\"Qwen/Qwen3-{variant}-FP8\")\n",
        "    # Avoid FP8 default on consumer GPUs that do not support fine-grained FP8 kernels.\n    model_spec = os.environ.get(f\"QWEN3_{variant}_PATH\", f\"Qwen/Qwen3-{variant}\")\n",
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

    replace_text(
        file_path,
        "        ckpt_full = os.path.join(models_dir, \"checkpoints\", \"ltx-2-19b-distilled.safetensors\")\n",
        "        ckpt_full = os.path.join(config.BACKEND_DIR, \"models\", \"ltx2\", \"ltx-2-19b-distilled.safetensors\")\n",
    )
    replace_text(
        file_path,
        "        ckpt_fp8 = os.path.join(models_dir, \"checkpoints\", \"ltx-2-19b-distilled-fp8.safetensors\")\n",
        "        ckpt_fp8 = os.path.join(config.BACKEND_DIR, \"models\", \"ltx2\", \"ltx-2-19b-distilled-fp8.safetensors\")\n",
    )
    replace_text(
        file_path,
        "            \"distilled_lora_path\": os.path.join(models_dir, \"checkpoints\", \"ltx-2-19b-distilled-lora-384.safetensors\"), \n",
        "            \"distilled_lora_path\": os.path.join(config.BACKEND_DIR, \"models\", \"ltx2\", \"ltx-2-19b-distilled-lora-384.safetensors\"), \n",
    )
    replace_text(
        file_path,
        "            \"spatial_upsampler_path\": os.path.join(models_dir, \"upscalers\", \"ltx-2-spatial-upscaler-x2-1.0.safetensors\"),\n",
        "            \"spatial_upsampler_path\": os.path.join(config.BACKEND_DIR, \"models\", \"ltx2\", \"ltx-2-spatial-upscaler-x2-1.0.safetensors\"),\n",
    )
    replace_text(
        file_path,
        "            \"temporal_upsampler_path\": os.path.join(models_dir, \"upscalers\", \"ltx-2-temporal-upscaler-x2-1.0.safetensors\"),\n",
        "            \"temporal_upsampler_path\": os.path.join(config.BACKEND_DIR, \"models\", \"ltx2\", \"ltx-2-temporal-upscaler-x2-1.0.safetensors\"),\n",
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

    # Normalize LTX asset paths to backend/models/ltx2 regardless of line endings.
    text = file_path.read_text(encoding="utf-8")
    updated = text
    updated = updated.replace(
        'os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled.safetensors")',
        'os.path.join(config.BACKEND_DIR, "models", "ltx2", "ltx-2-19b-distilled.safetensors")',
    )
    updated = updated.replace(
        'os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-fp8.safetensors")',
        'os.path.join(config.BACKEND_DIR, "models", "ltx2", "ltx-2-19b-distilled-fp8.safetensors")',
    )
    updated = updated.replace(
        'os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-lora-384.safetensors")',
        'os.path.join(config.BACKEND_DIR, "models", "ltx2", "ltx-2-19b-distilled-lora-384.safetensors")',
    )
    updated = updated.replace(
        'os.path.join(models_dir, "upscalers", "ltx-2-spatial-upscaler-x2-1.0.safetensors")',
        'os.path.join(config.BACKEND_DIR, "models", "ltx2", "ltx-2-spatial-upscaler-x2-1.0.safetensors")',
    )
    updated = updated.replace(
        'os.path.join(models_dir, "upscalers", "ltx-2-temporal-upscaler-x2-1.0.safetensors")',
        'os.path.join(config.BACKEND_DIR, "models", "ltx2", "ltx-2-temporal-upscaler-x2-1.0.safetensors")',
    )
    if updated != text:
        file_path.write_text(updated, encoding="utf-8")


def patch_storyboard_routes(backend_dir: Path) -> None:
    file_path = backend_dir / "routes" / "storyboard.py"
    if not file_path.exists():
        return

    ai_parse_replacement = (
        "@router.post(\"/projects/{project_id}/storyboard/ai-parse\")\n"
        "async def ai_parse(project_id: str, req: ScriptParseRequest):\n"
        "    \"\"\"AI-powered script parsing using Gemma 3.\n"
        "\n"
        "    Falls back to regex parser if Gemma/full LTX checkpoint is unavailable.\n"
        "    Both paths include element matching.\n"
        "    \"\"\"\n"
        "    from managers.element_manager import element_manager\n"
        "    project_elements = element_manager.get_elements(project_id)\n"
        "\n"
        "    ckpt_full = os.path.join(config.LTX_DIR, \"models\", \"checkpoints\", \"ltx-2-19b-distilled.safetensors\")\n"
        "    ckpt_fp8 = os.path.join(config.LTX_DIR, \"models\", \"checkpoints\", \"ltx-2-19b-distilled-fp8.safetensors\")\n"
        "    if not (os.path.exists(ckpt_full) or os.path.exists(ckpt_fp8)):\n"
        "        logger.info(\"AI parse unavailable in Lite mode (missing full LTX checkpoint). Using fallback parser.\")\n"
        "        parsed_scenes = script_parser.parse_script(\n"
        "            req.script_text,\n"
        "            parse_mode=\"auto\",\n"
        "            elements=project_elements if project_elements else None,\n"
        "        )\n"
        "        return {\"scenes\": parsed_scenes, \"mode\": \"fallback\"}\n"
        "\n"
        "    try:\n"
        "        from model_engine import manager\n"
        "\n"
        "        pipeline = await manager.load_pipeline(\"ti2vid\")\n"
        "        text_encoder = pipeline.stage_1_model_ledger.text_encoder()\n"
        "\n"
        "        parsed_scenes = ai_parse_script(\n"
        "            text=req.script_text,\n"
        "            text_encoder=text_encoder,\n"
        "            seed=42,\n"
        "            project_elements=project_elements if project_elements else None,\n"
        "        )\n"
        "\n"
        "        del text_encoder\n"
        "        from ltx_pipelines.utils.helpers import cleanup_memory\n"
        "        cleanup_memory()\n"
        "\n"
        "        return {\"scenes\": parsed_scenes, \"mode\": \"ai\"}\n"
        "\n"
        "    except Exception as e:\n"
        "        logger.warning(f\"AI parse failed, falling back to regex: {e}\")\n"
        "        try:\n"
        "            parsed_scenes = script_parser.parse_script(\n"
        "                req.script_text,\n"
        "                parse_mode=\"auto\",\n"
        "                elements=project_elements if project_elements else None,\n"
        "            )\n"
        "            return {\"scenes\": parsed_scenes, \"mode\": \"fallback\"}\n"
        "        except Exception as e2:\n"
        "            logger.error(f\"Fallback parse also failed: {e2}\")\n"
        "            raise HTTPException(status_code=500, detail=str(e2))\n"
        "\n"
    )
    replace_region(
        file_path,
        "@router.post(\"/projects/{project_id}/storyboard/ai-parse\")\n",
        "@router.post(\"/projects/{project_id}/storyboard/match-elements\")\n",
        ai_parse_replacement,
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
    patch_element_manager(backend_dir)
    patch_flux_wrapper(backend_dir)
    patch_model_engine(backend_dir)
    patch_storyboard_routes(backend_dir)
    patch_server_startup(backend_dir)
    patch_sam_startup(project_root)
    patch_flux2_text_encoder_defaults(project_root)

    print("Applied MilimoVideo-Lite backend patches")


if __name__ == "__main__":
    main()
