import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
	return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _has_nvidia_gpu() -> bool:
	nvidia_smi = shutil.which("nvidia-smi")
	if not nvidia_smi:
		return False
	result = _run([nvidia_smi, "-L"])
	return result.returncode == 0 and "GPU" in (result.stdout or "")


def _check_torch_cuda() -> Tuple[bool, str]:
	code = (
		"import torch\n"
		"print('torch=' + torch.__version__)\n"
		"print('cuda_available=' + str(torch.cuda.is_available()))\n"
		"print('cuda_version=' + str(getattr(torch.version, 'cuda', None)))\n"
	)
	result = _run([sys.executable, "-c", code])
	ok = result.returncode == 0 and "cuda_available=True" in result.stdout
	return ok, result.stdout.strip()


def _candidate_indexes() -> List[str]:
	raw = os.environ.get("MILIMO_TORCH_CUDA_INDEXES", "").strip()
	if raw:
		return [s.strip() for s in raw.split(",") if s.strip()]
	return [
		"https://download.pytorch.org/whl/cu128",
		"https://download.pytorch.org/whl/cu126",
		"https://download.pytorch.org/whl/cu124",
		"https://download.pytorch.org/whl/cu121",
	]


def _pip_install_torch(index_url: str) -> bool:
	cmd = [
		sys.executable,
		"-m",
		"pip",
		"install",
		"--upgrade",
		"--index-url",
		index_url,
		"torch",
		"torchvision",
		"torchaudio",
	]
	print(f"[ensure_torch_cuda] Trying torch install from {index_url}")
	result = _run(cmd)
	print(result.stdout)
	return result.returncode == 0


def _install_optional_accel(index_url: Optional[str]) -> None:
	if not index_url:
		return
	xformers_cmd = [
		sys.executable,
		"-m",
		"pip",
		"install",
		"--upgrade",
		"--extra-index-url",
		index_url,
		"xformers",
	]
	print("[ensure_torch_cuda] Installing xformers (optional)")
	x_result = _run(xformers_cmd)
	print(x_result.stdout)


def main() -> int:
	print("[ensure_torch_cuda] Detecting GPU and PyTorch runtime")

	pre_ok, pre_info = _check_torch_cuda()
	print("[ensure_torch_cuda] Current torch state:\n" + pre_info)

	if pre_ok:
		print("[ensure_torch_cuda] CUDA already available, skipping torch reinstall")
		return 0

	if not _has_nvidia_gpu():
		print("[ensure_torch_cuda] NVIDIA GPU not detected; keeping CPU runtime")
		return 0

	selected_index = None
	for index_url in _candidate_indexes():
		if not _pip_install_torch(index_url):
			continue
		ok, info = _check_torch_cuda()
		print("[ensure_torch_cuda] Torch state after install:\n" + info)
		if ok:
			selected_index = index_url
			print(f"[ensure_torch_cuda] CUDA enabled using {index_url}")
			break

	if selected_index:
		_install_optional_accel(selected_index)
		return 0

	print("[ensure_torch_cuda] WARNING: NVIDIA GPU detected but torch.cuda.is_available() is still False")
	print("[ensure_torch_cuda] Continuing with CPU mode; generation will be much slower")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
