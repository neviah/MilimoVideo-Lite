module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "sandbox/venv",
        message: [
          "git pull --ff-only",
          "if not exist sandbox\\workspace\\milimovideo echo MilimoVideo not installed. Run install first. && exit /b 1",
          "git -C sandbox/workspace/milimovideo pull --ff-only",
          "python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || echo CUDA torch update skipped",
          "python -c \"import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())\"",
          "python scripts/apply_backend_patch.py sandbox/workspace/milimovideo",
          "echo MilimoVideo-Lite updated"
        ]
      }
    }
  ]
};
