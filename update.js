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
          "python scripts/ensure_torch_cuda.py",
          "python scripts/apply_backend_patch.py sandbox/workspace/milimovideo",
          "echo MilimoVideo-Lite updated"
        ]
      }
    }
  ]
};
