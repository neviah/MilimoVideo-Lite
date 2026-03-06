module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "sandbox/venv",
        message: [
          "if not exist sandbox\\workspace\\milimovideo echo MilimoVideo not installed. Run install first. && exit /b 1",
          "cd sandbox/workspace/milimovideo",
          "git pull",
          "cd ..\\..\\..",
          "python scripts/apply_backend_patch.py sandbox/workspace/milimovideo",
          "echo MilimoVideo-Lite updated"
        ]
      }
    }
  ]
};
