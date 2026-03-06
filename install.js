module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "mkdir sandbox 2>nul",
          "mkdir sandbox\\workspace 2>nul"
        ]
      }
    },
    {
      when: "{{!exists('sandbox/venv/Scripts/python.exe')}}",
      method: "shell.run",
      params: {
        message: [
          "python -m venv sandbox/venv"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "sandbox/venv",
        message: [
          "python -m pip install --upgrade pip",
          "python -m pip install wheel setuptools",
          "python -m pip install huggingface_hub hf_transfer requests",
          "if exist sandbox\\workspace\\milimovideo rmdir /s /q sandbox\\workspace\\milimovideo",
          "git clone https://github.com/mainza-ai/milimovideo sandbox/workspace/milimovideo",
          "python -m pip install -e sandbox/workspace/milimovideo/LTX-2/packages/ltx-core",
          "python -m pip install -e sandbox/workspace/milimovideo/LTX-2/packages/ltx-pipelines",
          "python -m pip install -e sandbox/workspace/milimovideo/flux2",
          "python -m pip install -r sandbox/workspace/milimovideo/backend/requirements.txt",
          "python scripts/ensure_torch_cuda.py",
          "python -m pip install bitsandbytes || echo Optional bitsandbytes install skipped",
          "python -m pip install flash-attn || echo Optional flash-attn install skipped",
          "python -m pip install llama-cpp-python || echo Optional llama-cpp-python install skipped",
          "python -m pip install unsloth || echo Optional unsloth install skipped",
          "python scripts/apply_backend_patch.py sandbox/workspace/milimovideo",
          "npm --prefix sandbox/workspace/milimovideo/web-app install --include=dev",
          "node scripts/create_sandbox_config.js",
          "echo MilimoVideo-Lite install completed"
        ]
      }
    }
  ]
};
