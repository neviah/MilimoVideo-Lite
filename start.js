module.exports = {
  run: [
    {
      when: "{{!exists('sandbox/workspace/milimovideo/backend/server.py')}}",
      method: "shell.run",
      params: {
        message: [
          "echo MilimoVideo-Lite is not installed yet. Run install.js first."
        ]
      }
    },
    {
      when: "{{exists('sandbox/workspace/milimovideo/backend/server.py')}}",
      method: "local.set",
      params: {
        url: "http://127.0.0.1:5173/"
      }
    },
    {
      when: "{{exists('sandbox/workspace/milimovideo/backend/server.py')}}",
      method: "shell.run",
      params: {
        venv: "sandbox/venv",
        message: [
          "git pull --ff-only origin main",
          "git rev-parse --short HEAD",
          "npm --prefix sandbox/workspace/milimovideo/web-app install --include=dev --no-audit --no-fund",
          "node scripts/start_milimovideo_lite.js"
        ]
      }
    }
  ]
};
