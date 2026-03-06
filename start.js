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
          "node scripts/start_milimovideo_lite.js"
        ]
      }
    }
  ]
};
