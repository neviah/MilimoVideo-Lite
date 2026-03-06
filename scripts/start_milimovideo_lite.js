const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..", "sandbox", "workspace", "milimovideo");
const backendDir = path.join(root, "backend");
const webDir = path.join(root, "web-app");
const logDir = path.join(__dirname, "..", "sandbox");

if (!fs.existsSync(path.join(backendDir, "server.py"))) {
  throw new Error("Missing backend/server.py. Install first.");
}

fs.mkdirSync(logDir, { recursive: true });

const py = process.platform === "win32"
  ? path.join(__dirname, "..", "sandbox", "venv", "Scripts", "python.exe")
  : path.join(__dirname, "..", "sandbox", "venv", "bin", "python");

const backendEnv = {
  ...process.env,
  MILIMO_LITE_SANDBOX_ROOT: path.join(__dirname, "..", "sandbox"),
  MILIMO_VRAM_MODE: process.env.MILIMO_VRAM_MODE || "auto",
  PYTHONUNBUFFERED: "1",
};

const backend = spawn(py, ["server.py"], {
  cwd: backendDir,
  env: backendEnv,
  stdio: ["inherit", "pipe", "pipe"],
});

const backendLog = fs.createWriteStream(path.join(logDir, "backend.log"), { flags: "a" });
backend.stdout.pipe(process.stdout);
backend.stderr.pipe(process.stderr);
backend.stdout.pipe(backendLog);
backend.stderr.pipe(backendLog);

const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
const frontend = spawn(npmCmd, ["run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"], {
  cwd: webDir,
  env: { ...process.env, BROWSER: "none" },
  stdio: ["inherit", "pipe", "pipe"],
});

const frontendLog = fs.createWriteStream(path.join(logDir, "frontend.log"), { flags: "a" });
frontend.stdout.pipe(process.stdout);
frontend.stderr.pipe(process.stderr);
frontend.stdout.pipe(frontendLog);
frontend.stderr.pipe(frontendLog);

const shutdown = () => {
  if (!backend.killed) {
    backend.kill("SIGTERM");
  }
  if (!frontend.killed) {
    frontend.kill("SIGTERM");
  }
};

backend.on("exit", (code) => {
  console.log(`backend exited with code ${code}`);
  shutdown();
  process.exit(code || 0);
});

frontend.on("exit", (code) => {
  console.log(`frontend exited with code ${code}`);
  shutdown();
  process.exit(code || 0);
});

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
