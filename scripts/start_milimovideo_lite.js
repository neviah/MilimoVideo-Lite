const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..", "sandbox", "workspace", "milimovideo");
const backendDir = path.join(root, "backend");
const webDir = path.join(root, "web-app");
const logDir = path.join(__dirname, "..", "sandbox");

if (!fs.existsSync(path.join(backendDir, "server.py"))) {
  throw new Error("Missing backend/server.py. Install first.");
}

if (!fs.existsSync(webDir)) {
  throw new Error(`Missing web-app folder at ${webDir}. Re-run install.js.`);
}

fs.mkdirSync(logDir, { recursive: true });

function getViteCmdPath() {
  return path.join(webDir, "node_modules", ".bin", process.platform === "win32" ? "vite.cmd" : "vite");
}

function ensureFrontendDeps() {
  const viteCmd = getViteCmdPath();
  if (fs.existsSync(viteCmd)) {
    return viteCmd;
  }

  console.log("Installing frontend dependencies (missing vite binary)...");
  const result = process.platform === "win32"
    ? spawnSync("cmd.exe", ["/d", "/s", "/c", "npm install --include=dev --no-audit --no-fund"], {
        cwd: webDir,
        env: process.env,
        stdio: "inherit",
        windowsHide: true,
      })
    : spawnSync("npm", ["install", "--include=dev", "--no-audit", "--no-fund"], {
        cwd: webDir,
        env: process.env,
        stdio: "inherit",
      });

  if (result.status !== 0) {
    throw new Error(`npm install failed with exit code ${result.status}`);
  }

  if (!fs.existsSync(viteCmd)) {
    throw new Error("Frontend dependencies installed but vite binary is still missing.");
  }

  return viteCmd;
}

const viteCmdPath = ensureFrontendDeps();

const pyCandidate = process.platform === "win32"
  ? path.join(__dirname, "..", "sandbox", "venv", "Scripts", "python.exe")
  : path.join(__dirname, "..", "sandbox", "venv", "bin", "python");
const py = fs.existsSync(pyCandidate) ? pyCandidate : "python";

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
  windowsHide: true,
});

const backendLog = fs.createWriteStream(path.join(logDir, "backend.log"), { flags: "a" });
backend.stdout.pipe(process.stdout);
backend.stderr.pipe(process.stderr);
backend.stdout.pipe(backendLog);
backend.stderr.pipe(backendLog);

const frontendEnv = { ...process.env, BROWSER: "none" };
const frontend = process.platform === "win32"
  ? spawn(
      "cmd.exe",
      ["/d", "/s", "/c", "npm exec vite -- --host 127.0.0.1 --port 5173"],
      {
        cwd: webDir,
        env: frontendEnv,
        stdio: ["inherit", "pipe", "pipe"],
        windowsHide: true,
      }
    )
  : spawn(
      "npm",
      ["exec", "vite", "--", "--host", "127.0.0.1", "--port", "5173"],
      {
        cwd: webDir,
        env: frontendEnv,
        stdio: ["inherit", "pipe", "pipe"],
      }
    );

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

backend.on("error", (err) => {
  console.error("Failed to spawn backend process:", err);
  shutdown();
  process.exit(1);
});

frontend.on("exit", (code) => {
  console.log(`frontend exited with code ${code}`);
  shutdown();
  process.exit(code || 0);
});

frontend.on("error", (err) => {
  console.error("Failed to spawn frontend process:", err);
  shutdown();
  process.exit(1);
});

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
