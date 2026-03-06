const fs = require("fs");
const path = require("path");

const cfgPath = path.join(__dirname, "..", "sandbox", "config.json");
const cfg = {
  appName: "MilimoVideo-Lite",
  vramMode: "auto",
  backendPort: 8000,
  frontendPort: 5173,
  samPort: 8001,
  modelRoot: "backend/models"
};

fs.mkdirSync(path.dirname(cfgPath), { recursive: true });
fs.writeFileSync(cfgPath, JSON.stringify(cfg, null, 2), "utf8");
console.log(`Wrote ${cfgPath}`);
