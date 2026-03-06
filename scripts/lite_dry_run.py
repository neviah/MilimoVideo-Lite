from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MilimoVideo-Lite backend dry-run")
    parser.add_argument("project_root", help="Path to patched milimovideo clone")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    backend_dir = project_root / "backend"
    dry_file = backend_dir / "milimovideo_lite" / "dry_run.py"

    if not dry_file.exists():
        raise RuntimeError(f"Missing dry-run module: {dry_file}. Apply patches first.")

    cmd = [sys.executable, "-m", "milimovideo_lite.dry_run"]
    env = os.environ.copy()

    print(f"Running dry-run in {backend_dir}")
    subprocess.run(cmd, cwd=str(backend_dir), env=env, check=True)


if __name__ == "__main__":
    main()
