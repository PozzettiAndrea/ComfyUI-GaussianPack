"""ComfyUI-GaussianPack Prestartup Script."""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent
ASSETS_DIR = SCRIPT_DIR / "assets"
INPUT_DIR = COMFYUI_DIR / "input"

# Copy bundled example assets (apple.ply, ...) into ComfyUI's input/
# so workflows that reference them by basename work on a fresh
# install. Non-clobbering — pre-existing files in input/ are left
# alone, so user edits survive restarts. Only top-level files
# (subdirectories like .ipynb_checkpoints/ are skipped).
if ASSETS_DIR.is_dir():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in ASSETS_DIR.iterdir():
        if src.is_file() and not (INPUT_DIR / src.name).exists():
            shutil.copy2(src, INPUT_DIR / src.name)
