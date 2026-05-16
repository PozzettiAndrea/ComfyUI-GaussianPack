# SPDX-License-Identifier: GPL-3.0-or-later

"""GaussianExport — re-export a 3DGS PLY in a chosen format.

Supported targets:
  - `ply`         passthrough copy with a chosen output filename.
  - `ply_no_sh`   strip f_rest_* (view-dependent SH coefficients) and
                  optionally normals; keep position + scale + rotation
                  + opacity + f_dc_* (DC term = base color). Result is
                  a valid 3DGS PLY at SH degree 0, ~3-5x smaller. Pure
                  Python via plyfile (already a runtime dep).
  - `spz`         convert to SPZ v2 (gzipped) via the vendored
                  web/server_tools/ply_to_spz.mjs (Node subprocess).
                  Output gets `.spz` extension, not `.ply`.

Output is the absolute path to the produced file (STRING), ready to
wire into PreviewGaussians or GaussianMerge.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .common import COMFYUI_OUTPUT_FOLDER

log = logging.getLogger("comfyui-gaussianpack")

_THIS_DIR = Path(__file__).resolve().parent
_SPZ_SCRIPT = _THIS_DIR.parent / "web" / "server_tools" / "ply_to_spz.mjs"


def _output_dir() -> Path:
    if COMFYUI_OUTPUT_FOLDER:
        return Path(COMFYUI_OUTPUT_FOLDER)
    return Path.cwd()


def _export_passthrough(src: Path, dst: Path) -> Path:
    shutil.copy2(src, dst)
    return dst


def _export_ply_no_sh(src: Path, dst: Path) -> Path:
    """Drop f_rest_* properties (view-dependent SH coefficients).
    Keeps DC term (f_dc_0/1/2) so base color survives."""
    # plyfile lives in [pypi-dependencies] / requirements.txt
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(str(src))
    vertex = ply["vertex"]

    keep_names = [
        name for name in vertex.data.dtype.names
        if not name.startswith("f_rest_")
    ]
    if len(keep_names) == len(vertex.data.dtype.names):
        log.info("GaussianExport ply_no_sh: source had no f_rest_* — copy-through.")
        return _export_passthrough(src, dst)

    new_dtype = np.dtype([(n, vertex.data.dtype.fields[n][0]) for n in keep_names])
    new_data = np.empty(len(vertex.data), dtype=new_dtype)
    for n in keep_names:
        new_data[n] = vertex.data[n]

    new_element = PlyElement.describe(new_data, "vertex")
    PlyData([new_element], text=ply.text, byte_order=ply.byte_order).write(str(dst))

    n_dropped = len(vertex.data.dtype.names) - len(keep_names)
    log.info(
        "GaussianExport ply_no_sh: dropped %d f_rest_* props -> %s "
        "(%.1f MB)", n_dropped, dst.name, dst.stat().st_size / (1024 * 1024),
    )
    return dst


def _export_spz(src: Path, dst: Path) -> Path:
    """Invoke the vendored ply_to_spz.mjs (same script spz_route.py
    uses for the in-flight viewer transcode)."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "GaussianExport spz: 'node' not found on PATH. Install Node.js "
            "to enable SPZ export."
        )
    if not _SPZ_SCRIPT.is_file():
        raise RuntimeError(f"GaussianExport spz: converter missing: {_SPZ_SCRIPT}")

    proc = subprocess.run(
        [node, str(_SPZ_SCRIPT), str(src), str(dst)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ply_to_spz.mjs exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[:500]}"
        )
    log.info(
        "GaussianExport spz: %s -> %s (%.1f MB -> %.1f MB, %.1fx)",
        src.name, dst.name,
        src.stat().st_size / (1024 * 1024),
        dst.stat().st_size / (1024 * 1024),
        src.stat().st_size / max(1, dst.stat().st_size),
    )
    return dst


_FORMAT_EXTS = {
    "ply":        ".ply",
    "ply_no_sh":  ".ply",
    "spz":        ".spz",
}


class GaussianExport:
    """Convert a 3DGS PLY into another flavor (PLY pass-through, SH-stripped PLY, or SPZ)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY to re-export",
                }),
                "format": (list(_FORMAT_EXTS.keys()), {
                    "default": "ply_no_sh",
                    "tooltip": (
                        "ply        — passthrough copy.\n"
                        "ply_no_sh  — drop view-dependent SH (f_rest_*); "
                                     "keep base color (f_dc_*). ~3-5x smaller "
                                     "but loses specular highlights.\n"
                        "spz        — SPZ v2 (gzipped). ~9x smaller; SH "
                                     "coefficients quantize to 4 bits each. "
                                     "Requires Node.js on PATH."
                    ),
                }),
                "output_filename": ("STRING", {
                    "default": "exported",
                    "tooltip": "Basename for the output file (no extension).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out_path",)
    FUNCTION = "export"
    CATEGORY = "viewer"

    def export(self, ply_path: str, format: str, output_filename: str):
        if not ply_path or not os.path.exists(ply_path):
            raise FileNotFoundError(f"GaussianExport: input PLY not found: {ply_path!r}")
        if format not in _FORMAT_EXTS:
            raise ValueError(f"GaussianExport: unknown format {format!r}")

        src = Path(ply_path).resolve()
        out_dir = _output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{output_filename}{_FORMAT_EXTS[format]}"

        log.info("GaussianExport: %s -> %s (format=%s)", src, dst, format)

        if format == "ply":
            _export_passthrough(src, dst)
        elif format == "ply_no_sh":
            _export_ply_no_sh(src, dst)
        elif format == "spz":
            _export_spz(src, dst)

        return (str(dst),)
