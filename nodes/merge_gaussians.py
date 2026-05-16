# SPDX-License-Identifier: GPL-3.0-or-later

"""GaussianMerge — decimate a 3DGS PLY to a target splat count via
NanoGS pairwise merging (MPMM). Output is cached next to ComfyUI's
output directory; re-running with the same (input mtime, target_count,
opacity_threshold, k) hits the cache.

Algorithm: kNN graph over splat centers, mass-preserving moment matching
on each candidate pair, greedy edge selection by merge cost. CPU only —
the upstream NanoGS paper notes a GPU port as future work. ~10–30s for
1M -> 200K splats on a modern CPU. Same algorithm
`@playcanvas/splat-transform --decimate` uses, just in pure Python.
"""

import logging
import os
from pathlib import Path

from .common import COMFYUI_OUTPUT_FOLDER

log = logging.getLogger("comfyui-gaussianpack")


def _count_gaussians(path: str) -> int:
    """Header-only PLY vertex count. Returns 0 if unparseable."""
    try:
        with open(path, "rb") as f:
            buf = f.read(8192).decode("latin-1", errors="replace")
        for line in buf.split("\n"):
            if line.startswith("element vertex "):
                return int(line.split()[2])
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _output_dir() -> Path:
    """ComfyUI's output dir, or cwd as a last-resort fallback."""
    if COMFYUI_OUTPUT_FOLDER:
        return Path(COMFYUI_OUTPUT_FOLDER)
    return Path.cwd()


class GaussianMerge:
    """Pairwise-merge a 3DGS PLY down to a target Gaussian count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file",
                }),
                "target_count": ("INT", {
                    "default": 200_000, "min": 1_000, "max": 10_000_000, "step": 1_000,
                    "tooltip": (
                        "Approximate output Gaussian count. NanoGS works "
                        "in ratio terms — target / input_count is passed "
                        "as `--ratio`. If target >= input count, the input "
                        "is passed through unchanged."
                    ),
                }),
                "output_filename": ("STRING", {
                    "default": "merged",
                    "tooltip": "Basename for the output PLY (no extension).",
                }),
                "opacity_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Splats with opacity below min(threshold, "
                        "median(opacity)) are pruned before merging."
                    ),
                }),
                "k": ("INT", {
                    "default": 16, "min": 4, "max": 64,
                    "tooltip": (
                        "k nearest neighbors per splat that the merge graph "
                        "considers as candidate pairs. Higher k -> slightly "
                        "better quality (more options for the greedy "
                        "selector) but more RAM + slower (O(N*k) edges).\n"
                        "16 is the upstream sweet spot. 8 = fast draft. "
                        "24-32 helps on specular-heavy scenes; 32+ rarely "
                        "pays off."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "merge"
    CATEGORY = "viewer"

    def merge(
        self,
        ply_path: str,
        target_count: int,
        output_filename: str,
        opacity_threshold: float,
        k: int,
    ):
        if not ply_path or not os.path.exists(ply_path):
            raise FileNotFoundError(f"GaussianMerge: input PLY not found: {ply_path!r}")

        n_in = _count_gaussians(ply_path)
        if n_in <= 0:
            raise ValueError(f"GaussianMerge: couldn't parse PLY vertex count from {ply_path}")

        out_dir = _output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{output_filename}.ply"

        # Pass-through if target >= input. NanoGS demands ratio in the
        # OPEN interval (0, 1), so anything else has to short-circuit.
        if target_count >= n_in:
            log.info(
                "GaussianMerge: target %d >= input %d; passing through %s",
                target_count, n_in, ply_path,
            )
            return (ply_path,)

        ratio = max(1e-6, min(0.999999, target_count / n_in))

        # mtime-keyed cache: skip the merge if the output already matches
        # this (input mtime, target_count, opacity_threshold, k) tuple.
        cache_tag = f"{int(os.path.getmtime(ply_path))}.{target_count}.{opacity_threshold:.4f}.{k}"
        cache_sidecar = out_path.with_suffix(".ply.cachekey")
        if (
            out_path.is_file()
            and cache_sidecar.is_file()
            and cache_sidecar.read_text().strip() == cache_tag
        ):
            log.info("GaussianMerge: cache hit (%s); skipping merge", cache_tag)
            return (str(out_path),)

        # Lazy imports — NanoGS pulls scipy.cKDTree on first call; we
        # don't want to pay that on ComfyUI startup.
        from nanogs.simplification import simplify
        from nanogs.utils.params import RunParams, CostParams

        log.info(
            "GaussianMerge: %d -> %d splats (ratio=%.4f) via NanoGS MPMM on %s",
            n_in, target_count, ratio, ply_path,
        )
        simplify(
            ply_path,
            str(out_path),
            RunParams(ratio=ratio, merge_cap=0.5, k=k, opacity_threshold=opacity_threshold),
            CostParams(lam_geo=1.0, lam_sh=1.0),
        )

        cache_sidecar.write_text(cache_tag)
        n_out = _count_gaussians(str(out_path))
        log.info(
            "GaussianMerge: wrote %s (%d splats, %.1f MB)",
            out_path, n_out, out_path.stat().st_size / (1024 * 1024),
        )
        return (str(out_path),)
