# SPDX-License-Identifier: GPL-3.0-or-later

"""GaussiansFromPointCloud - convert a plain point-cloud PLY into a proper
3DGS PLY with sensible per-point initialization.

Why this exists: many upstream pipelines (PanoramaBuildPointCloud + GeomPack
SaveMesh, raw colmap, depth-to-pcd, etc.) produce a points-only PLY with
just `x/y/z` (+ optional `r/g/b`). The 3DGS trainers and viewers expect a
PLY with `rot_*`, `scale_*`, `opacity`, `f_dc_*` fields too. Without those,
loaders either reject the PLY or substitute degenerate defaults (e.g.
`_load_3dgs_ply` writes `log(1e-3)` for scales -> every gaussian is a
1 mm sphere regardless of local density). Training from that state takes
forever to converge.

This node initializes scales the same way upstream 3DGS does
(`hyworld2/worldgen/world_gs_trainer.py:405-408` sfm path):

    dist2_avg = (knn(points, k+1)[:, 1:] ** 2).mean(dim=-1)
    dist_avg  = sqrt(dist2_avg)
    scales    = log(dist_avg * init_scale)         # [N, 3]

Dense regions get small gaussians, sparse regions get large ones. The
input PLY's colors (if present) flow into sh0 via the standard inverse
(`(rgb - 0.5) / C0`). Identity quats, configurable initial opacity.
"""

import logging
import math
import os
from pathlib import Path

import numpy as np
import torch

from .common import COMFYUI_OUTPUT_FOLDER
from .preview_gaussian_camera import _load_3dgs_ply, _C0

log = logging.getLogger("comfyui-gaussianpack")


def _output_dir() -> Path:
    if COMFYUI_OUTPUT_FOLDER:
        return Path(COMFYUI_OUTPUT_FOLDER)
    return Path.cwd()


def _knn_derived_log_scales(
    pts: np.ndarray, k: int, init_scale: float
) -> np.ndarray:
    """Per-point log-scale = log(mean_knn_distance x init_scale).

    Uses sklearn's KDTree (already a runtime dep via torch's deps); for
    7M points it's ~3-10s, ~120 MB. Returns shape [N, 3] (same scale on
    every axis - gaussians initialized as isotropic spheres; training
    can deform them).
    """
    from sklearn.neighbors import NearestNeighbors

    N = pts.shape[0]
    k_query = min(k + 1, N)  # +1 because the first neighbor of each point is itself
    if k_query < 2:
        # Pathological: <2 points. Fall back to a fixed tiny scale.
        return np.full((N, 3), math.log(1e-3), dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=k_query, algorithm="auto").fit(pts)
    dists, _ = nn.kneighbors(pts)               # [N, k_query], col 0 = self
    # Squared-distance mean, then sqrt - matches upstream.
    dist2_avg = (dists[:, 1:] ** 2).mean(axis=1)
    dist_avg = np.sqrt(np.maximum(dist2_avg, 1e-30))
    scales_log = np.log(dist_avg * float(init_scale) + 1e-30).astype(np.float32)
    return np.tile(scales_log[:, None], (1, 3))


def _try_extract_rgb_sh0(ply_path: str, N_expected: int) -> np.ndarray | None:
    """Open the PLY and return [N, 1, 3] sh0 from `red/green/blue` fields,
    or None if those fields aren't present. Detects uint8 vs float by
    max value (uint8 typically has max > 1.5; float [0, 1] never does).

    Used to recover colors from trimesh-style PLYs (PanoramaBuildPointCloud
    -> GeomPackSaveMesh layout) since the shared `_load_3dgs_ply` parser
    only understands the 3DGS `f_dc_*` color layout and otherwise falls
    back to a constant white default.

    Formula: rgb_lin in [0, 1] -> sh0 = (rgb - 0.5) / C0 (inverse of the
    `rgb_to_sh` round-trip used everywhere else in the pipeline).
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        return None
    v = ply["vertex"].data
    names = set(v.dtype.names)
    if not all(k in names for k in ("red", "green", "blue")):
        return None
    if len(v) != N_expected:
        return None
    r = np.asarray(v["red"]).astype(np.float32)
    g = np.asarray(v["green"]).astype(np.float32)
    b = np.asarray(v["blue"]).astype(np.float32)
    if float(max(r.max(), g.max(), b.max())) > 1.5:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    rgb_lin = np.stack([r, g, b], axis=1).clip(0.0, 1.0)
    sh0 = ((rgb_lin - 0.5) / _C0).astype(np.float32)[:, None, :]
    return sh0


def _write_3dgs_ply(
    out_path: Path,
    means: np.ndarray,           # [N, 3] float32 world positions
    scales: np.ndarray,          # [N, 3] float32 log space
    quats: np.ndarray,           # [N, 4] float32 wxyz
    opacities: np.ndarray,       # [N]    float32 logit space
    sh0: np.ndarray,             # [N, 1, 3] float32 SH band-0
    shN: np.ndarray | None = None,  # [N, K_AC, 3] float32 or None
) -> None:
    """Write the graphdeco-inria 3DGS PLY layout (SuperSplat / PlayCanvas /
    Antimatter15 / gsplat all interop on this).

    Fields written:
      x, y, z                           position
      nx, ny, nz                        normal placeholders (0.0)
      f_dc_0..2                         sh0 (SH band-0)
      f_rest_0..(3*K_AC - 1)            shN (when present)
      opacity                           logit-space alpha
      scale_0..2                        log-space scales
      rot_0..3                          wxyz quat
    """
    from plyfile import PlyData, PlyElement

    N = means.shape[0]
    assert scales.shape == (N, 3)
    assert quats.shape == (N, 4)
    assert opacities.shape == (N,)
    assert sh0.shape == (N, 1, 3)

    # Build dtype.
    dtypes = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    K_AC = 0
    if shN is not None and shN.shape[1] > 0:
        K_AC = shN.shape[1]
        # 3DGS interleaves: first K_AC for R, then K_AC for G, then K_AC for B.
        for c in range(3 * K_AC):
            dtypes.append((f"f_rest_{c}", "f4"))
    dtypes.append(("opacity", "f4"))
    dtypes.extend([(f"scale_{i}", "f4") for i in range(3)])
    dtypes.extend([(f"rot_{i}", "f4") for i in range(4)])

    arr = np.zeros(N, dtype=dtypes)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    # nx/ny/nz stay zero.
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh0[:, 0, 0], sh0[:, 0, 1], sh0[:, 0, 2]
    if shN is not None and K_AC > 0:
        # Reshape [N, K_AC, 3] -> [N, 3, K_AC] -> [N, 3*K_AC] (R-block, G-block, B-block).
        flat = shN.transpose(0, 2, 1).reshape(N, 3 * K_AC)
        for c in range(3 * K_AC):
            arr[f"f_rest_{c}"] = flat[:, c]
    arr["opacity"] = opacities
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out_path))


class GaussiansFromPointCloud:
    """Convert a point-cloud PLY into a 3DGS PLY with knn-derived scales."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Path to an input PLY. Can be a plain point cloud "
                        "(x/y/z + optional r/g/b) or an existing 3DGS PLY "
                        "(in which case all fields are read; missing ones "
                        "get sensible defaults). knn-derived scales "
                        "OVERRIDE whatever scales the input had."
                    ),
                }),
                "output_filename": ("STRING", {
                    "default": "gaussians_from_pcd",
                    "tooltip": "Basename for the output PLY (no extension).",
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Directory to write the output PLY. Leave blank "
                        "to use ComfyUI's output folder."
                    ),
                }),
                "init_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": (
                        "Multiplier on the knn distance when computing "
                        "initial scales. 1.0 = exactly the mean distance "
                        "to k nearest neighbors. <1 = smaller gaussians "
                        "(sharper but possibly under-covered). >1 = "
                        "larger (smoother but blurrier)."
                    ),
                }),
                "init_opacity": ("FLOAT", {
                    "default": 0.1, "min": 0.001, "max": 0.999, "step": 0.001,
                    "tooltip": (
                        "Sigmoid-space initial alpha per gaussian. The "
                        "node applies logit(init_opacity) and writes that "
                        "to `opacity`. 0.1 matches upstream 3DGS sfm-init."
                    ),
                }),
                "knn_k": ("INT", {
                    "default": 3, "min": 1, "max": 10,
                    "tooltip": (
                        "k for the knn lookup. 3 matches upstream 3DGS "
                        "(mean distance to the 3 nearest neighbors after "
                        "skipping self). Larger k = smoother scale field."
                    ),
                }),
                "subsample_max": ("INT", {
                    "default": 0, "min": 0, "max": 50_000_000,
                    "tooltip": (
                        "Optional random cap on the gaussian count. 0 = "
                        "no cap (write every input point). Set a positive "
                        "value (e.g. 2_000_000) if a downstream node "
                        "OOMs at huge counts. Deterministic seed = N_in."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "convert"
    CATEGORY = "GaussianPack"

    def convert(
        self,
        ply_path: str,
        output_filename: str,
        output_dir: str = "",
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        knn_k: int = 3,
        subsample_max: int = 0,
    ):
        if not ply_path or not os.path.exists(ply_path):
            raise FileNotFoundError(
                f"GaussiansFromPointCloud: input PLY not found: {ply_path!r}"
            )

        # ---- Read input PLY (handles missing 3DGS fields with defaults). ----
        splat = _load_3dgs_ply(ply_path)
        means = splat["means"].numpy().astype(np.float32)
        N_in = means.shape[0]
        log.info(
            "[GaussiansFromPointCloud] read %d points from %s",
            N_in, Path(ply_path).name,
        )

        # ---- Override sh0 with `red/green/blue` colors if present. ----
        # `_load_3dgs_ply` only reads the 3DGS `f_dc_*` layout and falls
        # back to a constant white default. Trimesh-style PLYs (the kind
        # produced by PanoramaBuildPointCloud -> GeomPackSaveMesh) store
        # color as `red/green/blue/alpha` uint8 - we recover that here.
        rgb_sh0 = _try_extract_rgb_sh0(ply_path, N_in)
        if rgb_sh0 is not None:
            splat["sh0"] = torch.from_numpy(rgb_sh0)
            log.info(
                "[GaussiansFromPointCloud] sh0 initialized from "
                "red/green/blue (N=%d, sh0 mean=%.4f)",
                rgb_sh0.shape[0], float(rgb_sh0.mean()),
            )
        else:
            log.info(
                "[GaussiansFromPointCloud] no red/green/blue in PLY; "
                "sh0 stays at _load_3dgs_ply value (white default if no f_dc_*)"
            )

        # ---- Optional subsample. ----
        if subsample_max > 0 and N_in > subsample_max:
            g = torch.Generator(device="cpu").manual_seed(N_in)
            keep = torch.randperm(N_in, generator=g)[:subsample_max].numpy()
            means = means[keep]
            sh0 = splat["sh0"].numpy().astype(np.float32)[keep]
            shN_t = splat["shN"]
            shN = shN_t.numpy().astype(np.float32)[keep] if shN_t.shape[1] > 0 else None
            N = subsample_max
            log.info(
                "[GaussiansFromPointCloud] subsampled %d -> %d points "
                "(deterministic seed=%d)",
                N_in, N, N_in,
            )
        else:
            sh0 = splat["sh0"].numpy().astype(np.float32)
            shN_t = splat["shN"]
            shN = shN_t.numpy().astype(np.float32) if shN_t.shape[1] > 0 else None
            N = N_in

        # ---- Compute knn-derived per-point scales. ----
        scales = _knn_derived_log_scales(means, int(knn_k), float(init_scale))
        sl_min = float(scales.min())
        sl_max = float(scales.max())
        log.info(
            "[GaussiansFromPointCloud] knn(k=%d) scale init: log range "
            "[%.2f, %.2f] -> linear [%.2e, %.2e] (init_scale=%.3f)",
            int(knn_k), sl_min, sl_max, math.exp(sl_min), math.exp(sl_max),
            float(init_scale),
        )

        # ---- Initial opacities (logit space). ----
        # logit(p) = log(p / (1 - p)). Clamp p away from 0 / 1.
        p = max(min(float(init_opacity), 0.9999), 0.0001)
        opacity_logit = math.log(p / (1.0 - p))
        opacities = np.full((N,), opacity_logit, dtype=np.float32)

        # ---- Identity quats (point clouds have no orientation; trainer can deform). ----
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0  # w = 1, xyz = 0

        # ---- Resolve output path. ----
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = _output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{output_filename}.ply"

        # ---- Write the 3DGS PLY. ----
        _write_3dgs_ply(
            out_path,
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
        )
        size_mb = out_path.stat().st_size / 1e6
        log.info(
            "[GaussiansFromPointCloud] wrote 3DGS PLY: %s (%d gaussians, %.1f MB)",
            out_path, N, size_mb,
        )
        return (str(out_path),)
