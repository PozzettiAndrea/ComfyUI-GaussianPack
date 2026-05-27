# SPDX-License-Identifier: GPL-3.0-or-later

"""TransformGaussian — rotate a 3DGS PLY around X/Y/Z axes.

A gaussian splat has positions AND per-splat orientations (quaternions)
AND view-dependent spherical-harmonic coefficients, so rotating the
scene means:
  - rotate the (x, y, z) positions
  - compose the per-splat quaternion (rot_0..rot_3) with the global rotation
  - rotate (nx, ny, nz) normals if the PLY carries them
  - rotate the SH AC coefficients (f_rest_*) in the SH basis via
    Wigner D-matrices computed by Ivanic-Ruedenberg recursion
    (see _sh_rotation.py). DC term (f_dc_*) is rotation-invariant.

Result is a new PLY written to ComfyUI's output/ with a name encoding
the rotation. Output is the absolute path STRING, ready to pipe into
PreviewGaussians / GaussianMerge / GaussianExport.
"""

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .common import COMFYUI_OUTPUT_FOLDER
from ._sh_rotation import rotate_sh_ac, sh_degree_from_n_ac

log = logging.getLogger("comfyui-gaussianpack")


def _output_dir() -> Path:
    """ComfyUI's output dir, or cwd as a fallback."""
    if COMFYUI_OUTPUT_FOLDER:
        return Path(COMFYUI_OUTPUT_FOLDER)
    return Path.cwd()


def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Right-handed XYZ Euler rotation, applied in the order X → Y → Z.

    Matches geompack's TransformMesh._rotate so users get the same
    visual result for the same angle inputs across mesh/gaussian.
    """
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z).

    Numerically-stable branch by largest diagonal (Shepperd's method).
    3DGS PLYs store rotations as (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    per the Inria convention.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def _quat_mul_wxyz(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q = qa * qb. Both in (w, x, y, z).

    Supports broadcasting: qa shape (4,) and qb shape (N, 4) -> (N, 4).
    """
    aw, ax, ay, az = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    bw, bx, by, bz = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1).astype(np.float32)


def _make_output_path(src: Path, rx: float, ry: float, rz: float) -> Path:
    """Stable hashed filename so re-runs with same inputs don't collide
    and accidental same-second runs don't overwrite mid-flight."""
    stem = src.stem
    h = hashlib.sha1(f"{stem}|{rx}|{ry}|{rz}".encode()).hexdigest()[:8]
    rx_tag = f"{rx:+.0f}".replace("+", "p").replace("-", "m")
    ry_tag = f"{ry:+.0f}".replace("+", "p").replace("-", "m")
    rz_tag = f"{rz:+.0f}".replace("+", "p").replace("-", "m")
    return _output_dir() / f"{stem}_rot_{rx_tag}_{ry_tag}_{rz_tag}_{h}.ply"


class TransformGaussian:
    """Rotate a Gaussian Splatting PLY around X/Y/Z axes (degrees).

    Positions and per-splat quaternions are rotated together; normals
    too if present. SH AC coefficients are passed through unchanged
    (proper Wigner-D rotation is a follow-up).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a 3D Gaussian Splatting PLY file.",
                }),
                "rotate_x": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Rotation around X axis (degrees).",
                }),
                "rotate_y": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Rotation around Y axis (degrees).",
                }),
                "rotate_z": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Rotation around Z axis (degrees).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "transform"
    CATEGORY = "viewer"

    def transform(self, ply_path: str, rotate_x: float, rotate_y: float, rotate_z: float):
        if not ply_path:
            raise ValueError("TransformGaussian: no PLY path provided")
        src = Path(ply_path)
        if not src.is_file():
            raise FileNotFoundError(f"TransformGaussian: file not found: {src}")

        # No rotation -> just pass through the same path (no work).
        if abs(rotate_x) + abs(rotate_y) + abs(rotate_z) < 1e-6:
            log.info("TransformGaussian: all rotations 0 -> passthrough %s", src)
            return (str(src),)

        dst = _make_output_path(src, rotate_x, rotate_y, rotate_z)

        # Idempotent cache: same input + same angles -> reuse if it
        # already exists and source hasn't changed since.
        if dst.is_file() and dst.stat().st_mtime > src.stat().st_mtime:
            log.info("TransformGaussian: cached -> %s", dst)
            return (str(dst),)

        ply = PlyData.read(str(src))
        vertex = ply["vertex"]
        data = vertex.data
        N = len(data)
        log.info("TransformGaussian: %d splats, rotation X=%.1f Y=%.1f Z=%.1f",
                 N, rotate_x, rotate_y, rotate_z)

        R = _rotation_matrix_xyz(rotate_x, rotate_y, rotate_z)
        q_global_wxyz = _matrix_to_quat_wxyz(R)

        # In-place on a freshly-allocated numpy copy. The dtype must
        # match the source so we keep PLY round-trip bit-identical for
        # all the fields we DON'T touch (opacity, f_dc_*, f_rest_*, etc).
        new_data = data.copy()

        # 1) Positions: standard rotation.
        if all(k in data.dtype.names for k in ("x", "y", "z")):
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
            xyz_rot = xyz @ R.T   # (N, 3) @ (3, 3).T  = (N, 3)
            new_data["x"] = xyz_rot[:, 0]
            new_data["y"] = xyz_rot[:, 1]
            new_data["z"] = xyz_rot[:, 2]

        # 2) Per-splat quaternions: q_new = q_global * q_per_splat.
        # PLY field order is rot_0=w, rot_1=x, rot_2=y, rot_3=z.
        if all(k in data.dtype.names for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
            q_per = np.stack([
                data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"],
            ], axis=1).astype(np.float32)
            # 3DGS PLYs store UNNORMALIZED quats (the renderer normalizes
            # at use-time). Composition still works on un-normalized
            # inputs — Hamilton product is bilinear, so the result
            # encodes the same rotation; renderer normalizes later.
            q_new = _quat_mul_wxyz(q_global_wxyz, q_per)
            new_data["rot_0"] = q_new[:, 0]
            new_data["rot_1"] = q_new[:, 1]
            new_data["rot_2"] = q_new[:, 2]
            new_data["rot_3"] = q_new[:, 3]

        # 3) Normals (if present — many 3DGS PLYs ship zeroed normals).
        if all(k in data.dtype.names for k in ("nx", "ny", "nz")):
            n = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float32)
            n_rot = n @ R.T
            new_data["nx"] = n_rot[:, 0]
            new_data["ny"] = n_rot[:, 1]
            new_data["nz"] = n_rot[:, 2]

        # 4) Spherical-harmonic AC coefficients via Wigner-D rotation.
        # 3DGS PLY layout: f_rest_0..f_rest_{3*K_AC - 1} where the
        # first K_AC = (sh_deg+1)^2 - 1 fields are channel R, next K_AC
        # are G, last K_AC are B. Within each channel SH coefficients
        # are stored in band order: l=1 (3 coeffs), l=2 (5), l=3 (7).
        # DC fields f_dc_0..2 are rotation-invariant and not touched.
        f_rest_names = sorted(
            (n for n in data.dtype.names if re.fullmatch(r"f_rest_\d+", n)),
            key=lambda n: int(n.split("_")[-1]),
        )
        if f_rest_names:
            total = len(f_rest_names)
            if total % 3 != 0:
                log.warning(
                    "TransformGaussian: %d f_rest_* fields not divisible by 3 "
                    "(expected channel-major layout); skipping SH rotation",
                    total,
                )
            else:
                K_AC = total // 3
                try:
                    sh_deg = sh_degree_from_n_ac(K_AC)
                except ValueError as e:
                    log.warning(
                        "TransformGaussian: non-standard SH AC count (%d per "
                        "channel); skipping SH rotation. %s", K_AC, e,
                    )
                    sh_deg = 0
                if sh_deg >= 1:
                    f_rest = np.stack(
                        [data[n] for n in f_rest_names], axis=1,
                    ).astype(np.float32).reshape(N, 3, K_AC)
                    f_rest_rot = rotate_sh_ac(f_rest, R)
                    flat = f_rest_rot.reshape(N, 3 * K_AC)
                    for i, name in enumerate(f_rest_names):
                        new_data[name] = flat[:, i]
                    log.info(
                        "TransformGaussian: rotated SH AC (sh_degree=%d, "
                        "%d coeffs/channel × 3 channels) via Wigner-D",
                        sh_deg, K_AC,
                    )

        new_element = PlyElement.describe(new_data, "vertex")
        # Preserve any other elements (some PLYs carry a "header" or
        # custom blocks) and the byte order / ascii flag of the source.
        other_elements = [e for e in ply.elements if e.name != "vertex"]
        PlyData(
            [new_element, *other_elements],
            text=ply.text, byte_order=ply.byte_order,
        ).write(str(dst))

        size_mb = dst.stat().st_size / (1024 * 1024)
        log.info("TransformGaussian: wrote %s (%.1f MB)", dst.name, size_mb)
        return (str(dst),)
