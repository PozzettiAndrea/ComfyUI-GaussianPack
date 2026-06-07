# SPDX-License-Identifier: GPL-3.0-or-later

"""PreviewGaussianDual — side-by-side or slider comparison of two gaussian splats."""

import os

from .common import get_default_extrinsics
from .preview_gaussian import _resolve_for_view, _count_gaussians


class PreviewGaussianDual:
    """Compare two Gaussian splats side-by-side or with a slider."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path_1": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to the first Gaussian Splatting PLY file",
                }),
                "ply_path_2": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to the second Gaussian Splatting PLY file",
                }),
                "layout": (["side_by_side", "slider"], {
                    "default": "side_by_side",
                    "tooltip": (
                        "side_by_side — two synchronized viewports. "
                        "slider — overlaid with a draggable divider."
                    ),
                }),
                "camera_mode": (["orbit", "spectate"], {
                    "default": "orbit",
                    "tooltip": (
                        "orbit — trackball turntable camera. "
                        "\n"
                        "spectate — WASD fly-cam."
                    ),
                }),
                "fov_degrees": ("FLOAT", {
                    "default": 70.0, "min": 5.0, "max": 170.0, "step": 1.0,
                }),
                "renderer": (["playcanvas", "spark"], {
                    "default": "playcanvas",
                    "tooltip": (
                        "playcanvas — PlayCanvas Engine v2.19 GSplat. "
                        "spark — Three.js + WebGL2, all formats."
                    ),
                }),
                "transport_format": (["ply", "spz"], {
                    "default": "ply",
                    "tooltip": (
                        "ply — lossless, larger. "
                        "spz — server transcodes to SPZ v2, ~9x smaller."
                    ),
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "viewer"

    def preview(self, ply_path_1, ply_path_2, layout, camera_mode,
                fov_degrees, renderer, transport_format="ply"):
        for i, p in enumerate([ply_path_1, ply_path_2], 1):
            if not p:
                return {"ui": {"error": [f"No PLY path provided for input {i}"]}}
            if not os.path.exists(p):
                return {"ui": {"error": [f"File not found (input {i}): {p}"]}}

        fn1, sub1, kind1 = _resolve_for_view(ply_path_1)
        fn2, sub2, kind2 = _resolve_for_view(ply_path_2)

        ui = {
            "ply_file_1": [fn1],
            "ply_type_1": [kind1],
            "ply_subfolder_1": [sub1],
            "file_size_mb_1": [round(os.path.getsize(ply_path_1) / (1024 * 1024), 2)],
            "num_gaussians_1": [_count_gaussians(ply_path_1)],
            # Freshness token: changes whenever the file is rewritten, so the
            # viewer's caches key on content identity, not just the filename.
            "mtime_1": [int(os.path.getmtime(ply_path_1))],
            "ply_file_2": [fn2],
            "ply_type_2": [kind2],
            "ply_subfolder_2": [sub2],
            "file_size_mb_2": [round(os.path.getsize(ply_path_2) / (1024 * 1024), 2)],
            "num_gaussians_2": [_count_gaussians(ply_path_2)],
            "mtime_2": [int(os.path.getmtime(ply_path_2))],
            "layout": [layout],
            "camera_mode": [camera_mode],
            "fov_degrees": [fov_degrees],
            "renderer": [renderer],
            "transport_format": [transport_format],
        }
        if camera_mode == "spectate":
            ui["mode"] = ["spectate"]
        return {"ui": ui}
