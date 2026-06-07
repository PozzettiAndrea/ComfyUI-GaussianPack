# SPDX-License-Identifier: GPL-3.0-or-later

"""Dual Gaussian splat viewer — side-by-side or slider comparison."""

import os

from .common import (
    get_default_extrinsics,
    get_default_intrinsics,
)
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
                        "side_by_side — two synchronized viewports next to each other. "
                        "\n"
                        "slider — single viewport with a draggable divider revealing "
                        "one splat on each side."
                    ),
                }),
                "fov_degrees": ("FLOAT", {
                    "default": 50.0, "min": 5.0, "max": 170.0, "step": 1.0,
                    "tooltip": "Vertical field of view in degrees",
                }),
                "image_width": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 8,
                }),
                "image_height": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 8,
                }),
                "renderer": (["spark", "playcanvas"], {
                    "default": "spark",
                }),
                "transport_format": (["ply", "spz"], {
                    "default": "ply",
                    "tooltip": (
                        "ply — lossless float32. "
                        "\n"
                        "spz — ~9x smaller, server transcodes once and caches."
                    ),
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "viewer"

    def preview(self, ply_path_1, ply_path_2, layout, fov_degrees,
                image_width, image_height, renderer, transport_format="ply"):
        results = {}
        for idx, path in enumerate([ply_path_1, ply_path_2], start=1):
            suffix = str(idx)
            if not path:
                return {"ui": {"error": [f"No PLY path provided for input {idx}"]}}
            if not os.path.exists(path):
                return {"ui": {"error": [f"File not found (input {idx}): {path}"]}}

            filename, subfolder, folder_kind = _resolve_for_view(path)
            file_size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
            num_gaussians = _count_gaussians(path)

            results[f"ply_file_{suffix}"] = [filename]
            results[f"filename_{suffix}"] = [filename]
            results[f"ply_type_{suffix}"] = [folder_kind]
            results[f"ply_subfolder_{suffix}"] = [subfolder]
            results[f"file_size_mb_{suffix}"] = [file_size_mb]
            results[f"num_gaussians_{suffix}"] = [num_gaussians]

        intrinsics = get_default_intrinsics(image_width, image_height, fov_degrees)
        extrinsics = get_default_extrinsics()

        results["layout"] = [layout]
        results["extrinsics"] = [extrinsics]
        results["intrinsics"] = [intrinsics]
        results["fov_degrees"] = [fov_degrees]
        results["renderer"] = [renderer]
        results["transport_format"] = [transport_format]

        return {"ui": results}
