# SPDX-License-Identifier: GPL-3.0-or-later

"""Preview Gaussian Spectate — fly-camera viewer for Gaussian splats.

Same renderer/transport plumbing as PreviewGaussians, but the viewer
HTML swaps TrackballControls for a custom SpectateControls that uses:

    W / A / S / D    forward / left / back / right (in camera local frame)
    Space            up  (world up)
    C                down (world up)
    Shift            boost (3x speed)
    Left-mouse drag  look around (yaw + clamped pitch)
    Mouse wheel      scale speed up/down

The `move_speed` input seeds the initial speed; a slider in the viewer
UI lets the user adjust at runtime. All other behavior matches
PreviewGaussians (iframe path, /view URL resolution, camera-state
persistence via node.properties).
"""

import os

from .common import (
    get_default_extrinsics,
    get_default_intrinsics,
)
# Re-use the path resolution + header sniff helpers from the existing
# node; they're internal helpers but the API is stable.
from .preview_gaussian import _resolve_for_view, _count_gaussians


class PreviewGaussianSpectate:
    """Free-flight (spectate-mode) viewer for Gaussian splats."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file",
                }),
                "fov_degrees": ("FLOAT", {
                    "default": 70.0, "min": 5.0, "max": 170.0, "step": 1.0,
                    "tooltip": (
                        "Vertical field of view in degrees. Wider (70-90) "
                        "feels more natural in fly-mode; default raised "
                        "from PreviewGaussians' 50."
                    ),
                }),
                "image_width": ("INT", {
                    "default": 768, "min": 64, "max": 4096, "step": 8,
                }),
                "image_height": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 8,
                }),
                "move_speed": ("FLOAT", {
                    "default": 2.0, "min": 0.05, "max": 50.0, "step": 0.05,
                    "tooltip": (
                        "Initial movement speed in scene-units per second. "
                        "Adjust at runtime via the slider in the viewer or "
                        "scroll the mouse wheel. Shift = 3x boost."
                    ),
                }),
                "renderer": (["spark", "playcanvas"], {
                    "default": "spark",
                    "tooltip": (
                        "spark — Three.js + WebGL2, all formats. "
                        "playcanvas — WebGPU adapter (falls back to spark "
                        "for now)."
                    ),
                }),
                "transport_format": (["ply", "spz"], {
                    "default": "ply",
                    "tooltip": (
                        "ply — lossless, larger download. "
                        "spz — server transcodes to SPZ v2 once, ~9x smaller, "
                        "SH2/SH3 quantized to 4 bits."
                    ),
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "viewer"

    def preview(self, ply_path, fov_degrees, image_width, image_height,
                move_speed, renderer, transport_format="ply"):
        if not ply_path:
            return {"ui": {"error": ["No PLY path provided"]}}
        if not os.path.exists(ply_path):
            return {"ui": {"error": [f"File not found: {ply_path}"]}}

        filename, subfolder, folder_kind = _resolve_for_view(ply_path)

        file_size_mb = round(os.path.getsize(ply_path) / (1024 * 1024), 2)
        num_gaussians = _count_gaussians(ply_path)
        intrinsics = get_default_intrinsics(image_width, image_height, fov_degrees)
        extrinsics = get_default_extrinsics()

        return {"ui": {
            "ply_file": [filename],
            "filename": [filename],
            "ply_type": [folder_kind],
            "ply_subfolder": [subfolder],
            "file_size_mb": [file_size_mb],
            "num_gaussians": [num_gaussians],
            "extrinsics": [extrinsics],
            "intrinsics": [intrinsics],
            "fov_degrees": [fov_degrees],
            "renderer": [renderer],
            "transport_format": [transport_format],
            # Spectate-only — JS forwards via postMessage to the iframe.
            "mode": ["spectate"],
            "move_speed": [move_speed],
        }}
