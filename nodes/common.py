# SPDX-License-Identifier: GPL-3.0-or-later

import os

# Optional ComfyUI helpers - avoid hard dependency for offline editing
try:
    import importlib
    folder_paths = importlib.import_module("folder_paths")  # type: ignore[import-not-found]
    COMFYUI_INPUT_FOLDER: str | None = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER: str | None = folder_paths.get_output_directory()
except Exception:  # pragma: no cover - environment-specific
    folder_paths = None  # type: ignore[assignment]
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None


def get_default_extrinsics() -> list[list[float]]:
    """Return default 4x4 identity extrinsics matrix (camera at origin)."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def get_default_intrinsics(width: int = 512, height: int = 512, fov_degrees: float = 50.0) -> list[list[float]]:
    """Return default 3x3 intrinsics matrix from FOV."""
    import math

    fov_rad = math.radians(fov_degrees)
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx  # assume square pixels
    cx = width / 2.0
    cy = height / 2.0

    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def get_recommended_resolution(fov_degrees: float, target_scale: float = 10.0) -> tuple[int, int]:
    """Get recommended width/height based on FOV and target gaussian scale."""
    reference_points = [
        (10.0, 61440.0),
        (30.0, 15360.0),
        (50.0, 10240.0),
        (100.0, 4096.0),
        (120.0, 2560.0),
        (150.0, 1920.0),
        (180.0, 1600.0),
    ]

    fov_clamped = max(10.0, min(180.0, fov_degrees))

    product = 0.0
    if fov_clamped <= reference_points[0][0]:
        product = reference_points[0][1]
    elif fov_clamped >= reference_points[-1][0]:
        product = reference_points[-1][1]
    else:
        for i in range(len(reference_points) - 1):
            fov1, prod1 = reference_points[i]
            fov2, prod2 = reference_points[i + 1]
            if fov1 <= fov_clamped <= fov2:
                t = (fov_clamped - fov1) / (fov2 - fov1)
                product = prod1 + t * (prod2 - prod1)
                break

    base_resolution = product / 10.0
    target_resolution = base_resolution * (10.0 / target_scale)

    calibration_factor = 0.8
    target_resolution = target_resolution * calibration_factor

    resolution = int(target_resolution + 0.5)
    resolution = max(256, min(8192, resolution))
    resolution = ((resolution + 7) // 16) * 16

    return (resolution, resolution)
