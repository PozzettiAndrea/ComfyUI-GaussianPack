# SPDX-License-Identifier: GPL-3.0-or-later

"""GaussianPack nodes — preview, merge, and load PLY-format 3D Gaussian splats.

The root `__init__.py` re-exports `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS` from this subpackage. `WEB_DIRECTORY` is
set at the root, not here.
"""

from .preview_gaussian import PreviewGaussians
from .preview_gaussian_spectate import PreviewGaussianSpectate
from .preview_gaussian_camera import PreviewGaussianCamera
from .merge_gaussians import GaussianMerge
from .load_ply import LoadPLY
from .load_ply_output import LoadPLYOutput
from .analyze_gaussians import GaussianAnalysis
from .export_gaussians import GaussianExport
from .transform_gaussian import TransformGaussian
from .gaussians_from_point_cloud import GaussiansFromPointCloud
from .spz_route import register_routes as _register_spz_route
from .load_ply_output import register_routes as _register_load_ply_output_route

_register_spz_route()
_register_load_ply_output_route()

NODE_CLASS_MAPPINGS = {
    "PreviewGaussians": PreviewGaussians,
    "PreviewGaussianSpectate": PreviewGaussianSpectate,
    "PreviewGaussianCamera": PreviewGaussianCamera,
    "GaussianMerge": GaussianMerge,
    "LoadPLY": LoadPLY,
    "LoadPLYOutput": LoadPLYOutput,
    "GaussianAnalysis": GaussianAnalysis,
    "GaussianExport": GaussianExport,
    "TransformGaussian": TransformGaussian,
    "GaussiansFromPointCloud": GaussiansFromPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewGaussians": "Preview Gaussians",
    "PreviewGaussianSpectate": "Preview Gaussian Spectate",
    "PreviewGaussianCamera": "Preview Gaussian Camera",
    "GaussianMerge": "Gaussian Merge to Target",
    "LoadPLY": "Load PLY",
    "LoadPLYOutput": "Load PLY (from Outputs)",
    "GaussianAnalysis": "Gaussian Analysis",
    "GaussianExport": "Gaussian Export",
    "TransformGaussian": "Transform Gaussian",
    "GaussiansFromPointCloud": "Gaussians From Point Cloud",
}
