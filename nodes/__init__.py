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
from .analyze_gaussians import GaussianAnalysis
from .export_gaussians import GaussianExport
from .transform_gaussian import TransformGaussian
from .spz_route import register_routes as _register_spz_route

_register_spz_route()

NODE_CLASS_MAPPINGS = {
    "PreviewGaussians": PreviewGaussians,
    "PreviewGaussianSpectate": PreviewGaussianSpectate,
    "PreviewGaussianCamera": PreviewGaussianCamera,
    "GaussianMerge": GaussianMerge,
    "LoadPLY": LoadPLY,
    "GaussianAnalysis": GaussianAnalysis,
    "GaussianExport": GaussianExport,
    "TransformGaussian": TransformGaussian,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewGaussians": "Preview Gaussians",
    "PreviewGaussianSpectate": "Preview Gaussian Spectate",
    "PreviewGaussianCamera": "Preview Gaussian Camera",
    "GaussianMerge": "Gaussian Merge to Target",
    "LoadPLY": "Load PLY",
    "GaussianAnalysis": "Gaussian Analysis",
    "GaussianExport": "Gaussian Export",
    "TransformGaussian": "Transform Gaussian",
}
