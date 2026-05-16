# SPDX-License-Identifier: GPL-3.0-or-later

"""LoadPLY — file picker for 3D Gaussian Splatting PLY files.

Mirrors ComfyUI's LoadImage idiom: lists bare filenames from
`input/`, resolves via `folder_paths.get_annotated_filepath()` (which
handles `[input]/...` and `[temp]/...` annotations transparently),
and validates with `exists_annotated_filepath`.

For PLYs produced UPSTREAM in a workflow (e.g. `GaussianMerge` ->
`PreviewGaussians`), wire the path directly — don't go through this
node.
"""

import logging
import os

import folder_paths

log = logging.getLogger("comfyui-gaussianpack")


def _list_input_plys() -> list[str]:
    """Bare filenames of .ply files in ComfyUI's input/ directory."""
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []
    return sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".ply")
        and os.path.isfile(os.path.join(input_dir, f))
    )


class LoadPLY:
    """Browse a `.ply` file from ComfyUI's input/ directory."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_plys() or ["<no .ply files in input/>"]
        return {
            "required": {
                "ply_file": (files, {
                    "tooltip": "Pick a .ply file from ComfyUI's input/ folder.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, ply_file):
        path = folder_paths.get_annotated_filepath(ply_file)
        if path and os.path.exists(path):
            return str(os.path.getmtime(path))
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, ply_file):
        if not folder_paths.exists_annotated_filepath(ply_file):
            return f"Invalid PLY file: {ply_file}"
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "load"
    CATEGORY = "viewer"

    def load(self, ply_file: str):
        path = folder_paths.get_annotated_filepath(ply_file)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"LoadPLY: {ply_file!r} not found")
        log.info("LoadPLY: %s -> %s", ply_file, path)
        return (path,)
