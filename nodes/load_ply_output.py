# SPDX-License-Identifier: GPL-3.0-or-later

"""LoadPLYOutput — file picker for .ply files in ComfyUI's output/ folder.

Parallel to `LoadPLY` (which scans input/) and analogous to ComfyUI's
native `LoadImageOutput`. Recursively walks the output directory so deep
artifacts from training pipelines (e.g. HYWM2 / SHARP / gaussian-splat
training writes nested paths like
`output/hywm2_train_<ts>/gs_data/gs_result/ply/point_cloud_99.ply`)
show up in the dropdown without the user having to type a path.

Sorted by mtime descending so the most recently written PLY is the
default selection — typical use case is "load the splat I just trained."
"""

import logging
import os
from pathlib import Path

import folder_paths

log = logging.getLogger("comfyui-gaussianpack")


def _list_output_plys() -> list[str]:
    """Relative paths to .ply files under ComfyUI's output/ directory,
    sorted by mtime descending (newest first).

    Display strings are relative paths so the dropdown stays readable;
    `load()` re-joins with the output dir to get the absolute path.
    """
    out_dir = folder_paths.get_output_directory()
    if not out_dir or not os.path.isdir(out_dir):
        return []
    out_path = Path(out_dir)
    entries: list[tuple[float, str]] = []
    for p in out_path.rglob("*.ply"):
        if not p.is_file():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        try:
            rel = p.relative_to(out_path).as_posix()
        except ValueError:
            rel = p.as_posix()
        entries.append((mtime, rel))
    # Newest first.
    entries.sort(key=lambda e: e[0], reverse=True)
    return [rel for _, rel in entries]


def register_routes() -> None:
    """No-op kept for API compatibility — the dropdown driven by
    `_list_output_plys()` doesn't need a server route. ComfyUI's frontend
    only consumes remote-route options for image/video upload widgets
    (see `isMediaUploadComboInput` in the frontend bundle); generic
    COMBOs read their options statically from `INPUT_TYPES`. We surface
    a fresh list on every `INPUT_TYPES` call (every workflow page-load).
    """
    return


class LoadPLYOutput:
    """Browse a `.ply` file from ComfyUI's output/ directory (recursive)."""

    @classmethod
    def INPUT_TYPES(cls):
        # v1 list form. `(files_list, config_dict)` is the only form that
        # populates the dropdown options correctly for non-media-upload
        # widgets. The `remote` config in `config_dict` does NOT update
        # the dropdown's options list for generic COMBOs — verified by
        # reading the frontend bundle. The only place `widget.options`
        # gets set is from `e[0]` (this list) for v1 specs, or from
        # `e[1].options` for v2 specs. The `remote` flow updates only
        # the SELECTED VALUE via `control_after_refresh`, not the
        # available options. So we surface a fresh list on every
        # `INPUT_TYPES` call (every workflow page-load) and rely on the
        # user pressing Ctrl/Cmd+R after deleting / regenerating files.
        files = _list_output_plys() or ["<no .ply files in output/>"]
        return {
            "required": {
                "ply_file": (files, {
                    "tooltip": (
                        "Pick a .ply file from ComfyUI's output/ folder. "
                        "Scans recursively — training outputs deep in "
                        "subfolders (e.g. hywm2_train_*/gs_data/gs_result/"
                        "ply/point_cloud_99.ply) show up here. Sorted "
                        "newest-first by mtime.\n\n"
                        "Note: the dropdown list is captured at workflow "
                        "load time. After deleting/regenerating PLYs in "
                        "output/, press Ctrl/Cmd+R to reload the page — "
                        "ComfyUI's frontend has no per-widget refresh "
                        "mechanism for generic COMBOs (only for image/"
                        "video upload widgets, which we deliberately "
                        "don't pretend to be)."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, ply_file):
        out_dir = folder_paths.get_output_directory()
        if not out_dir or not ply_file or ply_file.startswith("<no "):
            return ""
        path = os.path.join(out_dir, ply_file)
        if os.path.exists(path):
            return str(os.path.getmtime(path))
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, ply_file):
        if ply_file is None or ply_file.startswith("<no "):
            return "No .ply files in output/. Generate some first."
        out_dir = folder_paths.get_output_directory()
        if not out_dir:
            return "ComfyUI output directory not configured."
        path = os.path.join(out_dir, ply_file)
        if not os.path.isfile(path):
            return f"PLY not found: {ply_file}"
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "load"
    CATEGORY = "viewer"

    def load(self, ply_file: str):
        out_dir = folder_paths.get_output_directory()
        if not out_dir:
            raise FileNotFoundError(
                "LoadPLYOutput: ComfyUI output directory not configured."
            )
        if ply_file is None or ply_file.startswith("<no "):
            raise FileNotFoundError(
                "LoadPLYOutput: no .ply files in output/ folder. "
                "Generate some first (e.g. via HYWM2GaussianTrain, "
                "SharpPredictGaussiansFromMetricDepth, or GaussianExport)."
            )
        path = os.path.join(out_dir, ply_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoadPLYOutput: {path!r} not found")
        log.info("LoadPLYOutput: %s -> %s", ply_file, path)
        return (path,)
