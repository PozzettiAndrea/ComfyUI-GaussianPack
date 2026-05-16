"""HTTP route that lazily transcodes a PLY -> SPZ v2 (gzipped) and serves it.

Registered into ComfyUI's aiohttp server. The first request for a given
source PLY shells out to the vendored `spz-js` (via Node) to produce a
`<basename>.spz` sibling file; later requests stream the cached file.

We don't use PyPI `spz` (writes v3/v4 which neither Spark nor PlayCanvas
decode in their current shipped builds) — `spz-js` writes v2/gzipped
which both renderers can read.

Query:
    /gaussianpack/spz?filename=<output-relative>.ply
                           &subfolder=<sub>
                           &type=output

Format choice (`ply` vs `spz`) lives on the previewer node's
`transport_format` widget; the JS only hits this route when the user
picks `spz`.
"""
import asyncio
import json
import logging
import os
import shutil
import threading
from pathlib import Path

from aiohttp import web

import folder_paths

log = logging.getLogger("comfyui-gaussianpack")

# Per-path lock — concurrent requests for the same PLY share the cost.
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()

# spz_route.py lives in `nodes/`; the JS subprocess tool lives next to the
# browser-facing assets in `web/server_tools/`. ../web/server_tools is
# resolved from this file's location.
THIS_DIR = Path(__file__).resolve().parent
NODE_SCRIPT = THIS_DIR.parent / "web" / "server_tools" / "ply_to_spz.mjs"


def _lock_for(p: str) -> threading.Lock:
    with _locks_lock:
        if p not in _locks:
            _locks[p] = threading.Lock()
        return _locks[p]


def _resolve_output_path(filename: str, subfolder: str, type_: str) -> Path | None:
    """Resolve a /view-style (filename, subfolder, type) to an absolute
    path, refusing anything that escapes the matching base directory."""
    base_dir = {
        "output": folder_paths.get_output_directory(),
        "input":  folder_paths.get_input_directory(),
        "temp":   folder_paths.get_temp_directory(),
    }.get(type_, folder_paths.get_output_directory())
    base = Path(base_dir).resolve()
    candidate = (base / (subfolder or "") / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


async def _transcode_ply_to_spz(ply_path: Path, spz_path: Path) -> dict:
    """Run the vendored spz-js converter via Node. Atomic via `.part`."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "'node' not found on PATH; install Node.js to enable SPZ transcoding."
        )
    if not NODE_SCRIPT.is_file():
        raise RuntimeError(f"converter script missing: {NODE_SCRIPT}")

    tmp = spz_path.with_suffix(spz_path.suffix + ".part")
    proc = await asyncio.create_subprocess_exec(
        node, str(NODE_SCRIPT), str(ply_path), str(tmp),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise RuntimeError(
            f"ply_to_spz.mjs exited {proc.returncode}: "
            f"{(err or out).decode('utf-8', 'replace')[:500]}"
        )

    tmp.replace(spz_path)
    try:
        return json.loads((out or b"{}").decode("utf-8", "replace").splitlines()[-1])
    except Exception:
        return {}


def register_routes() -> None:
    """Register the SPZ transcode route. Idempotent no-op if PromptServer
    isn't available (static analysis, tests, etc.).

    Catches Exception (not just ImportError) because `from server import
    PromptServer` triggers ComfyUI's nodes.py -> comfy.sd ->
    comfy.model_management, and model_management.py:254 evaluates
    `torch.device(torch.cuda.current_device())` at import time. On
    CPU-only systems (e.g. comfy-test's instantiation subprocess) that
    raises RuntimeError, not ImportError. In real ComfyUI startup
    `server` is already in sys.modules so the import is a sys.modules
    lookup and the cascade doesn't re-run."""
    try:
        from server import PromptServer
    except Exception as e:
        log.warning("SPZ route not registered (%s: %s)", type(e).__name__, e)
        return
    if PromptServer.instance is None:
        log.warning("PromptServer.instance is None; SPZ route not registered.")
        return

    routes = PromptServer.instance.routes

    @routes.get("/gaussianpack/spz")
    async def get_spz(request: web.Request) -> web.StreamResponse:
        filename  = request.query.get("filename", "")
        subfolder = request.query.get("subfolder", "")
        type_     = request.query.get("type", "output")
        if not filename:
            return web.Response(status=400, text="missing 'filename'")
        if not filename.lower().endswith(".ply"):
            return web.Response(status=400, text="filename must end in .ply")

        ply = _resolve_output_path(filename, subfolder, type_)
        if ply is None:
            return web.Response(status=403, text="path traversal rejected")
        if not ply.is_file():
            return web.Response(status=404, text=f"PLY not found: {ply}")

        spz = ply.with_suffix(".spz")
        cache_fresh = spz.is_file() and spz.stat().st_mtime >= ply.stat().st_mtime

        if not cache_fresh:
            with _lock_for(str(spz)):
                cache_fresh = spz.is_file() and spz.stat().st_mtime >= ply.stat().st_mtime
                if not cache_fresh:
                    log.info("SPZ cache miss; transcoding %s -> %s", ply.name, spz.name)
                    try:
                        stats = await _transcode_ply_to_spz(ply, spz)
                    except Exception as e:
                        log.exception("SPZ transcode failed for %s", ply)
                        return web.Response(status=500, text=f"SPZ transcode failed: {e}")
                    log.info(
                        "SPZ written: %s (%.1f MB -> %.1f MB, %.1fx) stats=%s",
                        spz.name,
                        ply.stat().st_size / (1024 * 1024),
                        spz.stat().st_size / (1024 * 1024),
                        ply.stat().st_size / max(1, spz.stat().st_size),
                        stats,
                    )

        return web.FileResponse(
            spz,
            headers={
                "Content-Type": "application/octet-stream",
                "Cache-Control": "no-cache",
            },
        )
