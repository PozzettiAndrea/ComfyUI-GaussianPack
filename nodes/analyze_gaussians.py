# SPDX-License-Identifier: GPL-3.0-or-later

"""GaussianAnalysis — inspect a 3DGS PLY's header and report what's inside.

Parses just the header (no full file read) so it's instant even for
hundreds-of-MB scenes. Detects:
  - element vertex count
  - PLY encoding (ascii / binary little/big endian)
  - SH degree (0/1/2/3) from the f_rest_N property count
  - presence of baked RGB (uchar red/green/blue)
  - presence of opacity, scale, rotation
  - per-property dtype distribution -> "all float32" vs "quantized"
  - file size + bytes per splat
  - a one-line "flavor" guess (raw 3DGS / SH-stripped / RGB-baked /
    compressed-ish)

Returns a multi-line STRING for chaining and also displays it in the
node body via the `ui` payload (OUTPUT_NODE pattern).
"""

import logging
import os

log = logging.getLogger("comfyui-gaussianpack")

# SH degree -> expected f_rest_* count (3 channels each, degrees 1..deg).
# deg 0: 0 rest (just f_dc_0/1/2 as DC term)
# deg 1: 3 coef * 3 ch = 9
# deg 2: (3+5) coef * 3 ch = 24
# deg 3: (3+5+7) coef * 3 ch = 45
_SH_BY_RESTS = {0: 0, 9: 1, 24: 2, 45: 3}


def _parse_ply_header(path: str) -> dict:
    """Read just the PLY ASCII header from disk. Returns a dict with:
      encoding (str), version (str), element_vertex (int),
      properties (list of (dtype, name)), header_bytes (int).
    Raises ValueError if not a PLY or header malformed.
    """
    with open(path, "rb") as f:
        # Headers are small; cap at 64KB to be safe.
        head = f.read(65536)
    if not head.startswith(b"ply\n"):
        raise ValueError("not a PLY file (missing 'ply' magic)")
    end = head.find(b"end_header\n")
    if end < 0:
        raise ValueError("PLY header missing 'end_header' line")
    header_text = head[: end + len(b"end_header\n")].decode("latin-1", errors="replace")
    header_bytes = end + len(b"end_header\n")

    encoding = "unknown"
    version = ""
    vcount = 0
    properties: list[tuple[str, str]] = []
    in_vertex_element = False

    for line in header_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        head_kw = parts[0]
        if head_kw == "format" and len(parts) >= 3:
            encoding = parts[1]
            version = parts[2]
        elif head_kw == "element":
            if len(parts) >= 3 and parts[1] == "vertex":
                vcount = int(parts[2])
                in_vertex_element = True
            else:
                in_vertex_element = False
        elif head_kw == "property" and in_vertex_element:
            # Two forms: "property <type> <name>", "property list ..."
            if parts[1] == "list":
                continue  # don't expect lists on 3DGS vertices
            dtype, name = parts[1], parts[2]
            properties.append((dtype, name))

    return {
        "encoding": encoding,
        "version": version,
        "element_vertex": vcount,
        "properties": properties,
        "header_bytes": header_bytes,
    }


def _classify(props: list[tuple[str, str]]) -> dict:
    """Derive flavor / feature flags from the property list."""
    names = {name: dtype for dtype, name in props}
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    has_f_dc = all(f"f_dc_{i}" in names for i in (0, 1, 2))
    has_rgb_uchar = all(
        names.get(c, "").lower() in {"uchar", "uint8", "u8"}
        for c in ("red", "green", "blue")
    ) and all(c in names for c in ("red", "green", "blue"))
    has_opacity = "opacity" in names
    has_scale = all(f"scale_{i}" in names for i in (0, 1, 2))
    has_rot = all(f"rot_{i}" in names for i in (0, 1, 2, 3))
    has_packed = any(n.startswith(("packed_", "chunk_")) for n in names)

    sh_degree = _SH_BY_RESTS.get(n_rest)
    if sh_degree is None:
        # Non-canonical count — report raw rest count
        sh_degree_str = f"non-standard ({n_rest} f_rest_* properties)"
    else:
        sh_degree_str = str(sh_degree)

    dtypes = {dt for dt, _ in props}
    if dtypes <= {"float", "float32"}:
        precision = "float32 (all)"
    elif dtypes <= {"uchar", "uint8", "u8"}:
        precision = "uint8 (all)"
    else:
        precision = f"mixed ({', '.join(sorted(dtypes))})"

    if has_packed:
        flavor = "compressed / packed (PlayCanvas/SuperSplat-style)"
    elif has_rgb_uchar and not has_f_dc:
        flavor = "RGB-baked PLY (no SH coefficients)"
    elif has_f_dc and n_rest == 0:
        flavor = "SH-stripped 3DGS (DC only, no view-dependent term)"
    elif has_f_dc and sh_degree is not None:
        flavor = f"raw 3DGS PLY (SH degree {sh_degree})"
    else:
        flavor = "unknown / custom"

    return {
        "flavor": flavor,
        "precision": precision,
        "sh_degree": sh_degree_str,
        "has_rgb_uchar": has_rgb_uchar,
        "has_f_dc": has_f_dc,
        "has_opacity": has_opacity,
        "has_scale": has_scale,
        "has_rotation": has_rot,
        "n_f_rest": n_rest,
    }


def _format_report(path: str, header: dict, info: dict, file_size: int) -> str:
    vcount = header["element_vertex"]
    bps = (file_size / vcount) if vcount else 0
    lines = [
        f"== Gaussian PLY analysis ==",
        f"  path:           {path}",
        f"  size:           {file_size / (1024 * 1024):.2f} MB  ({file_size:,} bytes)",
        f"  encoding:       {header['encoding']} {header['version']}",
        f"  splat count:    {vcount:,}",
        f"  bytes/splat:    {bps:.1f}",
        f"  flavor:         {info['flavor']}",
        f"  SH degree:      {info['sh_degree']}",
        f"  precision:      {info['precision']}",
        f"  has f_dc_*:     {info['has_f_dc']}",
        f"  has RGB uchar:  {info['has_rgb_uchar']}",
        f"  has opacity:    {info['has_opacity']}",
        f"  has scale_*:    {info['has_scale']}",
        f"  has rot_*:      {info['has_rotation']}",
        f"  total props:    {len(header['properties'])}",
    ]
    return "\n".join(lines)


class GaussianAnalysis:
    """Inspect a 3DGS PLY and report format / SH degree / precision / size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file to analyze",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "viewer"

    def analyze(self, ply_path: str):
        if not ply_path or not os.path.exists(ply_path):
            err = f"GaussianAnalysis: file not found: {ply_path!r}"
            log.warning(err)
            return {"ui": {"report": [err]}, "result": (err,)}

        try:
            header = _parse_ply_header(ply_path)
            info = _classify(header["properties"])
            file_size = os.path.getsize(ply_path)
            report = _format_report(ply_path, header, info, file_size)
        except Exception as e:
            err = f"GaussianAnalysis: failed to parse {ply_path}: {e}"
            log.exception(err)
            return {"ui": {"report": [err]}, "result": (err,)}

        log.info("GaussianAnalysis:\n%s", report)
        return {"ui": {"report": [report]}, "result": (report,)}
