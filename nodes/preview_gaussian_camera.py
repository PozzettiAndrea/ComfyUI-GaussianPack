# SPDX-License-Identifier: GPL-3.0-or-later

"""PreviewGaussianCamera — render a 3DGS PLY from given camera poses to IMAGE.

Tiered renderer:
  1. Try `gsplat.rasterization` (reference CUDA path). ~10-50 ms / frame
     at 1M gaussians, 1024² output, on a 3090. Declared as an OPTIONAL
     [cuda] dep in `nodes/comfy-env.toml`; cuda-wheels' gsplat recipe
     handles NVIDIA installs. Non-NVIDIA installs (AMD ROCm, Apple
     Silicon Metal, CPU-only) get a no-op skip at install time.
  2. Fall back to a pure-PyTorch EWA splatter. Works on any torch
     backend (CUDA, ROCm, MPS, CPU). Slower (~1-10 sec/frame depending
     on gaussian count + resolution) but correct.

Inputs (ComfyUI workflow sockets):
  ply_path    STRING (forceInput)  — path to a 3DGS PLY (e.g. from
                                     HYWM2GaussianTrain.ply_path,
                                     SharpPredictGaussiansFromMetricDepth,
                                     MergeGaussians, or any external
                                     3DGS tool).
  extrinsics  EXTRINSICS           — [N, 4, 4] world-to-camera (CameraPack
                                     convention). Renders N frames.
  intrinsics  INTRINSICS           — [N, 3, 3] pixel-K, or normalized-K
                                     ([0,1] coords, auto-detected and
                                     rescaled). Single [3, 3] is
                                     broadcast to all N views.

Output:
  image       IMAGE [N, H, W, 3]   — per-view rendered image in [0, 1].

Workflow position: drop after any node that produces a 3DGS PLY +
camera matrices to visually verify what the splat looks like from those
viewpoints. Faster than firing up an external viewer.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("comfyui-gaussianpack")

# SH band-0 to RGB conversion factor (3DGS convention).
#   rgb = sh0 · C0 + 0.5  in [0, 1]    where C0 = 1 / (2·√π) ≈ 0.282
_C0 = 1.0 / (2.0 * math.sqrt(math.pi))


def _p(msg: str) -> None:
    """Worker-stderr print. Matches the helper pattern other packs use."""
    print(f"[PreviewGaussianCamera] {msg}", file=sys.stderr, flush=True)


def _request_vram_eviction(needed_bytes: int) -> None:
    """Ask comfy-env's parent ComfyUI process to evict cross-worker models so
    this transient-tensor node has VRAM headroom. Mirrors the helpers in
    PanoramaDepthMerge / HYWM2GaussianTrain — see depth_merge.py:
    _request_vram_eviction for the full rationale.

    Three-step:
      1. Ask the parent ComfyUI process (via comfy_worker.call_parent IPC) to
         evict sibling-worker patchers across all worker subprocesses. The
         parent's _handle_vram_budget calls mm.free_memory(N * 1.1, device)
         on its side -- which iterates current_loaded_models across every
         registered patcher (HYWM2Reconstruct's DiT lives in the hywm2-nodes
         worker, MoGe2's model in moge2-nodes, etc.) and unloads as needed.
      2. In-worker mm.free_memory as belt-and-braces (no-op for GaussianPack
         since it doesn't register patchers itself).
      3. torch.cuda.empty_cache() to release any cached blocks held by our
         worker's allocator so the next big alloc sees the freshly-freed
         space contiguously.

    Cleanly no-ops outside a comfy-env worker subprocess (e.g. unit tests
    in plain Python without the worker shim).
    """
    try:
        import comfy_worker  # noqa: F401 - injected at worker startup
        try:
            comfy_worker.call_parent(
                "request_vram_budget", total_size=int(needed_bytes)
            )
            _p(f"  -> requested {needed_bytes / 1e9:.2f} GB eviction "
               f"via comfy_worker.call_parent")
        except Exception as e:
            _p(f"  -> comfy_worker.call_parent failed: {e}")
    except ImportError:
        _p("  -> comfy_worker module unavailable; local free_memory only")

    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        mm.free_memory(int(needed_bytes), device)
    except Exception as e:
        _p(f"  -> local mm.free_memory failed: {e}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def _load_3dgs_ply(path: str) -> dict:
    """Parse a 3DGS PLY into the canonical tensor dict.

    Returns dict with:
      means      [N, 3]   world-space positions (float32)
      quats      [N, 4]   wxyz quaternions (UNNORMALIZED; renderer
                          normalizes at use-time per 3DGS convention)
      scales     [N, 3]   per-axis LOG scales (i.e. real_scale = exp(scales))
      opacities  [N]      sigmoid-space opacities (i.e. real = sigmoid(opacities))
      sh0        [N, 1, 3] band-0 SH coefficients
      shN        [N, K, 3] higher SH coefficients; K = (sh_degree+1)**2 - 1.
                          Empty (K=0) for sh_degree=0 PLYs.

    Matches the field layout used by the original `graphdeco-inria/
    gaussian-splatting` code path that all downstream tools (gsplat,
    nerfstudio, SuperSplat, etc.) interop on.
    """
    from plyfile import PlyData

    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"{path}: PLY has no 'vertex' element (not a 3DGS PLY)")
    v = ply["vertex"].data
    N = len(v)
    names = set(v.dtype.names)

    # Positions.
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError(f"{path}: PLY vertex missing x/y/z (not a point cloud)")
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # Quaternions (rot_0=w, rot_1=x, rot_2=y, rot_3=z).
    if all(k in names for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                         axis=1).astype(np.float32)
    else:
        # Identity quat fallback — interprets the splat as axis-aligned.
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0

    # Log-scales.
    if all(k in names for k in ("scale_0", "scale_1", "scale_2")):
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                          axis=1).astype(np.float32)
    else:
        # Tiny default scale; renderer will see ~zero-area splats but won't crash.
        scales = np.full((N, 3), math.log(1e-3), dtype=np.float32)

    # Opacity (logit space).
    if "opacity" in names:
        opacities = np.asarray(v["opacity"]).astype(np.float32)
    else:
        opacities = np.zeros(N, dtype=np.float32)  # sigmoid(0) = 0.5

    # Band-0 SH (a.k.a. the diffuse color basis).
    if all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
                       axis=1).astype(np.float32)[:, None, :]   # [N, 1, 3]
    else:
        sh0 = np.full((N, 1, 3), 0.5 / _C0, dtype=np.float32)   # rgb ≈ 1.0

    # Higher SH bands (f_rest_*). Layout per the 3DGS convention:
    #   first K_AC entries = channel R, next K_AC = G, next K_AC = B
    # where K_AC = (sh_degree + 1)² - 1. Often absent (sh_degree = 0).
    rest_keys = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    K_total = len(rest_keys)
    if K_total > 0 and K_total % 3 == 0:
        K_AC = K_total // 3
        # Stack into [N, K_AC, 3] in (R, G, B) order.
        stacked = np.stack([np.asarray(v[k]) for k in rest_keys],
                           axis=1).astype(np.float32)   # [N, K_total]
        shN = stacked.reshape(N, 3, K_AC).transpose(0, 2, 1)  # [N, K_AC, 3]
    else:
        shN = np.zeros((N, 0, 3), dtype=np.float32)

    log.info(
        "[PreviewGaussianCamera] loaded %d gaussians from %s "
        "(sh_degree=%d)",
        N, Path(path).name,
        int(round(math.sqrt(1 + shN.shape[1]))) - 1 if shN.shape[1] > 0 else 0,
    )

    return {
        "means": torch.from_numpy(means),
        "quats": torch.from_numpy(quats),
        "scales": torch.from_numpy(scales),
        "opacities": torch.from_numpy(opacities),
        "sh0": torch.from_numpy(sh0),
        "shN": torch.from_numpy(shN),
    }


# ---------------------------------------------------------------------------
# Intrinsics convention sniff (same as Sharp's / HYWM2's)
# ---------------------------------------------------------------------------

def _normalize_K_to_pixel(intr: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Detect normalized-K (PanoPack convention, fx < 2 in [0,1] coords) and
    rescale to pixel-K. Pass-through if already pixel-K.
    """
    intr = intr.detach().float().clone()
    sample_fx = float(intr[0, 0, 0] if intr.dim() == 3 else intr[0, 0])
    if sample_fx < 2.0:
        if intr.dim() == 3:
            intr[:, 0, :] *= float(W)
            intr[:, 1, :] *= float(H)
        else:
            intr[0, :] *= float(W)
            intr[1, :] *= float(H)
        sample_fx_after = float(intr[0, 0, 0] if intr.dim() == 3 else intr[0, 0])
        _p(f"detected normalized intrinsics (fx<2); rescaled to pixel-K "
           f"for {W}×{H}: fx={sample_fx_after:.1f}")
    return intr


# ---------------------------------------------------------------------------
# Render path 1: gsplat (CUDA, reference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _render_gsplat(
    splat: dict,
    extrinsics: torch.Tensor,
    intrinsics_pixel: torch.Tensor,
    H: int,
    W: int,
    near: float,
    background: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Render via gsplat.rasterization, one view at a time.

    Why per-view: gsplat's multi-camera fused path allocates an
    `image_ids = batch_ids * C + camera_ids` index tensor at line 442
    of gsplat/rendering.py, sized by N_gauss × V × avg_tiles_per_gauss.
    At V=12, N=5.5M, 1024² with deep tile overlap, that's >24 GB and
    OOMs on a 3090. Per-view, the workspace shrinks by a factor of V
    (here 12×) and fits comfortably. We lose the batched-fusion
    marginal speedup but the cost is dwarfed by the cuda-kernel
    runtime which already dominates.

    Raises ImportError if gsplat isn't available -- the caller catches
    and falls back to the torch path.
    """
    from gsplat.rendering import rasterization   # noqa: F401 - present check

    means = splat["means"].to(device, torch.float32)
    quats = splat["quats"].to(device, torch.float32)
    scales = splat["scales"].to(device, torch.float32).exp()        # log -> linear
    opacities = splat["opacities"].to(device, torch.float32).sigmoid()
    sh0 = splat["sh0"].to(device, torch.float32)
    shN = splat["shN"].to(device, torch.float32)
    # gsplat expects [N, K_total, 3] for SH, where K_total = (sh_deg+1)²
    if shN.shape[1] > 0:
        colors = torch.cat([sh0, shN], dim=1)
    else:
        colors = sh0
    sh_degree = int(round(math.sqrt(colors.shape[1]))) - 1

    viewmats = extrinsics.to(device, torch.float32)
    if viewmats.dim() == 2:
        viewmats = viewmats.unsqueeze(0)
    Ks = intrinsics_pixel.to(device, torch.float32)
    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0).expand(viewmats.shape[0], 3, 3).contiguous()

    V = int(viewmats.shape[0])
    bg = background.to(device).unsqueeze(0)  # [1, 3]

    out_frames = []
    for v in range(V):
        # Slice ONE view at a time. Per-view workspace ≈ 1×V smaller than
        # the batched path, and the per-call overhead is negligible vs
        # the cuda-kernel cost.
        rc, _alphas, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats[v:v + 1],
            Ks=Ks[v:v + 1],
            width=int(W),
            height=int(H),
            near_plane=float(near),
            sh_degree=sh_degree,
            backgrounds=bg,
        )
        out_frames.append(rc[0].clamp(0.0, 1.0))   # [H, W, 3]
        # Release per-view cached blocks before the next iteration so the
        # caching allocator doesn't pile up across views.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.stack(out_frames, dim=0)          # [V, H, W, 3]


# ---------------------------------------------------------------------------
# Render path 2: pure-PyTorch EWA splatter (fallback)
# ---------------------------------------------------------------------------

def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """wxyz quaternions -> 3x3 rotation matrices. Handles un-normalized
    quats by normalizing first (matches 3DGS convention).
    """
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


@torch.no_grad()
def _render_torch(
    splat: dict,
    extrinsics: torch.Tensor,
    intrinsics_pixel: torch.Tensor,
    H: int,
    W: int,
    near: float,
    background: torch.Tensor,
    device: torch.device,
    chunk_size: int = 16_384,
) -> torch.Tensor:
    """Pure-PyTorch EWA splatter. ~150 LOC, no CUDA-specific deps.

    Algorithm per the 3DGS paper:
      1. Project gaussian means to camera space via w2c.
      2. Compute 2D image-plane covariance via the projection Jacobian:
           Σ_2d = J · R_cam · Σ_3d · R_camᵀ · Jᵀ
      3. Z-sort gaussians near-to-far.
      4. For each pixel, evaluate every overlapping gaussian's 2D
         Mahalanobis distance, compute α = opacity · exp(-0.5·d²), and
         alpha-composite via front-to-back accumulation.

    We process gaussians in chunks across the chunk dim (not pixels) to
    fit memory. Per chunk we evaluate against ALL H·W pixels at once via
    a single broadcasted op.

    Runs anywhere torch runs (CUDA, ROCm, MPS, CPU). Slower than gsplat
    but correct. SH band 0 only (matches sh_degree=0 PLYs); higher bands
    ignored for the fallback path -- gsplat is the fast path for those.
    """
    means_w = splat["means"].to(device, torch.float32)
    quats = splat["quats"].to(device, torch.float32)
    scales = splat["scales"].to(device, torch.float32).exp()
    opacities = splat["opacities"].to(device, torch.float32).sigmoid()
    sh0 = splat["sh0"].to(device, torch.float32)[:, 0]                    # [N, 3]
    rgbs = (sh0 * _C0 + 0.5).clamp(0.0, 1.0)                              # [N, 3]

    # 3D cov in WORLD: Σ_3d = R · diag(s²) · Rᵀ
    R_g = _quat_to_rotmat(quats)                                          # [N, 3, 3]
    cov_3d_world = R_g @ torch.diag_embed(scales * scales) @ R_g.transpose(-1, -2)

    if extrinsics.dim() == 2:
        extrinsics = extrinsics.unsqueeze(0)
    if intrinsics_pixel.dim() == 2:
        intrinsics_pixel = intrinsics_pixel.unsqueeze(0).expand(
            extrinsics.shape[0], 3, 3
        ).contiguous()
    V = int(extrinsics.shape[0])

    out_images = []
    for v in range(V):
        w2c = extrinsics[v].to(device, torch.float32)
        K = intrinsics_pixel[v].to(device, torch.float32)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # World -> camera frame.
        means_cam = means_w @ R.T + t                                     # [N, 3]
        z_cam = means_cam[:, 2]
        # Cull behind near plane.
        keep = z_cam > near
        if not torch.any(keep):
            out_images.append(background.expand(H, W, 3).clone())
            continue

        means_cam = means_cam[keep]
        z_cam = z_cam[keep]
        cov_3d_local = cov_3d_world[keep]
        rgbs_v = rgbs[keep]
        opa_v = opacities[keep]

        # Cov in camera frame: Σ_cam = R · Σ_world · Rᵀ
        cov_3d_cam = R @ cov_3d_local @ R.T

        # 2D projection Jacobian at the gaussian's camera-space center.
        #   u = fx · X/Z + cx,  v = fy · Y/Z + cy
        # J = [[fx/Z, 0, -fx·X/Z²], [0, fy/Z, -fy·Y/Z²]]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X = means_cam[:, 0]
        Y = means_cam[:, 1]
        Z = z_cam.clamp_min(near)
        N_kept = X.shape[0]
        J = torch.zeros(N_kept, 2, 3, device=device, dtype=torch.float32)
        J[:, 0, 0] = fx / Z
        J[:, 0, 2] = -fx * X / (Z * Z)
        J[:, 1, 1] = fy / Z
        J[:, 1, 2] = -fy * Y / (Z * Z)

        # 2D image-plane covariance + low-pass regularizer (matches 3DGS paper).
        cov_2d = J @ cov_3d_cam @ J.transpose(-1, -2)                     # [N, 2, 2]
        cov_2d[:, 0, 0] += 0.3
        cov_2d[:, 1, 1] += 0.3

        # Pixel centers of the projected gaussians.
        uv_pixel = torch.stack([fx * X / Z + cx, fy * Y / Z + cy], dim=-1)

        # Conic = inverse of cov_2d (for exp(-½ Δᵀ Σ⁻¹ Δ)).
        det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1] * cov_2d[:, 1, 0]
        det = det.clamp_min(1e-10)
        conic = torch.stack([
             cov_2d[:, 1, 1] / det,
            -cov_2d[:, 0, 1] / det,
             cov_2d[:, 0, 0] / det,
        ], dim=-1)                                                        # [N, 3] = (a, b, c)

        # Bound: 3σ radius from λ_max of cov_2d. Used to cull pixels.
        trace = cov_2d[:, 0, 0] + cov_2d[:, 1, 1]
        lambda_max = 0.5 * (trace + torch.sqrt((trace * trace - 4 * det).clamp_min(0)))
        radius = 3.0 * torch.sqrt(lambda_max.clamp_min(1e-6))

        # X/Y frustum cull: drop gaussians whose 3σ ellipse lies entirely
        # outside the image bounds. gsplat does this internally; the torch
        # fallback has to as well or it wastes per-pixel evaluation on
        # off-screen splats (a 12-view panorama renders ~12× more gaussians
        # than any one view actually sees).
        orig_N = uv_pixel.shape[0]
        in_bounds = (
            (uv_pixel[:, 0] + radius >= 0) & (uv_pixel[:, 0] - radius < W) &
            (uv_pixel[:, 1] + radius >= 0) & (uv_pixel[:, 1] - radius < H)
        )
        if not torch.all(in_bounds):
            uv_pixel = uv_pixel[in_bounds]
            conic = conic[in_bounds]
            radius = radius[in_bounds]
            rgbs_v = rgbs_v[in_bounds]
            opa_v = opa_v[in_bounds]
            Z = Z[in_bounds]
            _p(f"  view {v}: {orig_N} → {uv_pixel.shape[0]} in-frustum gaussians "
               f"({100 * uv_pixel.shape[0] / max(orig_N, 1):.0f}%)")
            if uv_pixel.shape[0] == 0:
                out_images.append(background.expand(H, W, 3).clone())
                continue

        # Z-sort near-to-far for front-to-back alpha-compositing.
        sort_idx = torch.argsort(Z)
        uv_pixel = uv_pixel[sort_idx]
        conic = conic[sort_idx]
        radius = radius[sort_idx]
        rgbs_v = rgbs_v[sort_idx]
        opa_v = opa_v[sort_idx]
        N_kept = uv_pixel.shape[0]

        # Pixel grid.
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")                    # [H, W]

        # Accumulators (front-to-back alpha compositing).
        T = torch.ones(H, W, device=device, dtype=torch.float32)
        accum = torch.zeros(H, W, 3, device=device, dtype=torch.float32)

        # Process in chunks across the gaussian dim. Per-chunk peak memory
        # is `chunk_size · H · W · 4 B` for the alpha map; at chunk=16k, H=W=1024
        # that's ~67 GB — too big. So further chunk to bounded slabs and accumulate.
        max_pixel_buf_gb = 1.5
        max_chunk = max(1, int(max_pixel_buf_gb * 1024**3 / (H * W * 4)))
        eff_chunk = min(chunk_size, max_chunk)

        for start in range(0, N_kept, eff_chunk):
            end = min(start + eff_chunk, N_kept)
            uv_c = uv_pixel[start:end]                                    # [C, 2]
            cn_c = conic[start:end]                                       # [C, 3]
            rgb_c = rgbs_v[start:end]                                     # [C, 3]
            opa_c = opa_v[start:end]                                      # [C]

            # Compute α[c, y, x] for every gaussian × every pixel in chunk.
            # Δ = pixel - center.
            dx = gx.unsqueeze(0) - uv_c[:, 0].view(-1, 1, 1)               # [C, H, W]
            dy = gy.unsqueeze(0) - uv_c[:, 1].view(-1, 1, 1)               # [C, H, W]
            # Mahalanobis (with conic = Σ⁻¹):
            #   d² = a·dx² + 2·b·dx·dy + c·dy²
            a = cn_c[:, 0].view(-1, 1, 1)
            b = cn_c[:, 1].view(-1, 1, 1)
            c = cn_c[:, 2].view(-1, 1, 1)
            mahal = a * dx * dx + 2 * b * dx * dy + c * dy * dy
            alpha = opa_c.view(-1, 1, 1) * torch.exp(-0.5 * mahal)
            alpha = alpha.clamp(0.0, 0.99)

            # Front-to-back compositing within this chunk: sequential per
            # gaussian (we can't parallelize the recursion without breaking
            # ordering). C is bounded by max_chunk, so this loop is short.
            for i in range(alpha.shape[0]):
                a_i = alpha[i]                                            # [H, W]
                contrib = T * a_i
                accum = accum + contrib.unsqueeze(-1) * rgb_c[i].view(1, 1, 3)
                T = T * (1.0 - a_i)
                # Early termination on saturated pixels.
                if torch.all(T < 1e-4):
                    break

        # Add background where transmittance survived.
        accum = accum + T.unsqueeze(-1) * background.view(1, 1, 3)
        out_images.append(accum.clamp(0.0, 1.0))

    return torch.stack(out_images, dim=0)                                  # [V, H, W, 3]


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class PreviewGaussianCamera:
    """Render a 3DGS PLY from given camera poses to an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a 3DGS PLY (HYWM2GaussianTrain.ply_path, "
                               "SharpPredictGaussiansFromMetricDepth, "
                               "MergeGaussians, or any external 3DGS tool).",
                }),
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "[N, 4, 4] world-to-camera. Renders N frames.",
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "[N, 3, 3] or [3, 3] pixel-K (or normalized-K, "
                               "auto-rescaled). Single [3,3] is broadcast to "
                               "all N views.",
                }),
                "image_width": ("INT", {
                    "default": 1024, "min": 32, "max": 8192, "step": 8,
                }),
                "image_height": ("INT", {
                    "default": 1024, "min": 32, "max": 8192, "step": 8,
                }),
            },
            "optional": {
                "near_plane": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 100.0, "step": 0.01,
                    "tooltip": "Near-plane culling distance in world units. "
                               "Gaussians with depth < near_plane are dropped.",
                }),
                "background": (["black", "white"], {
                    "default": "black",
                    "tooltip": "Background color where transmittance survives "
                               "all the gaussians.",
                }),
                "force_backend": (["auto", "gsplat", "torch"], {
                    "default": "auto",
                    "tooltip": "auto = try gsplat (fast CUDA); fall back to "
                               "pure-torch on ImportError or non-CUDA device. "
                               "gsplat = error if not available (use to verify "
                               "fast path). torch = always use the pure-PyTorch "
                               "fallback (slower; useful for AMD/Mac/CPU debug).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "GaussianPack"

    def render(
        self,
        ply_path: str,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
        near_plane: float = 0.01,
        background: str = "black",
        force_backend: str = "auto",
    ):
        if not ply_path or not Path(ply_path).is_file():
            raise FileNotFoundError(
                f"PreviewGaussianCamera: ply_path not found: {ply_path!r}"
            )

        # Pick device. Prefer CUDA if torch.cuda is available; else CPU.
        # MPS / ROCm autoselect via torch on those installs.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        _p(f"device={device}")

        # Load splat dict.
        splat = _load_3dgs_ply(ply_path)
        N_gauss = int(splat["means"].shape[0])

        # Normalize input shapes.
        ext = extrinsics.detach().float()
        if ext.dim() == 4 and ext.shape[0] == 1:
            ext = ext[0]
        if ext.dim() == 2:
            ext = ext.unsqueeze(0)
        if ext.dim() != 3 or ext.shape[-2:] != (4, 4):
            raise ValueError(
                f"PreviewGaussianCamera: extrinsics must be [N, 4, 4]; got "
                f"{tuple(extrinsics.shape)}"
            )
        V = int(ext.shape[0])

        intr = intrinsics.detach().float()
        if intr.dim() == 4 and intr.shape[0] == 1:
            intr = intr[0]
        if intr.dim() == 2:
            intr = intr.unsqueeze(0).expand(V, 3, 3).contiguous()
        if intr.dim() != 3 or intr.shape[-2:] != (3, 3):
            raise ValueError(
                f"PreviewGaussianCamera: intrinsics must be [N, 3, 3] or "
                f"[3, 3]; got {tuple(intrinsics.shape)}"
            )
        if intr.shape[0] == 1 and V > 1:
            intr = intr.expand(V, 3, 3).contiguous()
        if intr.shape[0] != V:
            raise ValueError(
                f"PreviewGaussianCamera: intrinsics N={intr.shape[0]} != "
                f"extrinsics N={V}"
            )

        # Convert normalized-K (PanoPack convention, fx<2) to pixel-K.
        intr_pixel = _normalize_K_to_pixel(intr, int(image_width), int(image_height))

        bg_color = (
            torch.zeros(3, dtype=torch.float32)
            if background == "black"
            else torch.ones(3, dtype=torch.float32)
        )
        bg_color = bg_color.to(device)

        _p(
            f"rendering {V} view(s) @ {image_width}×{image_height} "
            f"({N_gauss} gaussians)"
        )

        # Ask comfy to evict sibling-worker patchers so we have VRAM
        # headroom for the rasterizer. Estimate: roughly N_gauss × H × W
        # × 1e-3 bytes — empirically covers the gsplat per-view
        # workspace (tile-intersection buffers + per-pixel accumulators)
        # plus our own splat tensors with a comfortable margin. For 5.5M
        # gauss @ 1024² this is ~5.8 GB. Helper handles the cross-worker
        # IPC + local mm.free_memory + empty_cache.
        peak_bytes = int(N_gauss * int(image_height) * int(image_width) * 1e-3)
        _p(f"  -> requesting {peak_bytes / 1e9:.2f} GB VRAM headroom")
        _request_vram_eviction(peak_bytes)

        # Render: tiered backend selection.
        used_backend = None
        if force_backend == "gsplat":
            images = _render_gsplat(
                splat, ext, intr_pixel,
                int(image_height), int(image_width),
                float(near_plane), bg_color, device,
            )
            used_backend = "gsplat"
        elif force_backend == "torch":
            images = _render_torch(
                splat, ext, intr_pixel,
                int(image_height), int(image_width),
                float(near_plane), bg_color, device,
            )
            used_backend = "torch"
        else:
            # auto
            try:
                images = _render_gsplat(
                    splat, ext, intr_pixel,
                    int(image_height), int(image_width),
                    float(near_plane), bg_color, device,
                )
                used_backend = "gsplat"
            except ImportError as e:
                _p(f"gsplat unavailable ({e}); using torch fallback")
                images = _render_torch(
                    splat, ext, intr_pixel,
                    int(image_height), int(image_width),
                    float(near_plane), bg_color, device,
                )
                used_backend = "torch"

        _p(f"done (backend={used_backend})")

        # IMAGE convention is [B, H, W, 3] in [0, 1] on CPU/float32.
        return (images.detach().cpu().float().contiguous(),)


NODE_CLASS_MAPPINGS = {"PreviewGaussianCamera": PreviewGaussianCamera}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewGaussianCamera": "Preview Gaussian Camera",
}
