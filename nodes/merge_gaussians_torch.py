# SPDX-License-Identifier: GPL-3.0-or-later

"""PyTorch backend for MPMM Gaussian splat decimation.

Drop-in replacement for NanoGS's simplify() that runs kNN, edge costs,
and moment-matching merge on GPU (CUDA / ROCm / MPS) or accelerated CPU.
The greedy pair selection stays on CPU (inherently sequential, but cheap
after edge costs are sorted).

Only dependency beyond the standard library is torch, which ComfyUI
already provides.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

log = logging.getLogger("comfyui-gaussianpack")


def _progress(msg: str) -> None:
    """Print to the worker's stdout (forwarded to the ComfyUI console).

    log.info() is swallowed at the worker's default level, so progress for
    the long-running merge has to go through print(..., flush=True).
    """
    print(f"[GaussianMerge:torch_gpu] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Quaternion -> rotation matrix  (batched, torch)
# ---------------------------------------------------------------------------

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(B,4) wxyz -> (B,3,3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx),
        2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(B,3,3) -> (B,4) wxyz quaternion (Shepperd method)."""
    B = R.shape[0]
    q = torch.empty(B, 4, device=R.device, dtype=R.dtype)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    mask_tr = tr > 0
    mask_00 = (~mask_tr) & (m00 > m11) & (m00 > m22)
    mask_11 = (~mask_tr) & (~mask_00) & (m11 > m22)
    mask_22 = (~mask_tr) & (~mask_00) & (~mask_11)

    if mask_tr.any():
        S = torch.sqrt(tr[mask_tr] + 1.0) * 2.0
        q[mask_tr, 0] = 0.25 * S
        q[mask_tr, 1] = (R[mask_tr, 2, 1] - R[mask_tr, 1, 2]) / S
        q[mask_tr, 2] = (R[mask_tr, 0, 2] - R[mask_tr, 2, 0]) / S
        q[mask_tr, 3] = (R[mask_tr, 1, 0] - R[mask_tr, 0, 1]) / S
    if mask_00.any():
        S = torch.sqrt(1 + R[mask_00, 0, 0] - R[mask_00, 1, 1] - R[mask_00, 2, 2]) * 2.0
        q[mask_00, 0] = (R[mask_00, 2, 1] - R[mask_00, 1, 2]) / S
        q[mask_00, 1] = 0.25 * S
        q[mask_00, 2] = (R[mask_00, 0, 1] + R[mask_00, 1, 0]) / S
        q[mask_00, 3] = (R[mask_00, 0, 2] + R[mask_00, 2, 0]) / S
    if mask_11.any():
        S = torch.sqrt(1 + R[mask_11, 1, 1] - R[mask_11, 0, 0] - R[mask_11, 2, 2]) * 2.0
        q[mask_11, 0] = (R[mask_11, 0, 2] - R[mask_11, 2, 0]) / S
        q[mask_11, 1] = (R[mask_11, 0, 1] + R[mask_11, 1, 0]) / S
        q[mask_11, 2] = 0.25 * S
        q[mask_11, 3] = (R[mask_11, 1, 2] + R[mask_11, 2, 1]) / S
    if mask_22.any():
        S = torch.sqrt(1 + R[mask_22, 2, 2] - R[mask_22, 0, 0] - R[mask_22, 1, 1]) * 2.0
        q[mask_22, 0] = (R[mask_22, 1, 0] - R[mask_22, 0, 1]) / S
        q[mask_22, 1] = (R[mask_22, 0, 2] + R[mask_22, 2, 0]) / S
        q[mask_22, 2] = (R[mask_22, 1, 2] + R[mask_22, 2, 1]) / S
        q[mask_22, 3] = 0.25 * S

    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return q


# ---------------------------------------------------------------------------
# kNN - scipy cKDTree (O(N log N), always fastest for 3D)
# ---------------------------------------------------------------------------

def _knn_scipy(points_np: np.ndarray, k: int) -> np.ndarray:
    """(N, 3) float32 -> (N, k) int32 neighbor indices via cKDTree."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points_np)
    _, idx = tree.query(points_np, k=k + 1, workers=-1)
    return idx[:, 1:].astype(np.int32)


# ---------------------------------------------------------------------------
# Edge cost computation (torch port of nanogs full_cost_pairs)
# ---------------------------------------------------------------------------

def _gauss_logpdf_diagrot(
    x: torch.Tensor,         # (B, S, 3)
    mu: torch.Tensor,        # (B, 3)
    R: torch.Tensor,         # (B, 3, 3)
    invdiag: torch.Tensor,   # (B, 3)
    logdet: torch.Tensor,    # (B,)
) -> torch.Tensor:
    log2pi = math.log(2.0 * math.pi)
    d = x - mu.unsqueeze(1)                          # (B, S, 3)
    y = torch.matmul(d, R)                            # (B, S, 3)
    quad = (y * y * invdiag.unsqueeze(1)).sum(dim=2)  # (B, S)
    return -0.5 * (3.0 * log2pi + logdet.unsqueeze(1) + quad)


@torch.no_grad()
def _edge_costs(
    edges: torch.Tensor,  # (M, 2) long
    mu: torch.Tensor,     # (N, 3)
    sc: torch.Tensor,     # (N, 3)
    q: torch.Tensor,      # (N, 4) wxyz
    op: torch.Tensor,     # (N,)
    sh: torch.Tensor,     # (N, C)
    lam_geo: float = 1.0,
    lam_sh: float = 1.0,
    eps_cov: float = 1e-8,
    block: int = 500_000,
) -> torch.Tensor:
    """Compute MPMM merge cost for each edge. Returns (M,) float32."""
    M = edges.shape[0]
    device = mu.device
    costs = torch.empty(M, device=device, dtype=torch.float32)

    # precompute rotation matrices once
    R_all = _quat_to_rotmat(q)  # (N, 3, 3)

    for e0 in range(0, M, block):
        e1 = min(M, e0 + block)
        u = edges[e0:e1, 0]
        v = edges[e0:e1, 1]

        mu_i, mu_j = mu[u], mu[v]
        sc_i, sc_j = sc[u], sc[v]
        op_i, op_j = op[u], op[v]
        R_i, R_j = R_all[u], R_all[v]
        Rt_i = R_i.transpose(1, 2)
        Rt_j = R_j.transpose(1, 2)

        v_i = sc_i * sc_i + eps_cov
        v_j = sc_j * sc_j + eps_cov

        invdiag_i = 1.0 / v_i.clamp(min=1e-30)
        invdiag_j = 1.0 / v_j.clamp(min=1e-30)

        logdet_i = v_i.clamp(min=1e-30).log().sum(dim=1)
        logdet_j = v_j.clamp(min=1e-30).log().sum(dim=1)

        # mixture weights
        w_i = (2 * math.pi) ** 1.5 * op_i * sc_i.prod(dim=1) + 1e-12
        w_j = (2 * math.pi) ** 1.5 * op_j * sc_j.prod(dim=1) + 1e-12
        W = w_i + w_j
        W_safe = W.clamp(min=1e-12)
        pi = (w_i / W_safe).clamp(1e-12, 1.0 - 1e-12)
        log_pi = pi.log()
        log_pj = (1.0 - pi).log()

        # merged mean
        mu_m = pi.unsqueeze(1) * mu_i + (1.0 - pi).unsqueeze(1) * mu_j

        di = mu_i - mu_m
        dj = mu_j - mu_m
        odi = di.unsqueeze(2) * di.unsqueeze(1)  # (B, 3, 3)
        odj = dj.unsqueeze(2) * dj.unsqueeze(1)

        # full covariances
        Sig_i = torch.matmul(R_i * v_i.unsqueeze(1), Rt_i)
        Sig_j = torch.matmul(R_j * v_j.unsqueeze(1), Rt_j)

        Sig_m = pi[:, None, None] * (Sig_i + odi) + (1.0 - pi)[:, None, None] * (Sig_j + odj)
        I3 = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
        Sig_m = 0.5 * (Sig_m + Sig_m.transpose(1, 2)) + eps_cov * I3

        logdet_m = torch.linalg.slogdet(Sig_m).logabsdet

        # KL(p_mix || q_merge) via MC
        E_p_neglogq = 0.5 * (3.0 * math.log(2 * math.pi) + logdet_m + 3.0)

        # deterministic sample (single MC draw, matching nanogs default n_mc=1)
        Z = torch.zeros(1, 3, device=device, dtype=torch.float32)

        std_i = v_i.clamp(min=0).sqrt()
        std_j = v_j.clamp(min=0).sqrt()

        Zi = Z.unsqueeze(0) * std_i.unsqueeze(1)  # (B, 1, 3)
        Zj = Z.unsqueeze(0) * std_j.unsqueeze(1)

        x_i = mu_i.unsqueeze(1) + torch.matmul(Zi, Rt_i)  # (B, 1, 3)
        x_j = mu_j.unsqueeze(1) + torch.matmul(Zj, Rt_j)

        logNi_on_i = _gauss_logpdf_diagrot(x_i, mu_i, R_i, invdiag_i, logdet_i)
        logNj_on_i = _gauss_logpdf_diagrot(x_i, mu_j, R_j, invdiag_j, logdet_j)
        logp_on_i = torch.logaddexp(log_pi.unsqueeze(1) + logNi_on_i,
                                     log_pj.unsqueeze(1) + logNj_on_i)
        Ei = logp_on_i.mean(dim=1)

        logNi_on_j = _gauss_logpdf_diagrot(x_j, mu_i, R_i, invdiag_i, logdet_i)
        logNj_on_j = _gauss_logpdf_diagrot(x_j, mu_j, R_j, invdiag_j, logdet_j)
        logp_on_j = torch.logaddexp(log_pi.unsqueeze(1) + logNi_on_j,
                                     log_pj.unsqueeze(1) + logNj_on_j)
        Ej = logp_on_j.mean(dim=1)

        E_p_logp = pi * Ei + (1.0 - pi) * Ej
        geo = E_p_logp + E_p_neglogq

        # SH L2
        if sh.shape[1] > 0:
            diff = sh[u] - sh[v]
            c_sh = (diff * diff).sum(dim=1)
        else:
            c_sh = torch.zeros_like(geo)

        costs[e0:e1] = lam_geo * geo + lam_sh * c_sh

    return costs


# ---------------------------------------------------------------------------
# Edges from kNN (torch, then to numpy for greedy)
# ---------------------------------------------------------------------------

def _knn_to_edges(nbr: np.ndarray) -> np.ndarray:
    """(N, k) int32 -> (M, 2) int32 undirected unique edges."""
    N, k = nbr.shape
    ii = np.repeat(np.arange(N, dtype=np.int64), k)
    jj = nbr.reshape(-1).astype(np.int64)
    u = np.minimum(ii, jj)
    v = np.maximum(ii, jj)
    mask = u != v
    u, v = u[mask], v[mask]
    # pack into single int64 for fast unique
    packed = u * np.int64(N) + v
    packed = np.unique(packed)
    edges = np.stack([packed // N, packed % N], axis=1)
    return edges.astype(np.int32)


# ---------------------------------------------------------------------------
# Greedy pair selection (CPU - inherently sequential)
# ---------------------------------------------------------------------------

def _greedy_pairs(
    edges: np.ndarray,  # (M, 2) int32
    w: np.ndarray,      # (M,) float32
    N: int,
    P: int | None,
) -> np.ndarray:
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    valid = np.isfinite(w)
    if not valid.any():
        return np.zeros((0, 2), dtype=np.int32)
    idx = np.nonzero(valid)[0]
    order = idx[np.argsort(w[idx], kind="mergesort")]
    used = np.zeros(N, dtype=bool)
    pairs = []
    for ei in order:
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        pairs.append((u, v))
        if P is not None and len(pairs) >= P:
            break
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


# ---------------------------------------------------------------------------
# Batched 3x3 symmetric eigendecomposition - avoids cuSOLVER entirely
# ---------------------------------------------------------------------------

def _batched_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose batched 3x3 SPD matrices on any device.

    Falls back to CPU numpy to avoid cuSOLVER batch-size issues.
    For typical merge counts (< 2M pairs) this takes < 1s.
    """
    A_np = A.detach().cpu().numpy()
    evals_np, evecs_np = np.linalg.eigh(A_np)
    evals = torch.from_numpy(evals_np.astype(np.float32)).to(A.device)
    evecs = torch.from_numpy(evecs_np.astype(np.float32)).to(A.device)
    return evals, evecs


# ---------------------------------------------------------------------------
# Moment-matching merge (torch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _moment_match(
    mu_i: torch.Tensor, sc_i: torch.Tensor, q_i: torch.Tensor,
    op_i: torch.Tensor, sh_i: torch.Tensor,
    mu_j: torch.Tensor, sc_j: torch.Tensor, q_j: torch.Tensor,
    op_j: torch.Tensor, sh_j: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    R_i = _quat_to_rotmat(q_i)
    R_j = _quat_to_rotmat(q_j)
    s2_i = sc_i * sc_i
    s2_j = sc_j * sc_j
    Sig_i = torch.matmul(R_i * s2_i.unsqueeze(1), R_i.transpose(1, 2))
    Sig_j = torch.matmul(R_j * s2_j.unsqueeze(1), R_j.transpose(1, 2))

    w_i = (2 * math.pi) ** 1.5 * op_i * sc_i.prod(dim=1) + 1e-12
    w_j = (2 * math.pi) ** 1.5 * op_j * sc_j.prod(dim=1) + 1e-12
    W = (w_i + w_j).clamp(min=1e-12)

    mu_m = (w_i.unsqueeze(1) * mu_i + w_j.unsqueeze(1) * mu_j) / W.unsqueeze(1)

    di = mu_i - mu_m
    dj = mu_j - mu_m
    odi = di.unsqueeze(2) * di.unsqueeze(1)
    odj = dj.unsqueeze(2) * dj.unsqueeze(1)

    Sig_m = (w_i[:, None, None] * (Sig_i + odi) + w_j[:, None, None] * (Sig_j + odj)) / W[:, None, None]
    I3 = torch.eye(3, device=mu_i.device, dtype=torch.float32).unsqueeze(0)
    Sig_m = 0.5 * (Sig_m + Sig_m.transpose(1, 2)) + 1e-8 * I3

    evals, evecs = _batched_eigh(Sig_m)
    evals = evals.clamp(min=1e-18)

    # sort descending
    order = evals.argsort(dim=1, descending=True)
    evals = evals.gather(1, order)
    evecs = evecs.gather(2, order.unsqueeze(1).expand_as(evecs))

    # enforce right-handed
    det = torch.linalg.det(evecs)
    flip = det < 0
    if flip.any():
        evecs[flip, :, 2] *= -1.0

    scales_m = evals.sqrt()
    quat_m = _rotmat_to_quat(evecs)
    op_m = op_i + op_j - op_i * op_j

    if sh_i.shape[1] > 0:
        sh_m = (w_i.unsqueeze(1) * sh_i + w_j.unsqueeze(1) * sh_j) / W.unsqueeze(1)
    else:
        sh_m = sh_i

    return mu_m, scales_m, quat_m, op_m, sh_m


# ---------------------------------------------------------------------------
# Merge pairs - select, merge, concatenate
# ---------------------------------------------------------------------------

@torch.no_grad()
def _merge_pairs(
    mu: torch.Tensor, sc: torch.Tensor, q: torch.Tensor,
    op: torch.Tensor, sh: torch.Tensor,
    pairs: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    if pairs.shape[0] == 0:
        return mu, sc, q, op, sh

    i = torch.from_numpy(pairs[:, 0]).long().to(mu.device)
    j = torch.from_numpy(pairs[:, 1]).long().to(mu.device)

    mu_m, sc_m, q_m, op_m, sh_m = _moment_match(
        mu[i], sc[i], q[i], op[i], sh[i],
        mu[j], sc[j], q[j], op[j], sh[j],
    )

    used = torch.zeros(mu.shape[0], dtype=torch.bool, device=mu.device)
    used[i] = True
    used[j] = True
    keep = ~used

    mu2 = torch.cat([mu[keep], mu_m])
    sc2 = torch.cat([sc[keep], sc_m])
    q2 = torch.cat([q[keep], q_m])
    op2 = torch.cat([op[keep], op_m])
    sh2 = torch.cat([sh[keep], sh_m]) if sh.shape[1] > 0 else torch.empty(mu2.shape[0], 0, device=mu.device)

    return mu2, sc2, q2, op2, sh2


# ---------------------------------------------------------------------------
# Opacity pruning
# ---------------------------------------------------------------------------

def _prune_by_opacity(
    mu: torch.Tensor, sc: torch.Tensor, q: torch.Tensor,
    op: torch.Tensor, sh: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, ...]:
    median_op = op.median().item()
    threshold = min(threshold, median_op)
    log.info("Opacity mean=%.5f median=%.5f threshold=%.4f",
             op.mean().item(), median_op, threshold)
    keep = op >= threshold
    log.info("After opacity pruning: %d -> %d", mu.shape[0], keep.sum().item())
    mu, sc, q, op = mu[keep], sc[keep], q[keep], op[keep]
    sh = sh[keep] if sh.shape[1] > 0 else torch.empty(mu.shape[0], 0, device=mu.device)
    return mu, sc, q, op, sh


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def simplify_torch(
    in_path: str,
    out_path: str,
    ratio: float,
    k: int = 16,
    merge_cap: float = 0.5,
    opacity_threshold: float = 0.1,
    lam_geo: float = 1.0,
    lam_sh: float = 1.0,
) -> None:
    """MPMM simplification using PyTorch (GPU-accelerated where available)."""
    from nanogs.utils.ply_utils import read_ply, store_ply

    device = _best_device()
    log.info("Torch MPMM backend: device=%s", device)
    _progress(f"backend device={device}")

    hdr, mu_np, op_np, sc_np, q_np, sh_np, app_names = read_ply(in_path)
    N0 = mu_np.shape[0]
    target = max(int(math.ceil(N0 * ratio)), 1)
    log.info("Loaded %d splats, target %d (ratio=%.4f)", N0, target, ratio)
    _progress(f"loaded {N0} splats, target {target} (ratio={ratio:.4f})")

    mu = torch.from_numpy(mu_np).to(device)
    sc = torch.from_numpy(sc_np).to(device)
    q = torch.from_numpy(q_np).to(device)
    op = torch.from_numpy(op_np).to(device)
    sh = torch.from_numpy(sh_np).to(device)

    n_before_prune = mu.shape[0]
    mu, sc, q, op, sh = _prune_by_opacity(mu, sc, q, op, sh, opacity_threshold)
    _progress(f"opacity prune: {n_before_prune} -> {mu.shape[0]} splats")

    p_cap = max(1, int(merge_cap * N0))
    iteration = 0

    while mu.shape[0] > target:
        N = mu.shape[0]
        # fraction of the way from start (N0) down to the target count
        done = (N0 - N) / max(1, N0 - target)
        log.info("Pass %d: %d splats", iteration + 1, N)
        _progress(f"pass {iteration + 1}: {N} splats -> target {target} ({done * 100:.0f}%)")

        k_eff = min(max(1, k), max(1, N - 1))
        _progress(f"  building kNN graph (k={k_eff})...")
        nbr = _knn_scipy(mu.cpu().numpy(), k=k_eff)
        edges_np = _knn_to_edges(nbr)
        edges_t = torch.from_numpy(edges_np).long().to(device)

        log.info("  Computing edge costs for %d edges...", edges_t.shape[0])
        _progress(f"  computing edge costs for {edges_t.shape[0]} edges...")
        w = _edge_costs(edges_t, mu, sc, q, op, sh,
                        lam_geo=lam_geo, lam_sh=lam_sh)
        w_np = w.cpu().numpy()

        merges_needed = N - target
        P = min(merges_needed, p_cap) if merges_needed > 0 else None

        _progress("  selecting merge pairs (greedy)...")
        pairs = _greedy_pairs(edges_np, w_np, N, P)
        log.info("  edges=%d pairs=%d (need %d)", edges_np.shape[0], pairs.shape[0], merges_needed)
        _progress(f"  merging {pairs.shape[0]} pairs (need {merges_needed})")

        mu, sc, q, op, sh = _merge_pairs(mu, sc, q, op, sh, pairs)
        iteration += 1

    log.info("Final: %d splats", mu.shape[0])
    _progress(f"done: {mu.shape[0]} splats after {iteration} pass(es), writing PLY...")

    # back to numpy for PLY writing
    mu_out = mu.cpu().numpy().astype(np.float32)
    sc_out = sc.cpu().numpy().astype(np.float32)
    q_out = q.cpu().numpy().astype(np.float32)
    op_out = op.cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    sh_out = sh.cpu().numpy().astype(np.float32)

    store_ply(out_path, hdr, mu_out, op_out, sc_out, q_out, sh_out, app_names)
