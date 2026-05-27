# SPDX-License-Identifier: GPL-3.0-or-later

"""Real spherical harmonic rotation via Ivanic-Ruedenberg recursion.

Used by TransformGaussian to rotate the SH AC coefficients (f_rest_*)
on a 3DGS PLY so view-dependent highlights track correctly through a
global rotation.

Reference:
    Ivanic & Ruedenberg, "Rotation Matrices for Real Spherical
    Harmonics. Direct Determination by Recursion", J. Phys. Chem.
    1996, 100, 6342-6347 + 1998 erratum.

Conventions:
    - Real SH basis, m ordered from -l to +l (mathematical order).
    - Band 1 mapping matches 3DGS / Inria:
          (m=-1, 0, +1)  <->  (Y, Z, X) basis directions.
    - For point rotation x' = R x, SH coefficients of a function
      transform as c'[n] = sum_m c[m] D_l[m, n]   (i.e., c' = c @ D_l).

3DGS PLY layout for max_sh_degree=3:
    - 15 SH AC coefficients per channel, layout:
          sh_idx  0..2  -> band l=1 (3 coeffs, m=-1,0,+1)
          sh_idx  3..7  -> band l=2 (5 coeffs, m=-2,-1,0,+1,+2)
          sh_idx  8..14 -> band l=3 (7 coeffs, m=-3,-2,-1,0,+1,+2,+3)
    - Start index for band l within the AC list is `l**2 - 1`.
    - Channel order is channel-major: f_rest_0..K-2 = R, then G, then B.
"""

from __future__ import annotations

import numpy as np


# -------------------------------------------------------------------
# Ivanic-Ruedenberg recursion
# -------------------------------------------------------------------

def _M(D, m, n):
    """Index D_l matrix by mathematical (m, n) instead of array (m+l, n+l).
    Works for any l since the matrix shape implies it.
    """
    l = (D.shape[0] - 1) // 2
    return D[m + l, n + l]


def _P(R1, D_prev, i, l, a, b):
    """Auxiliary P_i^{l, a, b} from Ivanic-Ruedenberg.

    i is one of {-1, 0, +1}; selects an R1 column.
    a is the band-l 'row' index; b is the band-(l-1) 'column' index.
    """
    # R1 is the band-1 rotation, indexed by m, m' in {-1, 0, +1}.
    # In 3DGS basis (m=-1 -> y, m=0 -> z, m=+1 -> x).
    if b == l:
        return _M(R1, i, 1) * _M(D_prev, a, l - 1) \
               - _M(R1, i, -1) * _M(D_prev, a, -(l - 1))
    elif b == -l:
        return _M(R1, i, 1) * _M(D_prev, a, -(l - 1)) \
               + _M(R1, i, -1) * _M(D_prev, a, l - 1)
    else:
        return _M(R1, i, 0) * _M(D_prev, a, b)


def _U(R1, D_prev, l, m, n):
    return _P(R1, D_prev, 0, l, m, n)


def _V(R1, D_prev, l, m, n):
    if m == 0:
        return _P(R1, D_prev, 1, l, 1, n) + _P(R1, D_prev, -1, l, -1, n)
    elif m > 0:
        d = 1 if m == 1 else 0
        return _P(R1, D_prev,  1, l,  m - 1, n) * np.sqrt(1 + d) \
             - _P(R1, D_prev, -1, l, -m + 1, n) * (1 - d)
    else:  # m < 0
        d = 1 if m == -1 else 0
        return _P(R1, D_prev,  1, l,  m + 1, n) * (1 - d) \
             + _P(R1, D_prev, -1, l, -m - 1, n) * np.sqrt(1 + d)


def _W(R1, D_prev, l, m, n):
    if m == 0:
        # By construction W vanishes at m=0; the recursion doesn't use it.
        return 0.0
    elif m > 0:
        return _P(R1, D_prev,  1, l,  m + 1, n) \
             + _P(R1, D_prev, -1, l, -m - 1, n)
    else:
        return _P(R1, D_prev,  1, l,  m - 1, n) \
             - _P(R1, D_prev, -1, l, -m + 1, n)


def _uvw(l, m, n):
    """Ivanic-Ruedenberg normalization coefficients u, v, w for entry (m, n)."""
    if abs(n) < l:
        denom = (l + n) * (l - n)
    else:
        denom = (2 * l) * (2 * l - 1)

    d = 1 if m == 0 else 0
    u = np.sqrt((l + m) * (l - m) / denom)
    v =  0.5 * np.sqrt((1 + d) * (l + abs(m) - 1) * (l + abs(m)) / denom) * (1 - 2 * d)
    if abs(m) < l:
        w = -0.5 * np.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d)
    else:
        w = 0.0
    return u, v, w


def real_sh_rotation_matrices(R: np.ndarray, lmax: int) -> list[np.ndarray]:
    """Compute real-SH rotation matrices D_l for l = 0..lmax inclusive.

    Args:
        R: 3x3 rotation matrix in standard Cartesian (x, y, z) basis.
        lmax: maximum SH band to compute (3DGS uses 3 at most).

    Returns:
        List of (2l+1, 2l+1) numpy arrays, indexed by l. D_l acts on
        coefficient vectors in (m = -l, ..., +l) order via c' = c @ D_l.
    """
    R = np.asarray(R, dtype=np.float64)
    Ds: list[np.ndarray] = [np.array([[1.0]])]
    if lmax == 0:
        return Ds

    # Band-1: R rebased from (x, y, z) -> (y, z, x).
    # In the 3DGS convention Y_1^{-1} ~ y, Y_1^0 ~ z, Y_1^{+1} ~ x.
    R1 = np.array([
        [R[1, 1], R[1, 2], R[1, 0]],
        [R[2, 1], R[2, 2], R[2, 0]],
        [R[0, 1], R[0, 2], R[0, 0]],
    ], dtype=np.float64)
    Ds.append(R1)
    if lmax == 1:
        return Ds

    # Bands l >= 2 via Ivanic-Ruedenberg.
    for l in range(2, lmax + 1):
        D_prev = Ds[l - 1]
        size = 2 * l + 1
        D = np.zeros((size, size), dtype=np.float64)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                u, v, w = _uvw(l, m, n)
                val = 0.0
                if u != 0.0:
                    val += u * _U(R1, D_prev, l, m, n)
                if v != 0.0:
                    val += v * _V(R1, D_prev, l, m, n)
                if w != 0.0:
                    val += w * _W(R1, D_prev, l, m, n)
                D[m + l, n + l] = val
        Ds.append(D)
    return Ds


# -------------------------------------------------------------------
# 3DGS PLY layout helpers
# -------------------------------------------------------------------

def sh_degree_from_n_ac(n_ac_per_channel: int) -> int:
    """How many SH bands (excluding DC) are encoded in N AC coefficients?

    K_AC = (sh_degree+1)**2 - 1   =>   sh_degree = sqrt(K_AC + 1) - 1
    Returns 0 if n_ac_per_channel is 0 (DC only, no rotation needed).
    """
    if n_ac_per_channel <= 0:
        return 0
    total = n_ac_per_channel + 1
    deg = int(round(np.sqrt(total))) - 1
    if (deg + 1) ** 2 != total:
        raise ValueError(
            f"N AC coefficients per channel = {n_ac_per_channel} does not "
            f"form a complete SH AC set ((sh_deg+1)^2 - 1). "
            f"3DGS supports sh_deg=1 (3 AC), sh_deg=2 (8 AC), sh_deg=3 (15 AC)."
        )
    return deg


def rotate_sh_ac(f_rest: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate SH AC coefficients of a 3DGS PLY by R.

    Args:
        f_rest: shape (N, 3, K_AC). Channel-major (R, G, B), then SH
                coefficients in the standard 3DGS / Inria order:
                    [l=1 m=-1, l=1 m=0, l=1 m=+1, l=2 m=-2, ..., l=3 m=+3]
        R: 3x3 rotation matrix in Cartesian (x, y, z) basis.

    Returns:
        Rotated f_rest, same shape.
    """
    if f_rest.size == 0:
        return f_rest.copy()
    N, C, K_AC = f_rest.shape
    assert C == 3, f"expected 3 channels, got {C}"
    sh_deg = sh_degree_from_n_ac(K_AC)
    Ds = real_sh_rotation_matrices(R, sh_deg)

    out = f_rest.astype(np.float32, copy=True)
    for l in range(1, sh_deg + 1):
        start = l * l - 1   # 1-1=0, 4-1=3, 9-1=8
        end = start + (2 * l + 1)
        D_l = Ds[l].astype(np.float32)
        # band has shape (N, 3, 2l+1); rotate with c' = c @ D_l.
        out[..., start:end] = out[..., start:end] @ D_l
    return out


# -------------------------------------------------------------------
# Self-test (run module directly: python -m nodes._sh_rotation)
# -------------------------------------------------------------------

def _test():
    """Sanity checks. Raises AssertionError on any failure."""

    def rx(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def rz(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    # --- 1. Identity rotation -> identity matrices for every band ---
    Ds = real_sh_rotation_matrices(np.eye(3), 3)
    for l, D in enumerate(Ds):
        assert np.allclose(D, np.eye(2 * l + 1), atol=1e-10), \
            f"identity rotation failed at l={l}: D=\n{D}"

    # --- 2. 180° around Z: D_2 should equal diag(1, -1, 1, -1, 1) ---
    #
    # Real l=2 SH basis (in m=-2..+2 order):
    #     m=-2: ∝ xy        — invariant under (x→-x, y→-y)
    #     m=-1: ∝ yz        — flips sign
    #     m= 0: ∝ 3z²-r²    — invariant
    #     m=+1: ∝ xz        — flips sign
    #     m=+2: ∝ x²-y²     — invariant
    Ds = real_sh_rotation_matrices(rz(np.pi), 3)
    expected_D2 = np.diag([1.0, -1.0, 1.0, -1.0, 1.0])
    assert np.allclose(Ds[2], expected_D2, atol=1e-10), \
        f"180°-Z D_2 mismatch:\n{Ds[2]}\nexpected:\n{expected_D2}"

    # --- 3. D_1 for 180°-Z: x->-x, y->-y, z->z.
    # In SH (m=-1, 0, +1) <-> (y, z, x) basis: diag(-1, 1, -1).
    assert np.allclose(Ds[1], np.diag([-1, 1, -1]), atol=1e-10), \
        f"180°-Z D_1 mismatch:\n{Ds[1]}"

    # --- 4. Orthogonality: D_l must be orthogonal for any l ---
    for R in (rx(0.7), ry(1.3), rz(-0.4), rx(0.7) @ ry(1.3) @ rz(-0.4)):
        Ds = real_sh_rotation_matrices(R, 3)
        for l, D in enumerate(Ds):
            err = np.abs(D @ D.T - np.eye(2 * l + 1)).max()
            assert err < 1e-9, f"D_{l} not orthogonal for R, max |D D^T - I| = {err}"

    # --- 5. Composition: D_l(R1 R2) == D_l(R1) @ D_l(R2) ---
    R1 = rx(0.4) @ rz(0.8)
    R2 = ry(-0.6) @ rx(0.3)
    Dcomp = real_sh_rotation_matrices(R1 @ R2, 3)
    Da = real_sh_rotation_matrices(R1, 3)
    Db = real_sh_rotation_matrices(R2, 3)
    for l in range(1, 4):
        err = np.abs(Dcomp[l] - Da[l] @ Db[l]).max()
        assert err < 1e-9, f"composition failed at l={l}, max err {err}"

    # --- 6. rotate_sh_ac round-trip: rotate then unrotate is identity ---
    rng = np.random.default_rng(42)
    f_rest = rng.standard_normal((50, 3, 15)).astype(np.float32)  # sh_deg=3
    R = rx(0.4) @ ry(-0.7) @ rz(1.1)
    rotated = rotate_sh_ac(f_rest, R)
    unrotated = rotate_sh_ac(rotated, R.T)
    err = np.abs(unrotated - f_rest).max()
    assert err < 1e-4, f"round-trip max err {err}"

    print("real-SH rotation self-test: all 6 checks passed ✓")


if __name__ == "__main__":
    _test()
