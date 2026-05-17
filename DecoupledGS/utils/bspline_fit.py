import torch

from utils.bspline_surface import bspline_basis_matrix


def _build_design_matrix(bu, bv):
    # bu: [N, U], bv: [N, V] -> A: [N, U*V]
    return torch.einsum("nu,nv->nuv", bu, bv).reshape(bu.shape[0], -1)


def _build_laplacian_rows(num_ctrl_u, num_ctrl_v, device, dtype):
    rows = []

    def flat_idx(i, j):
        return i * num_ctrl_v + j

    for i in range(1, num_ctrl_u - 1):
        for j in range(num_ctrl_v):
            r = torch.zeros(num_ctrl_u * num_ctrl_v, device=device, dtype=dtype)
            r[flat_idx(i - 1, j)] = 1.0
            r[flat_idx(i, j)] = -2.0
            r[flat_idx(i + 1, j)] = 1.0
            rows.append(r)
    for i in range(num_ctrl_u):
        for j in range(1, num_ctrl_v - 1):
            r = torch.zeros(num_ctrl_u * num_ctrl_v, device=device, dtype=dtype)
            r[flat_idx(i, j - 1)] = 1.0
            r[flat_idx(i, j)] = -2.0
            r[flat_idx(i, j + 1)] = 1.0
            rows.append(r)

    if len(rows) == 0:
        return None
    return torch.stack(rows, dim=0)


def fit_bspline_control_points(
    points,
    u,
    v,
    num_ctrl_u=6,
    num_ctrl_v=6,
    degree_u=3,
    degree_v=3,
    weights=None,
    lambda_smooth=1e-3,
    lambda_reg=1e-6,
):
    if points.shape[0] < 8:
        raise ValueError("Need at least 8 points for stable B-spline fitting.")
    if num_ctrl_u <= degree_u or num_ctrl_v <= degree_v:
        raise ValueError("num_ctrl must be > degree in both axes.")

    device = points.device
    dtype = points.dtype

    bu = bspline_basis_matrix(u, num_ctrl_u, degree_u)
    bv = bspline_basis_matrix(v, num_ctrl_v, degree_v)
    a = _build_design_matrix(bu, bv)  # [N, M]
    n, m = a.shape

    if weights is None:
        w = torch.ones(n, device=device, dtype=dtype)
    else:
        w = weights.reshape(-1).to(device=device, dtype=dtype).clamp_min(1e-8)
    sw = torch.sqrt(w)
    aw = a * sw[:, None]
    pw = points * sw[:, None]

    ata = aw.transpose(0, 1) @ aw
    atp = aw.transpose(0, 1) @ pw

    if lambda_smooth > 0:
        l = _build_laplacian_rows(num_ctrl_u, num_ctrl_v, device=device, dtype=dtype)
        if l is not None and l.numel() > 0:
            ata = ata + lambda_smooth * (l.transpose(0, 1) @ l)

    ata = ata + lambda_reg * torch.eye(m, device=device, dtype=dtype)
    ctrl_flat = torch.linalg.solve(ata, atp)  # [M, 3]
    ctrl = ctrl_flat.reshape(num_ctrl_u, num_ctrl_v, 3)
    return ctrl
