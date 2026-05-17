import torch


def make_open_uniform_knots(num_ctrl, degree, device=None, dtype=torch.float32):
    if num_ctrl <= degree:
        raise ValueError("num_ctrl must be greater than degree.")
    knot_count = num_ctrl + degree + 1
    knots = torch.zeros(knot_count, device=device, dtype=dtype)
    interior = num_ctrl - degree - 1
    if interior > 0:
        knots[degree + 1 : degree + 1 + interior] = torch.linspace(
            0.0, 1.0, interior + 2, device=device, dtype=dtype
        )[1:-1]
    knots[num_ctrl:] = 1.0
    return knots


def bspline_basis_matrix(t, num_ctrl, degree, knots=None):
    if knots is None:
        knots = make_open_uniform_knots(num_ctrl, degree, device=t.device, dtype=t.dtype)

    t = t.clamp(0.0, 1.0)
    n = t.shape[0]
    b = torch.zeros(n, num_ctrl, device=t.device, dtype=t.dtype)

    for i in range(num_ctrl):
        left = knots[i]
        right = knots[i + 1]
        mask = (t >= left) & (t < right)
        b[:, i] = mask.to(t.dtype)
    b[t >= 1.0, -1] = 1.0

    for p in range(1, degree + 1):
        b_new = torch.zeros_like(b)
        for i in range(num_ctrl):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            term1 = 0.0
            term2 = 0.0
            if float(denom1) > 1e-12:
                term1 = ((t - knots[i]) / denom1) * b[:, i]
            if i + 1 < num_ctrl and float(denom2) > 1e-12:
                term2 = ((knots[i + p + 1] - t) / denom2) * b[:, i + 1]
            b_new[:, i] = term1 + term2
        b = b_new
    return b


def eval_surface_points(ctrl, u, v, degree_u=3, degree_v=3, knots_u=None, knots_v=None):
    if ctrl.ndim != 3 or ctrl.shape[-1] != 3:
        raise ValueError("ctrl must have shape [num_ctrl_u, num_ctrl_v, 3].")
    bu = bspline_basis_matrix(u, ctrl.shape[0], degree_u, knots=knots_u)
    bv = bspline_basis_matrix(v, ctrl.shape[1], degree_v, knots=knots_v)
    tmp = torch.einsum("nu,uvc->nvc", bu, ctrl)
    pts = torch.einsum("nv,nvc->nc", bv, tmp)
    return pts


def eval_surface_normals_fd(ctrl, u, v, degree_u=3, degree_v=3, eps=1e-3):
    u0 = u.clamp(0.0, 1.0)
    v0 = v.clamp(0.0, 1.0)
    pu0 = eval_surface_points(ctrl, (u0 + eps).clamp(0.0, 1.0), v0, degree_u, degree_v)
    pu1 = eval_surface_points(ctrl, (u0 - eps).clamp(0.0, 1.0), v0, degree_u, degree_v)
    pv0 = eval_surface_points(ctrl, u0, (v0 + eps).clamp(0.0, 1.0), degree_u, degree_v)
    pv1 = eval_surface_points(ctrl, u0, (v0 - eps).clamp(0.0, 1.0), degree_u, degree_v)
    du = (pu0 - pu1) / (2.0 * eps)
    dv = (pv0 - pv1) / (2.0 * eps)
    n = torch.cross(du, dv, dim=-1)
    n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-9)
    return n
