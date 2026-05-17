import torch

from utils.bspline_surface import eval_surface_points


@torch.no_grad()
def project_points_to_surface_grid(
    points,
    ctrl,
    degree_u=3,
    degree_v=3,
    grid_u=64,
    grid_v=64,
):
    device = points.device
    dtype = points.dtype

    us = torch.linspace(0.0, 1.0, grid_u, device=device, dtype=dtype)
    vs = torch.linspace(0.0, 1.0, grid_v, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(us, vs, indexing="ij")
    u_flat = uu.reshape(-1)
    v_flat = vv.reshape(-1)
    surf = eval_surface_points(ctrl, u_flat, v_flat, degree_u=degree_u, degree_v=degree_v)  # [G,3]

    d2 = torch.cdist(points, surf, p=2.0) ** 2
    idx = torch.argmin(d2, dim=1)
    proj = surf[idx]
    u_proj = u_flat[idx]
    v_proj = v_flat[idx]
    dist = torch.sqrt(torch.clamp_min((points - proj).pow(2).sum(dim=-1), 1e-12))
    return u_proj, v_proj, proj, dist
