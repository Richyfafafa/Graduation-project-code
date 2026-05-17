import torch

from utils.bspline_fit import fit_bspline_control_points
from utils.bspline_projection import project_points_to_surface_grid


def select_mirror_gaussians(gaussians, mirror_weight_thr=0.5, refl_thr=-1.0):
    mirror_w = gaussians.get_mirror_weight.flatten()
    m = mirror_w > mirror_weight_thr
    if refl_thr >= 0.0:
        m = torch.logical_and(m, gaussians.get_refl.flatten() > refl_thr)
    return m


@torch.no_grad()
def init_uv_from_pca(points):
    center = points.mean(dim=0)
    x = points - center
    cov = x.transpose(0, 1) @ x / max(points.shape[0] - 1, 1)
    _evals, evecs = torch.linalg.eigh(cov)
    axis_u = evecs[:, 2]
    axis_v = evecs[:, 1]

    u_raw = x @ axis_u
    v_raw = x @ axis_v

    u_min, u_max = u_raw.min(), u_raw.max()
    v_min, v_max = v_raw.min(), v_raw.max()
    u = (u_raw - u_min) / (u_max - u_min + 1e-9)
    v = (v_raw - v_min) / (v_max - v_min + 1e-9)

    frame = {
        "center": center,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
    }
    return u, v, frame


@torch.no_grad()
def uv_from_frame(points, frame):
    x = points - frame["center"]
    u_raw = x @ frame["axis_u"]
    v_raw = x @ frame["axis_v"]
    u = (u_raw - frame["u_min"]) / (frame["u_max"] - frame["u_min"] + 1e-9)
    v = (v_raw - frame["v_min"]) / (frame["v_max"] - frame["v_min"] + 1e-9)
    return u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)


@torch.no_grad()
def fit_surface_from_mirror_points(
    points,
    weights=None,
    num_ctrl_u=6,
    num_ctrl_v=6,
    degree_u=3,
    degree_v=3,
    lambda_smooth=1e-3,
):
    u, v, frame = init_uv_from_pca(points)
    ctrl = fit_bspline_control_points(
        points,
        u,
        v,
        num_ctrl_u=num_ctrl_u,
        num_ctrl_v=num_ctrl_v,
        degree_u=degree_u,
        degree_v=degree_v,
        weights=weights,
        lambda_smooth=lambda_smooth,
    )
    return ctrl, frame


@torch.no_grad()
def build_surface_targets(points, ctrl, frame, degree_u=3, degree_v=3, grid_u=64, grid_v=64):
    u, v = uv_from_frame(points, frame)
    _, _, proj, dist = project_points_to_surface_grid(
        points,
        ctrl,
        degree_u=degree_u,
        degree_v=degree_v,
        grid_u=grid_u,
        grid_v=grid_v,
    )
    return u, v, proj, dist
