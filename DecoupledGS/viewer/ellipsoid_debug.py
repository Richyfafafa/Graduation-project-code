import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import diff_gaussian_rasterization_c3

from gaussian_renderer import GaussianModel
from scene.cameras import Camera, MiniCam
from utils.general_utils import build_rotation
from viewer.viewer_camera import build_projection_matrix_correct


def _scale_hwk(hwk, scale: float):
    h, w, k = hwk
    scale = float(min(1.0, max(scale, 1e-3)))
    if abs(scale - 1.0) < 1e-6:
        return (int(h), int(w), k.copy()), (int(w), int(h)), 1.0

    new_w = max(64, int(round(w * scale)))
    actual_scale = new_w / float(w)
    new_h = max(64, int(round(h * actual_scale)))
    actual_scale = new_h / float(h)
    new_w = max(64, int(round(w * actual_scale)))

    new_k = k.copy()
    new_k[0, :] *= actual_scale
    new_k[1, :] *= actual_scale
    return (new_h, new_w, new_k), (new_w, new_h), actual_scale


def build_ellipsoid_debug_cache(gaussians: GaussianModel):
    xyz = gaussians.get_xyz.detach()
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1).contiguous()
    scales = gaussians.get_scaling.detach()
    quat = gaussians.get_rotation.detach()
    rots = build_rotation(gaussians.get_rotation.detach())
    cov3d = rots @ torch.diag_embed(scales * scales) @ rots.transpose(1, 2)
    dc_color = (gaussians.get_features[:, 0, :].detach() * 0.2 + 0.5).clamp(0.0, 1.0)
    return {
        "xyz_h": xyz_h,
        "xyz": xyz.contiguous(),
        "scales": scales.contiguous(),
        "quat": quat.contiguous(),
        "rots": rots.contiguous(),
        "cov3d": cov3d.contiguous(),
        "opacity": gaussians.get_opacity.detach().squeeze(-1).contiguous(),
        "dc_color": dc_color.contiguous(),
    }


def _blend_debug_ellipse(canvas, cx, cy, rx, ry, angle_deg, color, alpha):
    h, w = canvas.shape[:2]
    pad = 3
    x0 = max(0, int(math.floor(cx - rx - pad)))
    x1 = min(w, int(math.ceil(cx + rx + pad)) + 1)
    y0 = max(0, int(math.floor(cy - ry - pad)))
    y1 = min(h, int(math.ceil(cy + ry + pad)) + 1)
    if x1 <= x0 or y1 <= y0:
        return

    patch = canvas[y0:y1, x0:x1]
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    center = (int(round(cx)) - x0, int(round(cy)) - y0)
    axes = (max(1, int(round(rx))), max(1, int(round(ry))))
    cv2.ellipse(mask, center, axes, angle_deg, 0, 360, 255, -1, lineType=cv2.LINE_8)

    alpha_mask = (mask.astype(np.float32) / 255.0) * float(alpha)
    patch *= 1.0 - alpha_mask[..., None]
    patch += color.reshape(1, 1, 3) * alpha_mask[..., None]


def _make_scaled_view(view: Camera, canvas_scale: float):
    h, w, _ = view.HWK
    canvas_scale = float(min(1.0, max(canvas_scale, 0.25)))
    scaled_hwk, _, actual_scale = _scale_hwk(view.HWK, canvas_scale)
    if abs(actual_scale - 1.0) < 1e-6:
        return view, (w, h), 1.0

    projection = build_projection_matrix_correct(
        view.znear,
        view.zfar,
        scaled_hwk[0],
        scaled_hwk[1],
        scaled_hwk[2],
    ).transpose(0, 1).to(device=view.world_view_transform.device, dtype=view.world_view_transform.dtype)
    full_proj = (
        view.world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0))
    ).squeeze(0)
    scaled_view = MiniCam(
        width=int(scaled_hwk[1]),
        height=int(scaled_hwk[0]),
        fovy=view.FoVy,
        fovx=view.FoVx,
        znear=view.znear,
        zfar=view.zfar,
        world_view_transform=view.world_view_transform,
        full_proj_transform=full_proj,
    )
    scaled_view.HWK = scaled_hwk
    return scaled_view, (w, h), actual_scale


def _project_ellipsoids(
    view: Camera,
    ellipsoid_cache,
    max_gaussians: int,
    sigma_scale: float,
    min_major_radius: float = 0.0,
    min_opacity: float = 0.0,
):
    h, w, k = view.HWK
    xyz_h = ellipsoid_cache["xyz_h"]
    cov3d = ellipsoid_cache["cov3d"]
    opacity = ellipsoid_cache["opacity"]

    w2c = view.world_view_transform.transpose(0, 1)
    cam_pts = xyz_h @ w2c.transpose(0, 1)
    cam_xyz = cam_pts[:, :3]
    z = cam_xyz[:, 2]
    visible = z > 0.05
    if not torch.any(visible):
        return None

    visible_idx = torch.nonzero(visible, as_tuple=False).squeeze(1)
    cam_xyz = cam_xyz[visible_idx]
    opacity = opacity[visible_idx]
    cov3d = cov3d[visible_idx]

    rot_wc = w2c[:3, :3]
    cov_cam = rot_wc.unsqueeze(0) @ cov3d @ rot_wc.transpose(0, 1).unsqueeze(0)

    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    z = cam_xyz[:, 2].clamp_min(1e-6)
    inv_z = z.reciprocal()
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    proj_j = torch.zeros((cam_xyz.shape[0], 2, 3), dtype=cam_xyz.dtype, device=cam_xyz.device)
    proj_j[:, 0, 0] = fx * inv_z
    proj_j[:, 0, 2] = -fx * x * inv_z * inv_z
    proj_j[:, 1, 1] = fy * inv_z
    proj_j[:, 1, 2] = -fy * y * inv_z * inv_z

    cov2d = proj_j @ cov_cam @ proj_j.transpose(1, 2)
    a = cov2d[:, 0, 0] + 1e-6
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + 1e-6
    trace = 0.5 * (a + c)
    radius_term = torch.sqrt(((a - c) * 0.5) ** 2 + b * b + 1e-12)
    eig_small = (trace - radius_term).clamp_min(1e-8)
    eig_large = (trace + radius_term).clamp_min(1e-8)
    minor_r = sigma_scale * torch.sqrt(eig_small)
    major_r = (sigma_scale * torch.sqrt(eig_large)).clamp_max(max(h, w) * 0.45)
    axis_angle = 0.5 * torch.atan2(2.0 * b, a - c)

    u = fx * x * inv_z + cx
    v = fy * y * inv_z + cy
    in_frame = (
        (u + major_r >= 0)
        & (u - major_r < w)
        & (v + major_r >= 0)
        & (v - major_r < h)
        & (major_r >= float(min_major_radius))
        & (opacity >= float(min_opacity))
    )
    if not torch.any(in_frame):
        return None

    frame_idx = torch.nonzero(in_frame, as_tuple=False).squeeze(1)
    if max_gaussians > 0 and frame_idx.numel() > max_gaussians:
        importance = (opacity[frame_idx] * major_r[frame_idx] * minor_r[frame_idx]).clamp_min(0.0)
        keep = torch.topk(importance, k=max_gaussians, largest=True, sorted=False).indices
        frame_idx = frame_idx[keep]

    draw_depth = z[frame_idx]
    draw_order = torch.argsort(draw_depth, descending=True)
    frame_idx = frame_idx[draw_order]
    source_idx = visible_idx[frame_idx]

    return {
        "source_idx": source_idx,
        "u": u[frame_idx],
        "v": v[frame_idx],
        "minor": minor_r[frame_idx],
        "major": major_r[frame_idx],
        "opacity": opacity[frame_idx],
        "angle": axis_angle[frame_idx],
    }


def _collect_projected_ellipsoids(
    view: Camera,
    ellipsoid_cache,
    max_gaussians: int,
    sigma_scale: float,
    min_major_radius: float = 0.0,
    min_opacity: float = 0.0,
):
    projected = _project_ellipsoids(
        view,
        ellipsoid_cache,
        max_gaussians=max_gaussians,
        sigma_scale=sigma_scale,
        min_major_radius=min_major_radius,
        min_opacity=min_opacity,
    )
    if projected is None:
        return None
    return {
        "source_idx": projected["source_idx"],
        "u": projected["u"].detach().cpu().numpy(),
        "v": projected["v"].detach().cpu().numpy(),
        "minor": projected["minor"].detach().cpu().numpy(),
        "major": projected["major"].detach().cpu().numpy(),
        "opacity": projected["opacity"].detach().cpu().numpy(),
        "angle": projected["angle"].detach().cpu().numpy(),
    }


def render_ellipsoid_debug_cpu(
    view: Camera,
    background: torch.Tensor,
    ellipsoid_cache,
    max_gaussians: int,
    sigma_scale: float,
    min_major_radius: float,
    min_opacity: float,
    scale_mult: float,
    canvas_scale: float,
) -> torch.Tensor:
    h, w, _ = view.HWK
    canvas_scale = float(min(1.0, max(canvas_scale, 0.25)))
    draw_w = max(64, int(round(w * canvas_scale)))
    draw_h = max(64, int(round(h * canvas_scale)))
    background_np = background.detach().cpu().numpy().astype(np.float32)
    canvas = np.empty((draw_h, draw_w, 3), dtype=np.float32)
    canvas[...] = background_np

    projected = _collect_projected_ellipsoids(
        view,
        ellipsoid_cache,
        max_gaussians=max_gaussians,
        sigma_scale=sigma_scale,
        min_major_radius=min_major_radius,
        min_opacity=min_opacity,
    )
    if projected is None:
        if draw_w != w or draw_h != h:
            canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(canvas).permute(2, 0, 1)

    draw_colors = ellipsoid_cache["dc_color"][projected["source_idx"]].cpu().numpy().astype(np.float32)

    for idx in range(projected["u"].shape[0]):
        color = draw_colors[idx]
        alpha = float(np.clip(0.20 + 0.70 * projected["opacity"][idx], 0.15, 0.90))
        angle_deg = float(np.degrees(projected["angle"][idx]))
        _blend_debug_ellipse(
            canvas,
            projected["u"][idx] * canvas_scale,
            projected["v"][idx] * canvas_scale,
            projected["major"][idx] * float(scale_mult) * canvas_scale,
            projected["minor"][idx] * float(scale_mult) * canvas_scale,
            angle_deg,
            color,
            alpha,
        )

    if draw_w != w or draw_h != h:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(canvas).permute(2, 0, 1)


def render_ellipsoid_debug_gpu(
    view: Camera,
    background: torch.Tensor,
    ellipsoid_cache,
    max_gaussians: int,
    sigma_scale: float,
    min_major_radius: float,
    min_opacity: float,
    scale_mult: float,
    canvas_scale: float,
) -> torch.Tensor:
    draw_view, output_wh, _ = _make_scaled_view(view, canvas_scale)
    draw_h, draw_w, _ = draw_view.HWK
    projected = _project_ellipsoids(
        draw_view,
        ellipsoid_cache,
        max_gaussians=max_gaussians,
        sigma_scale=sigma_scale,
        min_major_radius=min_major_radius,
        min_opacity=min_opacity,
    )
    bg_map = background[:, None, None].to(device="cuda").expand(3, draw_h, draw_w)
    if projected is None or projected["source_idx"].numel() == 0:
        image = bg_map
    else:
        source_idx = projected["source_idx"]
        means3D = ellipsoid_cache["xyz"][source_idx]
        opacities = ellipsoid_cache["opacity"][source_idx].unsqueeze(-1)
        scales = ellipsoid_cache["scales"][source_idx]
        rotations = ellipsoid_cache["quat"][source_idx]
        colors = ellipsoid_cache["dc_color"][source_idx]

        tanfovx = math.tan(draw_view.FoVx * 0.5)
        tanfovy = math.tan(draw_view.FoVy * 0.5)
        raster_settings = diff_gaussian_rasterization_c3.GaussianRasterizationSettings(
            image_height=int(draw_view.image_height),
            image_width=int(draw_view.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=float(scale_mult),
            viewmatrix=draw_view.world_view_transform,
            projmatrix=draw_view.full_proj_transform,
            sh_degree=0,
            campos=draw_view.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = diff_gaussian_rasterization_c3.GaussianRasterizer(raster_settings)
        means2D = torch.zeros_like(means3D)
        image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            bg_map=bg_map,
        )

    out_h, out_w = output_wh[1], output_wh[0]
    if draw_w != out_w or draw_h != out_h:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image
