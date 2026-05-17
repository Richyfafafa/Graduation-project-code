
import json
import os
import re
from argparse import ArgumentParser
from random import randint

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from train import prepare_output_and_logger
from utils.general_utils import PILtoTorch, safe_state


def restore_env_map_from_checkpoint_sidecar(gaussians, checkpoint_path):
    if checkpoint_path is None:
        return
    m = re.search(r"chkpnt(\d+)\.pth$", os.path.basename(checkpoint_path))
    if m is None:
        return
    iter_tag = int(m.group(1))
    map_path = os.path.join(
        os.path.dirname(checkpoint_path),
        "point_cloud",
        f"iteration_{iter_tag}",
        "point_cloud.map",
    )
    if not os.path.exists(map_path):
        return
    try:
        env_state = torch.load(map_path, map_location="cpu")
        gaussians.env_map.load_state_dict(env_state)
        print(f"[INFO] Restored env_map from {map_path}")
    except Exception as e:
        print(f"[WARN] Failed to restore env_map from {map_path}: {e}")


def load_homography_points(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"homography_points.json not found: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    mirror_pts = payload.get("mirror_points")
    bg_pts = payload.get("bg_points")
    if mirror_pts is None or bg_pts is None:
        raise ValueError("homography_points.json must contain mirror_points and bg_points")
    return mirror_pts, bg_pts


def compute_homography(mirror_pts, bg_pts):
    mirror = torch.tensor(mirror_pts, dtype=torch.float32).cpu().numpy()
    bg = torch.tensor(bg_pts, dtype=torch.float32).cpu().numpy()
    H, _ = cv2.findHomography(mirror, bg, method=0)
    if H is None:
        raise RuntimeError("Failed to compute homography from points.")
    return torch.tensor(H, dtype=torch.float32)


def build_u0_map(h, w, H_tensor, device):
    ys, xs = torch.meshgrid(torch.arange(h, device=device).float(),
                            torch.arange(w, device=device).float(),
                            indexing="ij")
    ones = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=-1)
    uvw = coords @ H_tensor.to(device).T
    u = uvw[..., 0] / (uvw[..., 2] + 1e-6)
    v = uvw[..., 1] / (uvw[..., 2] + 1e-6)
    return torch.stack([u, v], dim=-1)


def load_texture(texture_path, dataset_root, device):
    if texture_path is None:
        return None
    if not os.path.isabs(texture_path):
        texture_path = os.path.join(dataset_root, texture_path)
    if not os.path.exists(texture_path):
        raise FileNotFoundError(f"Background texture not found: {texture_path}")
    image = Image.open(texture_path).convert("RGB")
    tensor = PILtoTorch(image, image.size).float()
    return tensor.to(device)


def resolve_image_path(dataset_root, images_dir, frame_name):
    if frame_name is None:
        return None
    if os.path.isabs(frame_name):
        return frame_name
    candidate = os.path.join(dataset_root, images_dir, frame_name)
    if os.path.exists(candidate):
        return candidate
    fallback = os.path.join(dataset_root, frame_name)
    if os.path.exists(fallback):
        return fallback
    return frame_name


def render_mirror_mask(camera, gaussians, pipe):
    mirror_weights = gaussians.get_mirror_weight
    mirror_colors = mirror_weights.repeat(1, 3)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    render_pkg = render(camera, gaussians, pipe, background, override_color=mirror_colors)
    mask = render_pkg["render"][:1].clamp(0.0, 1.0)
    return mask


def compute_mask_loss(pred_mask, gt_mask, loss_type):
    pred_mask = pred_mask.clamp(1e-4, 1.0 - 1e-4)
    if loss_type == "bce":
        return F.binary_cross_entropy(pred_mask, gt_mask)
    if loss_type == "hybrid":
        bce = F.binary_cross_entropy(pred_mask, gt_mask)
        inter = (pred_mask * gt_mask).sum()
        dice = 1.0 - (2.0 * inter + 1e-6) / (pred_mask.sum() + gt_mask.sum() + 1e-6)

        sobel_x = torch.tensor([[[[1.0, 0.0, -1.0],
                                  [2.0, 0.0, -2.0],
                                  [1.0, 0.0, -1.0]]]], device=pred_mask.device)
        sobel_y = torch.tensor([[[[1.0, 2.0, 1.0],
                                  [0.0, 0.0, 0.0],
                                  [-1.0, -2.0, -1.0]]]], device=pred_mask.device)
        pred_gx = F.conv2d(pred_mask.unsqueeze(0), sobel_x, padding=1)
        pred_gy = F.conv2d(pred_mask.unsqueeze(0), sobel_y, padding=1)
        gt_gx = F.conv2d(gt_mask.unsqueeze(0), sobel_x, padding=1)
        gt_gy = F.conv2d(gt_mask.unsqueeze(0), sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_gx * pred_gx + pred_gy * pred_gy + 1e-6)
        gt_edge = torch.sqrt(gt_gx * gt_gx + gt_gy * gt_gy + 1e-6)
        edge_l1 = (pred_edge - gt_edge).abs().mean()

        return bce + 0.5 * dice + 0.1 * edge_l1
    return (pred_mask - gt_mask).abs().mean()


def fit_mirror_weights(scene, gaussians, pipe, args):
    cameras = [cam for cam in scene.getTrainCameras() if cam.spec_mask is not None]
    if not cameras:
        raise RuntimeError("No masks found in training cameras; cannot fit mirror weights.")

    # Freeze base parameters and optimize only the mirror mask weights.
    for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                  gaussians._scaling, gaussians._rotation, gaussians._opacity,
                  gaussians._refl_strength]:
        param.requires_grad_(False)
    gaussians._mirror_weight.requires_grad_(True)

    optimizer = torch.optim.Adam([gaussians._mirror_weight], lr=args.mirror_lr, eps=1e-15)

    viewpoint_stack = None
    for iteration in range(1, args.mask_fit_iters + 1):
        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        gt_mask = viewpoint_cam.spec_mask
        gt_mask = (gt_mask > 0.5).float()

        pred_mask = render_mirror_mask(viewpoint_cam, gaussians, pipe)
        loss = compute_mask_loss(pred_mask, gt_mask, args.mask_loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iteration % args.log_interval == 0:
            print(f"[ITER {iteration}] mask_loss={loss.item():.6f}")

        if iteration % args.debug_interval == 0:
            debug_dir = os.path.join(scene.model_path, "mirror_mask_debug")
            os.makedirs(debug_dir, exist_ok=True)
            save_image(pred_mask.detach(), os.path.join(debug_dir, f"{viewpoint_cam.image_name}_pred.png"))
            save_image(gt_mask.detach(), os.path.join(debug_dir, f"{viewpoint_cam.image_name}_gt.png"))

    if args.save_mask_checkpoint:
        ckpt_path = os.path.join(scene.model_path, f"mirror_mask_chkpnt{args.mask_fit_iters}.pth")
        print(f"[INFO] Saving mirror mask checkpoint to {ckpt_path}")
        torch.save((gaussians.capture(), args.mask_fit_iters), ckpt_path)

    if args.save_ply:
        # Write mirror weights to the ply file for direct loading by the viewer.
        scene.save(args.mask_fit_iters)


def build_texture_sampler(args, dataset, device):
    texture = load_texture(args.texture_path, dataset.source_path, device)
    if texture is None:
        return None, None, None, None

    if args.homography_json is None:
        return texture, None, None, None

    mirror_pts, bg_pts = load_homography_points(args.homography_json)
    frame_path = resolve_image_path(dataset.source_path, dataset.images, args.homography_frame)
    if frame_path is None or not os.path.exists(frame_path):
        raise FileNotFoundError("homography_frame is required when homography_json is provided.")
    orig_w, orig_h = Image.open(frame_path).size
    return texture, mirror_pts, bg_pts, (orig_w, orig_h)


def sample_texture(texture, u0_map):
    tex_h, tex_w = texture.shape[1], texture.shape[2]
    u = u0_map[..., 0]
    v = u0_map[..., 1]
    grid_x = 2.0 * (u / (tex_w - 1.0)) - 1.0
    grid_y = 2.0 * (v / (tex_h - 1.0)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(texture.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0]


def render_preview(scene, gaussians, pipe, dataset, args):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    texture, mirror_pts, bg_pts, orig_size = build_texture_sampler(args, dataset, background.device)

    if args.preview_split == "train":
        cameras = scene.getTrainCameras()
    elif args.preview_split == "test":
        cameras = scene.getTestCameras()
    else:
        cameras = scene.getTrainCameras() + scene.getTestCameras()

    cameras = cameras[:args.preview_max_views] if args.preview_max_views > 0 else cameras
    out_dir = os.path.join(scene.model_path, args.preview_dir)
    os.makedirs(out_dir, exist_ok=True)

    homography_cache = {}

    with torch.no_grad():
        for cam in cameras:
            render_pkg = render(cam, gaussians, pipe, background)
            base = render_pkg["render"].clamp(0.0, 1.0)

            pred_mask = render_mirror_mask(cam, gaussians, pipe)
            pred_mask = pred_mask.clamp(0.0, 1.0)

            if texture is None:
                spec = torch.zeros_like(base)
            elif mirror_pts is None:
                spec = F.interpolate(texture.unsqueeze(0), size=(cam.image_height, cam.image_width), mode="bilinear", align_corners=False)[0]
            else:
                key = (cam.image_height, cam.image_width)
                if key not in homography_cache:
                    scale_x = cam.image_width / float(orig_size[0])
                    scale_y = cam.image_height / float(orig_size[1])
                    mirror_scaled = [[pt[0] * scale_x, pt[1] * scale_y] for pt in mirror_pts]
                    H_tensor = compute_homography(mirror_scaled, bg_pts)
                    homography_cache[key] = build_u0_map(cam.image_height, cam.image_width, H_tensor, background.device)
                u0_map = homography_cache[key]
                spec = sample_texture(texture, u0_map)

            final = base * (1.0 - pred_mask) + spec * pred_mask
            mask_rgb = pred_mask.repeat(3, 1, 1)

            save_image(base, os.path.join(out_dir, f"{cam.image_name}_base.png"))
            save_image(mask_rgb, os.path.join(out_dir, f"{cam.image_name}_mask.png"))
            save_image(spec, os.path.join(out_dir, f"{cam.image_name}_spec.png"))
            save_image(final, os.path.join(out_dir, f"{cam.image_name}_final.png"))


def run(dataset, opt, pipe, args):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    if not args.start_checkpoint:
        raise ValueError("Stage2 mirror mask requires --start_checkpoint from stage1.")

    model_params, _first_iter = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)

    # Keep env lighting consistent with stage1: recover companion point_cloud.map
    # of the start checkpoint iteration (if present), then stage2 save_ply will
    # write this env_map into new .map alongside stage2 ply.
    restore_env_map_from_checkpoint_sidecar(gaussians, args.start_checkpoint)

    if args.mask_fit_iters > 0:
        fit_mirror_weights(scene, gaussians, pipe, args)
    else:
        print("[INFO] mask_fit_iters=0, skip mask fitting.")

    if args.render_preview:
        render_preview(scene, gaussians, pipe, dataset, args)

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage2 mirror mask fitting + preview")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--mask_fit_iters", type=int, default=2000)
    parser.add_argument("--mask_loss", type=str, choices=["l1", "bce", "hybrid"], default="hybrid")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--debug_interval", type=int, default=500)
    parser.add_argument("--no_save_mask_checkpoint", action="store_true", default=False)
    parser.add_argument("--save_ply", action="store_true", default=False)

    parser.add_argument("--no_render_preview", action="store_true", default=False)
    parser.add_argument("--preview_split", type=str, choices=["train", "test", "all"], default="test")
    parser.add_argument("--preview_max_views", type=int, default=10)
    parser.add_argument("--preview_dir", type=str, default="mirror_preview")

    parser.add_argument("--texture_path", type=str, default=None)
    parser.add_argument("--homography_json", type=str, default=None)
    parser.add_argument("--homography_frame", type=str, default=None)

    args = parser.parse_args()

    # Organize behavior and default learning rate based on parameter settings
    args.save_mask_checkpoint = not args.no_save_mask_checkpoint
    args.render_preview = not args.no_render_preview
    if args.mirror_lr <= 0.0:
        args.mirror_lr = 1e-2

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    run(lp.extract(args), op.extract(args), pp.extract(args), args)
