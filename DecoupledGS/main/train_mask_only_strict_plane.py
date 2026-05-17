#
# Strict mask-only finetuning:
# optimize reflection/env parameters using losses computed only inside spec masks.
#
#
# -----------------------------------------------------------------------------
# Plane-Constrained Variant: design notes for train_mask_only_strict_plane.py
# -----------------------------------------------------------------------------
# This script is a non-breaking extension of train_mask_only_strict.py.
# Original code is kept untouched; this file adds optional geometry constraints
# for mirror-like regions to reduce "depth spread" when viewed from the side.
#
# Why this variant:
# 1) In strict training, reflection/env can improve appearance but mirror
#    gaussians may still have noticeable front/back thickness.
# 2) Excess thickness causes unstable side-view reflections and wobbly normals.
# 3) We want planar mirror geometry while preserving env-map based rendering.
#
# Core change idea:
# - Keep the original strict loss terms.
# - Add a mirror-plane regularization term on selected mirror gaussians.
# - Optionally unfreeze geometry (xyz/scaling/rotation) so this regularization
#   can actually move gaussians toward a plane.
#
# Selection of constrained gaussians:
# - Use soft confidence from mirror and reflection attributes:
#     sel = (mirror_weight > t_mirror) AND (refl_strength > t_refl)
# - Thresholds are configurable:
#     --plane_weight_mirror_thr
#     --plane_weight_refl_thr
#
# Mathematical method:
# (A) Weighted plane fitting (per step, no gradient through fit):
#     Given selected points p_i in R^3 and weights w_i,
#     normalized weights: w'_i = w_i / sum_j w_j
#     weighted center:     c = sum_i w'_i p_i
#     weighted covariance: C = sum_i w'_i (p_i-c)(p_i-c)^T
#     plane normal n:      eigenvector of C with smallest eigenvalue
#     plane offset d:      d = -n^T c
#
# (B) Distance-to-plane loss (robust):
#     signed distance: s_i = n^T p_i + d
#     robust distance: rho(s_i) = sqrt(s_i^2 + eps)
#     L_plane_dist = sum_i w_i * rho(s_i) / sum_i w_i
#
# (C) Thickness loss along plane normal:
#     projection: z_i = n^T p_i
#     weighted mean: mu = sum_i w_i z_i / sum_i w_i
#     L_plane_thickness = sum_i w_i (z_i - mu)^2 / sum_i w_i
#
# Final objective in this script:
#     L_total = L_strict_original
#             + lambda_plane_dist * L_plane_dist
#             + lambda_plane_thickness * L_plane_thickness
#
# Optimization policy:
# - Default behavior remains close to strict: geometry frozen.
# - If --plane_optimize_geometry is set:
#     * unfreeze xyz/scaling/rotation
#     * scale their LR by --plane_geo_lr_scale for stability
# - This keeps geometry updates conservative and avoids destroying appearance.
#
# Practical expectations:
# - Proper weights reduce mirror depth spread and improve side-view coherence.
# - Overly large lambdas can over-flatten geometry and hurt reflection fit.
# - Recommended workflow: start small, inspect side-view + reflection quality,
#   then increase plane weights gradually.
#
# Added CLI args (relative to strict):
#   --lambda_plane_dist
#   --lambda_plane_thickness
#   --plane_weight_mirror_thr
#   --plane_weight_refl_thr
#   --plane_max_points
#   --plane_robust_eps
#   --plane_optimize_geometry
#   --plane_geo_lr_scale
#
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# 中文版直观说明（给调参与复现实验时快速查阅）
# -----------------------------------------------------------------------------
# 这个脚本是在 train_mask_only_strict.py 的基础上“新增”平面约束，
# 不是改写原逻辑。原始 strict 的损失、训练流程都保留。
#
# 改动目的：
# - 让镜面区域的高斯从侧面看不要前后分层太厚；
# - 同时保留原有反射/envmap 的显示能力，不把反射链路删掉。
#
# 直观做法：
# 1) 先挑出“更像镜面”的高斯：
#    mirror_weight 高 + refl_strength 高。
# 2) 用这些点每步拟合一个“当前最佳平面”。
# 3) 对这些点加两种惩罚：
#    - 离平面太远要罚（距离项）；
#    - 沿法向厚度太大要罚（厚度方差项）。
# 4) 把这两个惩罚加到 strict 原始总损失中。
#
# 数学上对应：
# - 距离项：L_plane_dist
# - 厚度项：L_plane_thickness
# - 总损失：L_total = L_strict + λ1 * L_plane_dist + λ2 * L_plane_thickness
#
# 重要开关：
# - 默认仍按 strict 冻结几何，只优化反射相关参数；
# - 只有加 --plane_optimize_geometry，才会放开 xyz/scaling/rotation，
#   让平面约束真正推动几何变化；
# - --plane_geo_lr_scale 用于把几何学习率缩小，避免形变过猛。
#
# 参数理解（建议先小后大）：
# - --lambda_plane_dist：控制“贴近平面”强度；
# - --lambda_plane_thickness：控制“法向变薄”强度；
# - --plane_weight_mirror_thr / --plane_weight_refl_thr：控制参与平面约束的
#   高斯筛选门槛；
# - --plane_max_points：大场景时采样上限，防止每步平面拟合太慢。
#
# 经验：
# - 若侧视仍散：先小幅增大 λ_thickness；
# - 若反射被压坏：减小两项 λ 或降低几何学习率缩放；
# - 推荐每次只改一个参数，观察侧视和正视一起评估。
#
# -----------------------------------------------------------------------------

import os
import re
import sys
from argparse import ArgumentParser
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from train import prepare_output_and_logger
from utils.general_utils import safe_state
from utils.spec_mask_utils import get_spec_mask_region_labels, get_spec_mask_union, regionwise_weighted_l1_from_labels_with_info


def _fit_weighted_plane(points, weights, eps=1e-8):
    w = weights / (weights.sum() + eps)
    center = (points * w[:, None]).sum(dim=0)
    q = points - center[None]
    cov = (q * w[:, None]).transpose(0, 1) @ q
    evals, evecs = torch.linalg.eigh(cov)
    normal = evecs[:, 0]
    normal = normal / (torch.norm(normal) + eps)
    d = -torch.dot(normal, center)
    return normal, d


def mirror_plane_loss(gaussians, args):
    mirror_w = gaussians.get_mirror_weight.flatten().detach()
    refl_w = gaussians.get_refl.flatten().detach()
    sel = (mirror_w > args.plane_weight_mirror_thr) & (refl_w > args.plane_weight_refl_thr)
    if not torch.any(sel):
        z = gaussians.get_xyz.sum() * 0.0
        return z, z, torch.tensor(0, device=gaussians.get_xyz.device)

    xyz = gaussians.get_xyz
    pts = xyz[sel]
    w = (mirror_w[sel] * refl_w[sel]).clamp_min(1e-6)
    n_pts = pts.shape[0]
    if args.plane_max_points > 0 and n_pts > args.plane_max_points:
        idx = torch.randperm(n_pts, device=pts.device)[: args.plane_max_points]
        pts = pts[idx]
        w = w[idx]
        n_pts = pts.shape[0]

    with torch.no_grad():
        n, d = _fit_weighted_plane(pts.detach(), w.detach())

    signed = torch.matmul(pts, n) + d
    robust = torch.sqrt(signed * signed + args.plane_robust_eps)
    l_dist = (robust * w).sum() / (w.sum() + 1e-8)

    proj = torch.matmul(pts, n)
    mu = (proj * w).sum() / (w.sum() + 1e-8)
    l_thick = ((proj - mu) ** 2 * w).sum() / (w.sum() + 1e-8)
    return l_dist, l_thick, torch.tensor(int(n_pts), device=pts.device)


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


def render_mirror_mask(camera, gaussians, pipe):
    mirror_weights = gaussians.get_mirror_weight
    mirror_colors = mirror_weights.repeat(1, 3)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    render_pkg = render(camera, gaussians, pipe, bg, override_color=mirror_colors)
    return render_pkg["render"][:1].clamp(0.0, 1.0)


def get_camera_region_labels(camera, mask_erode):
                                                                 
                                              
    return get_spec_mask_region_labels(camera, erode_kernel=mask_erode, return_region_ids=True)


def pick_compare_cameras(scene, split, max_views, only_masked):
    if split == "train":
        cams = scene.getTrainCameras()
    elif split == "test":
        cams = scene.getTestCameras()
    else:
        cams = scene.getTrainCameras() + scene.getTestCameras()
    if only_masked:
        cams = [cam for cam in cams if getattr(cam, "spec_mask", None) is not None]
    cams = sorted(cams, key=lambda c: c.image_name)
    if max_views > 0:
        cams = cams[:max_views]
    return cams


@torch.no_grad()
def collect_compare_results(cameras, gaussians, pipe, background):
    results = {}
    for cam in cameras:
        render_pkg = render(cam, gaussians, pipe, background, initial_stage=False, use_mirror_gate=False)
        rgb = render_pkg["render"].clamp(0.0, 1.0).detach().cpu()
        gt = cam.original_image.cuda().clamp(0.0, 1.0).detach().cpu()
        pred_mask = render_mirror_mask(cam, gaussians, pipe).detach().cpu()
        gt_mask = get_spec_mask_union(cam)
        if gt_mask is not None:
            gt_mask = gt_mask.detach().cpu()
        else:
            gt_mask = torch.zeros_like(pred_mask)
        results[cam.image_name] = {
            "rgb": rgb,
            "gt": gt,
            "pred_mask": pred_mask,
            "gt_mask": gt_mask,
        }
    return results


def save_compare_pass(results, out_dir, tag):
    pass_dir = os.path.join(out_dir, tag)
    os.makedirs(pass_dir, exist_ok=True)
    for name, pack in results.items():
        save_image(pack["rgb"], os.path.join(pass_dir, f"{name}_{tag}_rgb.png"))
        save_image(pack["pred_mask"], os.path.join(pass_dir, f"{name}_{tag}_pred_mask.png"))
        if tag == "before":
            save_image(pack["gt"], os.path.join(pass_dir, f"{name}_gt_rgb.png"))
            save_image(pack["gt_mask"], os.path.join(pass_dir, f"{name}_gt_mask.png"))


def save_compare_panels(before, after, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    common = sorted(set(before.keys()).intersection(set(after.keys())))
    for name in common:
        b = before[name]
        a = after[name]
        rgb_diff = ((a["rgb"] - b["rgb"]).abs() * 4.0).clamp(0.0, 1.0)
        rgb_panel = torch.cat([b["gt"], b["rgb"], a["rgb"], rgb_diff], dim=2)
        save_image(rgb_panel, os.path.join(out_dir, f"{name}_rgb_gt_before_after_diffx4.png"))

        gm = b["gt_mask"].repeat(3, 1, 1)
        bm = b["pred_mask"].repeat(3, 1, 1)
        am = a["pred_mask"].repeat(3, 1, 1)
        md = ((a["pred_mask"] - b["pred_mask"]).abs()).repeat(3, 1, 1).clamp(0.0, 1.0)
        mask_panel = torch.cat([gm, bm, am, md], dim=2)
        save_image(mask_panel, os.path.join(out_dir, f"{name}_mask_gt_before_after_diff.png"))


def erode_binary_mask(mask, kernel_size):
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    inv = 1.0 - mask.unsqueeze(0)
    inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (1.0 - inv[0]).clamp(0.0, 1.0)


def env_tv_loss(gaussians):
    texture = gaussians.get_envmap.params["Cubemap_texture"]
    tv_h = (texture[:, :, :, 1:] - texture[:, :, :, :-1]).abs().mean()
    tv_v = (texture[:, :, 1:, :] - texture[:, :, :-1, :]).abs().mean()
    return tv_h + tv_v


@torch.no_grad()
def precompute_reference_renders(cameras, gaussians, pipe, background):
    refs = {}
    for cam in cameras:
        render_pkg = render(cam, gaussians, pipe, background, initial_stage=False, use_mirror_gate=False)
        refs[cam.image_name] = render_pkg["render"].clamp(0.0, 1.0).half().cpu()
    return refs


def set_strict_trainable(gaussians, opt, train_refl):
    for param in [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians._mirror_weight,
    ]:
        param.requires_grad_(False)
    gaussians._refl_strength.requires_grad_(bool(train_refl))
    for param in gaussians.env_map.parameters():
        param.requires_grad_(True)

    for group in gaussians.optimizer.param_groups:
        name = group.get("name", "")
        if name == "env":
            group["lr"] = opt.envmap_cubemap_lr
        elif name == "refl":
            group["lr"] = opt.refl_lr if train_refl else 0.0
        else:
            group["lr"] = 0.0


def maybe_enable_geometry_for_plane(gaussians, opt, args):
    if not args.plane_optimize_geometry:
        return
    gaussians._xyz.requires_grad_(True)
    gaussians._scaling.requires_grad_(True)
    gaussians._rotation.requires_grad_(True)
    for group in gaussians.optimizer.param_groups:
        name = group.get("name", "")
        if name in ("xyz", "scaling", "rotation"):
            group["lr"] = float(group["lr"]) * float(args.plane_geo_lr_scale)


def strict_mask_training(dataset, opt, pipe, args):
    if args.start_checkpoint is None:
        raise ValueError("strict mask-only training requires --start_checkpoint.")

    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    need_test_cameras = bool(args.render_compare and args.compare_split in ["test", "all"])
    scene = Scene(dataset, gaussians, load_test_cameras=need_test_cameras)
    gaussians.training_setup(opt)

    model_params, first_iter = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)

    # NOTE:
    # training checkpoints (.pth) in this repo do not reliably carry env_map weights.
    # To avoid random/rainbow reflections, recover env_map from companion point_cloud.map
    # of the same iteration if available.
    restore_env_map_from_checkpoint_sidecar(gaussians, args.start_checkpoint)

    set_strict_trainable(gaussians, opt, args.strict_train_refl)
    maybe_enable_geometry_for_plane(gaussians, opt, args)

    train_cams = [cam for cam in scene.getTrainCameras() if getattr(cam, "spec_mask", None) is not None]
    if not train_cams:
        raise RuntimeError("No training camera has spec_mask; please check data/masks/*.png and names.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    compare_cams = []
    before_results = None
    compare_root = os.path.join(scene.model_path, args.compare_out_dir)
    print("[INFO] Precomputing reference renders for non-mask keep loss...")
    ref_renders = precompute_reference_renders(train_cams, gaussians, pipe, background)
    if args.render_compare:
        compare_cams = pick_compare_cameras(scene, args.compare_split, args.compare_max_views, args.compare_only_masked)
        if not compare_cams:
            raise RuntimeError("No camera selected for before/after rendering. Check --compare_split and masks.")
        before_results = collect_compare_results(compare_cams, gaussians, pipe, background)
        save_compare_pass(before_results, compare_root, "before")
        print(f"[INFO] saved BEFORE renders: {len(compare_cams)} views -> {os.path.join(compare_root, 'before')}")

    view_stack = []
    progress = tqdm(range(1, args.strict_iters + 1), desc="Strict mask-only training")
    ema = 0.0

    for local_step in progress:
        global_iter = first_iter + local_step
        gaussians.update_learning_rate(global_iter)

        if not view_stack:
            view_stack = train_cams.copy()
        cam = view_stack.pop(randint(0, len(view_stack) - 1))

        render_pkg = render(cam, gaussians, pipe, background, initial_stage=False, use_mirror_gate=False)
        pred = render_pkg["render"]
        gt = cam.original_image.cuda()

              
                                      
                           
        region_labels, region_ids = get_camera_region_labels(cam, args.strict_mask_erode)
        if region_labels is None:
            continue

                             
                          
                   
        mask = (region_labels > 0).unsqueeze(0).float()
        if mask.sum().item() < 1.0:
            continue

              
                                 
                                                                    
                                       
        rgb_l1, region_loss_info = regionwise_weighted_l1_from_labels_with_info(pred, gt, region_labels, region_ids=region_ids)
        if rgb_l1 is None:
            continue
        loss = rgb_l1

        if args.region_loss_debug and local_step % args.region_loss_debug_every == 0:
            print(f"[REGION_LOSS] step={local_step} cam={cam.image_name}")
            for item in region_loss_info:
                print(
                    f"  region={item['region_index']} "
                    f"region_id={item.get('region_id', None)} "
                    f"pixels={item['pixel_count']} "
                    f"loss={item['loss']:.6f}",
                    flush=True,
                )

        keep_loss = torch.tensor(0.0, device=loss.device)
        if args.lambda_nonmask_keep > 0:
            ref = ref_renders[cam.image_name].to(device=pred.device, dtype=pred.dtype)
            mask3 = mask.expand_as(pred)
                                                            
            nonmask3 = 1.0 - mask3
            keep_loss = (torch.abs(pred - ref) * nonmask3).sum() / (nonmask3.sum() + 1e-6)
            loss = loss + args.lambda_nonmask_keep * keep_loss

        leak_loss = torch.tensor(0.0, device=loss.device)
        if args.lambda_mirror_leak > 0 and "refl_strength_map" in render_pkg:
                                                            
            leak_loss = (render_pkg["refl_strength_map"] * (1.0 - mask)).mean()
            loss = loss + args.lambda_mirror_leak * leak_loss

        tv = torch.tensor(0.0, device=loss.device)
        if args.lambda_env_tv > 0:
            tv = env_tv_loss(gaussians)
            loss = loss + args.lambda_env_tv * tv

        plane_dist = torch.tensor(0.0, device=loss.device)
        plane_thick = torch.tensor(0.0, device=loss.device)
        plane_pts = torch.tensor(0, device=loss.device)
        if args.lambda_plane_dist > 0 or args.lambda_plane_thickness > 0:
            plane_dist, plane_thick, plane_pts = mirror_plane_loss(gaussians, args)
            loss = (
                loss
                + args.lambda_plane_dist * plane_dist
                + args.lambda_plane_thickness * plane_thick
            )

        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        ema = 0.4 * loss.item() + 0.6 * ema
        if local_step % args.log_interval == 0:
            progress.set_postfix(
                {
                    "loss": f"{ema:.6f}",
                    "rgb_l1": f"{rgb_l1.item():.6f}",
                    "keep": f"{keep_loss.item():.6f}",
                    "leak": f"{leak_loss.item():.6f}",
                    "plane": f"{plane_dist.item():.6f}",
                    "thick": f"{plane_thick.item():.6f}",
                    "pts": int(plane_pts.item()),
                }
            )

        if args.save_every > 0 and local_step % args.save_every == 0:
            scene.save(global_iter)

        if args.checkpoint_every > 0 and local_step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(scene.model_path, f"strict_mask_chkpnt{global_iter}.pth")
            torch.save((gaussians.capture(), global_iter), ckpt_path)

    final_iter = first_iter + args.strict_iters
    scene.save(final_iter)
    final_ckpt = os.path.join(scene.model_path, f"strict_mask_chkpnt{final_iter}.pth")
    torch.save((gaussians.capture(), final_iter), final_ckpt)

    if args.render_compare and before_results is not None:
        after_results = collect_compare_results(compare_cams, gaussians, pipe, background)
        save_compare_pass(after_results, compare_root, "after")
        save_compare_panels(before_results, after_results, os.path.join(compare_root, "compare"))
        print(f"[INFO] saved AFTER and COMPARE panels -> {compare_root}")

    print(f"[DONE] saved final ply/checkpoint at iter {final_iter}")
    print(f"[DONE] checkpoint: {final_ckpt}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Strict mask-only finetuning")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--strict_iters", type=int, default=4000)
    parser.add_argument("--strict_train_refl", action="store_true", default=False)
    parser.add_argument("--strict_mask_erode", type=int, default=3)
    parser.add_argument("--lambda_nonmask_keep", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--render_compare", action="store_true", default=False)
    parser.add_argument("--compare_split", type=str, choices=["train", "test", "all"], default="train")
    parser.add_argument("--compare_max_views", type=int, default=6)
    parser.add_argument("--compare_only_masked", action="store_true", default=False)
    parser.add_argument("--compare_out_dir", type=str, default="strict_before_after")
    parser.add_argument("--region_loss_debug", action="store_true", default=False)
    parser.add_argument("--region_loss_debug_every", type=int, default=1)
    parser.add_argument("--lambda_plane_dist", type=float, default=0.0)
    parser.add_argument("--lambda_plane_thickness", type=float, default=0.0)
    parser.add_argument("--plane_weight_mirror_thr", type=float, default=0.5)
    parser.add_argument("--plane_weight_refl_thr", type=float, default=0.05)
    parser.add_argument("--plane_max_points", type=int, default=50000)
    parser.add_argument("--plane_robust_eps", type=float, default=1e-6)
    parser.add_argument("--plane_optimize_geometry", action="store_true", default=False)
    parser.add_argument("--plane_geo_lr_scale", type=float, default=0.25)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    strict_mask_training(lp.extract(args), op.extract(args), pp.extract(args), args)
