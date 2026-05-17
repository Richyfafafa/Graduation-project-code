#
# Strict mask-only finetuning, plane-constrained 5.3 variant.
#
# This file intentionally does not replace train_mask_only_strict_plane.py.
# It adds a more stable plane geometry constraint:
#   1. assign mirror-like Gaussians to glass regions by projecting them into
#      annotated mask views and voting;
#   2. fit one plane per region instead of one global plane;
#   3. smooth fitted planes with EMA;
#   4. optionally add anchor, normal alignment, and normal-scale losses.
#

import math
import os
import sys
from argparse import ArgumentParser
from random import randint

import torch
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import train_mask_only_strict_plane as base
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import build_rotation, safe_state
from utils.spec_mask_utils import regionwise_weighted_l1_from_labels_with_info


def _ckpt_path(model_path, iteration):
    return os.path.join(model_path, f"strict_mask_plane_5.3_chkpnt{iteration}.pth")


def _zero_from_gaussians(gaussians):
    return gaussians.get_xyz.sum() * 0.0


def _region_key(region_id):
    if isinstance(region_id, torch.Tensor):
        region_id = region_id.detach().cpu().tolist()
    if isinstance(region_id, (list, tuple)):
        return tuple(int(v) for v in region_id)
    return int(region_id)


def _region_name(region_id):
    if isinstance(region_id, tuple):
        return ",".join(str(v) for v in region_id)
    return str(region_id)


def _fit_weighted_plane(points, weights, eps=1e-8):
    w = weights / (weights.sum() + eps)
    center = (points * w[:, None]).sum(dim=0)
    q = points - center[None]
    cov = (q * w[:, None]).transpose(0, 1) @ q
    _evals, evecs = torch.linalg.eigh(cov)
    normal = evecs[:, 0]
    normal = normal / (torch.norm(normal) + eps)
    d = -torch.dot(normal, center)
    return normal, d


def _maybe_reject_outliers(points, weights, normal, d, args):
    q = float(args.plane53_outlier_quantile)
    if q <= 0.0 or q >= 1.0:
        return points, weights
    if points.shape[0] < max(2 * int(args.plane53_min_region_points), 16):
        return points, weights
    signed = torch.abs(torch.matmul(points, normal) + d)
    cutoff = torch.quantile(signed.detach(), q)
    keep = signed <= cutoff
    if int(keep.sum().item()) < int(args.plane53_min_region_points):
        return points, weights
    return points[keep], weights[keep]


def _collect_region_maps(cameras, mask_erode):
    key_to_index = {}
    index_to_key = []
    for cam in cameras:
        labels, region_ids = base.get_camera_region_labels(cam, mask_erode)
        if labels is None or not region_ids:
            continue
        for region_id in region_ids:
            key = _region_key(region_id)
            if key not in key_to_index:
                key_to_index[key] = len(index_to_key)
                index_to_key.append(key)
    return key_to_index, index_to_key


def _camera_local_to_global(cam, region_key_to_index, mask_erode, device):
    labels, region_ids = base.get_camera_region_labels(cam, mask_erode)
    if labels is None or not region_ids:
        return None, None
    max_label = int(labels.max().item()) if labels.numel() > 0 else 0
    if max_label <= 0:
        return None, None
    local_to_global = torch.full((max_label + 1,), -1, device=device, dtype=torch.long)
    for local_idx, region_id in enumerate(region_ids, start=1):
        if local_idx >= local_to_global.numel():
            break
        key = _region_key(region_id)
        global_idx = region_key_to_index.get(key, None)
        if global_idx is not None:
            local_to_global[local_idx] = int(global_idx)
    return labels.to(device=device, non_blocking=True), local_to_global


def _camera_k_tensor(cam, device, dtype):
    if hasattr(cam, "HWK"):
        return torch.as_tensor(cam.HWK[2], device=device, dtype=dtype)
    width = float(cam.image_width)
    height = float(cam.image_height)
    fx = width / (2.0 * math.tan(float(cam.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(cam.FoVy) * 0.5))
    return torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def _vote_regions_for_candidates(points, cameras, region_key_to_index, args):
    n_points = int(points.shape[0])
    n_regions = len(region_key_to_index)
    device = points.device
    if n_points == 0 or n_regions == 0:
        return torch.full((n_points,), -1, device=device, dtype=torch.long)

    sorted_cameras = sorted(cameras, key=lambda c: c.image_name)
    max_cams = int(args.plane53_assign_max_cameras)
    if max_cams > 0:
        sorted_cameras = sorted_cameras[:max_cams]

    votes = torch.zeros((n_points, n_regions), device=device, dtype=torch.float32)
    chunk = max(1024, int(args.plane53_projection_chunk))
    for cam in sorted_cameras:
        labels, local_to_global = _camera_local_to_global(
            cam, region_key_to_index, args.strict_mask_erode, device
        )
        if labels is None:
            continue

        K = _camera_k_tensor(cam, device, points.dtype)
        R = cam.R.to(device=device, dtype=points.dtype)
        T = cam.T.to(device=device, dtype=points.dtype)
        H = int(cam.image_height)
        W = int(cam.image_width)

        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            pts = points[start:end]
            cam_pts = pts @ R + T[None]
            z = cam_pts[:, 2]
            safe_z = z.clamp_min(1e-6)
            px = torch.round(K[0, 0] * cam_pts[:, 0] / safe_z + K[0, 2]).long()
            py = torch.round(K[1, 1] * cam_pts[:, 1] / safe_z + K[1, 2]).long()
            valid = (z > 1e-5) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if not bool(valid.any()):
                continue

            local_labels = torch.zeros((end - start,), device=device, dtype=torch.long)
            local_labels[valid] = labels[py[valid], px[valid]].long()
            valid = valid & (local_labels > 0) & (local_labels < local_to_global.numel())
            if not bool(valid.any()):
                continue

            global_labels = torch.full_like(local_labels, -1)
            global_labels[valid] = local_to_global[local_labels[valid]]
            valid = valid & (global_labels >= 0)
            if not bool(valid.any()):
                continue

            point_ids = torch.arange(start, end, device=device, dtype=torch.long)[valid]
            votes[point_ids, global_labels[valid]] += 1.0

    max_votes, assignment = votes.max(dim=1)
    assignment[max_votes < int(args.plane53_assignment_min_votes)] = -1
    return assignment


@torch.no_grad()
def _select_candidate_indices(gaussians, args):
    mirror_w = gaussians.get_mirror_weight.flatten().detach()
    refl_w = gaussians.get_refl.flatten().detach()
    conf = (mirror_w * refl_w).clamp_min(0.0)
    sel = (mirror_w > args.plane_weight_mirror_thr) & (refl_w > args.plane_weight_refl_thr)
    idx = torch.nonzero(sel, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx

    max_points = int(args.plane_max_points)
    if max_points > 0 and idx.numel() > max_points:
        if args.plane53_sample_strategy == "random":
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
            idx = idx[perm]
        else:
            _vals, order = torch.topk(conf[idx], max_points, sorted=False)
            idx = idx[order]
    return idx


class Plane53State:
    def __init__(self):
        self.candidate_idx = None
        self.anchor_xyz = None
        self.assignment = None
        self.region_key_to_index = {}
        self.index_to_region_key = []
        self.planes = {}
        self.plane_steps = {}
        self.refresh_step = -1

    def refresh(self, gaussians, train_cams, args, local_step):
        self.candidate_idx = _select_candidate_indices(gaussians, args)
        self.refresh_step = int(local_step)
        if self.candidate_idx.numel() == 0:
            self.anchor_xyz = None
            self.assignment = None
            print("[Plane5.3] no candidate Gaussians matched mirror/refl thresholds.")
            return

        self.anchor_xyz = gaussians.get_xyz[self.candidate_idx].detach().clone()
        if args.plane53_mode == "global":
            self.assignment = torch.zeros_like(self.candidate_idx)
            self.region_key_to_index = {"global": 0}
            self.index_to_region_key = ["global"]
        else:
            self.region_key_to_index, self.index_to_region_key = _collect_region_maps(
                train_cams, args.strict_mask_erode
            )
            if not self.region_key_to_index:
                print("[Plane5.3] no region ids found; falling back to one global plane.")
                self.assignment = torch.zeros_like(self.candidate_idx)
                self.region_key_to_index = {"global": 0}
                self.index_to_region_key = ["global"]
            else:
                self.assignment = _vote_regions_for_candidates(
                    self.anchor_xyz, train_cams, self.region_key_to_index, args
                )

        assigned = int((self.assignment >= 0).sum().item()) if self.assignment is not None else 0
        total = int(self.candidate_idx.numel())
        groups = int(torch.unique(self.assignment[self.assignment >= 0]).numel()) if assigned > 0 else 0
        print(
            f"[Plane5.3] candidates={total} assigned={assigned} "
            f"groups={groups} mode={args.plane53_mode}"
        )

    def should_refresh(self, local_step, args):
        if self.candidate_idx is None:
            return True
        interval = int(args.plane53_reassign_every)
        return interval > 0 and (local_step - self.refresh_step) >= interval

    def update_plane(self, group_id, points, weights, args, local_step):
        with torch.no_grad():
            normal, d = _fit_weighted_plane(points.detach(), weights.detach())
            fit_points, fit_weights = _maybe_reject_outliers(
                points.detach(), weights.detach(), normal, d, args
            )
            if fit_points.shape[0] != points.shape[0]:
                normal, d = _fit_weighted_plane(fit_points, fit_weights)

            old = self.planes.get(int(group_id), None)
            decay = float(args.plane53_ema_decay)
            if old is not None and decay > 0.0:
                old_n, old_d = old
                if torch.dot(old_n, normal) < 0:
                    normal = -normal
                    d = -d
                normal = decay * old_n + (1.0 - decay) * normal
                normal = normal / (torch.norm(normal) + 1e-8)
                d = decay * old_d + (1.0 - decay) * d

            self.planes[int(group_id)] = (normal.detach(), d.detach())
            self.plane_steps[int(group_id)] = int(local_step)

    def plane_for_group(self, group_id):
        return self.planes.get(int(group_id), None)

    def should_refit_group(self, group_id, local_step, args):
        if int(group_id) not in self.planes:
            return True
        interval = int(args.plane53_refit_every)
        if interval <= 1:
            return True
        return (local_step - self.plane_steps.get(int(group_id), -1)) >= interval


def _combine_group_values(values, group_weights, balance):
    if len(values) == 0:
        return None
    if balance == "weight":
        weights = torch.stack(group_weights)
        weights = weights / (weights.sum() + 1e-8)
        stacked = torch.stack(values)
        return (stacked * weights).sum()
    return torch.stack(values).mean()


def _normal_alignment_loss(gaussians, indices, normal, weights):
    scales = gaussians.get_scaling[indices]
    min_axis_id = torch.argmin(scales.detach(), dim=-1, keepdim=True)
    min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1.0)
    rot_matrix = build_rotation(gaussians.get_rotation[indices])
    axis = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)
    align = 1.0 - torch.abs(torch.matmul(axis, normal)).clamp(0.0, 1.0)
    return (align * weights).sum() / (weights.sum() + 1e-8)


def _normal_scale_loss(gaussians, indices, normal, weights):
    scales = gaussians.get_scaling[indices]
    rot_matrix = build_rotation(gaussians.get_rotation[indices])
    normal_local = torch.matmul(
        rot_matrix.transpose(1, 2),
        normal.view(1, 3, 1).expand(indices.shape[0], -1, -1),
    ).squeeze(-1)
    extent = torch.sqrt(((normal_local * scales) ** 2).sum(dim=-1) + 1e-12)
    return (extent * weights).sum() / (weights.sum() + 1e-8)


def plane53_loss(gaussians, train_cams, state, args, local_step):
    if state.should_refresh(local_step, args):
        state.refresh(gaussians, train_cams, args, local_step)

    z = _zero_from_gaussians(gaussians)
    stats = {
        "pts": 0,
        "groups": 0,
        "unassigned": 0,
        "candidates": 0,
    }
    if state.candidate_idx is None or state.candidate_idx.numel() == 0:
        return z, z, z, z, z, stats

    idx = state.candidate_idx
    pts = gaussians.get_xyz[idx]
    mirror_w = gaussians.get_mirror_weight.flatten().detach()[idx]
    refl_w = gaussians.get_refl.flatten().detach()[idx]
    weights = (mirror_w * refl_w).flatten().clamp_min(1e-6)
    assignment = state.assignment
    valid_assignment = assignment >= 0

    stats["candidates"] = int(idx.numel())
    stats["unassigned"] = int((~valid_assignment).sum().item())
    if not bool(valid_assignment.any()):
        return z, z, z, z, z, stats

    group_ids = torch.unique(assignment[valid_assignment])
    dist_values = []
    thick_values = []
    anchor_values = []
    align_values = []
    scale_values = []
    group_weights = []
    used_groups = 0
    used_points = 0

    for group_id in group_ids.tolist():
        group_mask = assignment == int(group_id)
        n_group = int(group_mask.sum().item())
        if n_group < int(args.plane53_min_region_points):
            continue

        group_pts = pts[group_mask]
        group_idx = idx[group_mask]
        group_w = weights[group_mask]
        if state.should_refit_group(group_id, local_step, args):
            state.update_plane(group_id, group_pts, group_w, args, local_step)
        plane = state.plane_for_group(group_id)
        if plane is None:
            continue

        normal, d = plane
        signed = torch.matmul(group_pts, normal) + d
        robust = torch.sqrt(signed * signed + args.plane_robust_eps)
        dist = (robust * group_w).sum() / (group_w.sum() + 1e-8)

        proj = torch.matmul(group_pts, normal)
        mu = (proj * group_w).sum() / (group_w.sum() + 1e-8)
        thick = (((proj - mu) ** 2) * group_w).sum() / (group_w.sum() + 1e-8)

        anchor = z
        if float(args.lambda_plane_anchor) > 0.0 and state.anchor_xyz is not None:
            group_anchor = state.anchor_xyz[group_mask]
            drift = torch.sqrt(((group_pts - group_anchor) ** 2).sum(dim=-1) + args.plane_robust_eps)
            anchor = (drift * group_w).sum() / (group_w.sum() + 1e-8)

        align = z
        if float(args.lambda_plane_normal_align) > 0.0:
            align = _normal_alignment_loss(gaussians, group_idx, normal, group_w)

        scale = z
        if float(args.lambda_plane_normal_scale) > 0.0:
            scale = _normal_scale_loss(gaussians, group_idx, normal, group_w)

        dist_values.append(dist)
        thick_values.append(thick)
        anchor_values.append(anchor)
        align_values.append(align)
        scale_values.append(scale)
        group_weights.append(group_w.sum().detach())
        used_groups += 1
        used_points += n_group

    if used_groups == 0:
        return z, z, z, z, z, stats

    stats["groups"] = used_groups
    stats["pts"] = used_points
    balance = args.plane53_region_balance
    dist = _combine_group_values(dist_values, group_weights, balance)
    thick = _combine_group_values(thick_values, group_weights, balance)
    anchor = _combine_group_values(anchor_values, group_weights, balance)
    align = _combine_group_values(align_values, group_weights, balance)
    scale = _combine_group_values(scale_values, group_weights, balance)
    return dist, thick, anchor, align, scale, stats


def strict_mask_training_5_3(dataset, opt, pipe, args):
    if args.start_checkpoint is None:
        raise ValueError("strict mask-only training requires --start_checkpoint.")

    base.prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    need_test_cameras = bool(args.render_compare and args.compare_split in ["test", "all"])
    scene = Scene(dataset, gaussians, load_test_cameras=need_test_cameras)
    gaussians.training_setup(opt)

    model_params, first_iter = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    base.restore_env_map_from_checkpoint_sidecar(gaussians, args.start_checkpoint)

    base.set_strict_trainable(gaussians, opt, args.strict_train_refl)
    base.maybe_enable_geometry_for_plane(gaussians, opt, args)
    if not args.plane_optimize_geometry and (
        args.lambda_plane_dist > 0
        or args.lambda_plane_thickness > 0
        or args.lambda_plane_anchor > 0
        or args.lambda_plane_normal_align > 0
        or args.lambda_plane_normal_scale > 0
    ):
        print(
            "[WARN] Plane losses are enabled, but geometry is frozen. "
            "Add --plane_optimize_geometry if you want xyz/scaling/rotation to move."
        )

    train_cams = [cam for cam in scene.getTrainCameras() if getattr(cam, "spec_mask", None) is not None]
    if not train_cams:
        raise RuntimeError("No training camera has spec_mask; please check data/masks/*.png and names.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    compare_cams = []
    before_results = None
    compare_root = os.path.join(scene.model_path, args.compare_out_dir)
    print("[INFO] Precomputing reference renders for non-mask keep loss...")
    ref_renders = base.precompute_reference_renders(train_cams, gaussians, pipe, background)
    if args.render_compare:
        compare_cams = base.pick_compare_cameras(
            scene, args.compare_split, args.compare_max_views, args.compare_only_masked
        )
        if not compare_cams:
            raise RuntimeError("No camera selected for before/after rendering. Check --compare_split and masks.")
        before_results = base.collect_compare_results(compare_cams, gaussians, pipe, background)
        base.save_compare_pass(before_results, compare_root, "before")
        print(f"[INFO] saved BEFORE renders: {len(compare_cams)} views -> {os.path.join(compare_root, 'before')}")

    plane_state = Plane53State()
    view_stack = []
    progress = tqdm(range(1, args.strict_iters + 1), desc="Strict plane 5.3 training")
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

        region_labels, region_ids = base.get_camera_region_labels(cam, args.strict_mask_erode)
        if region_labels is None:
            continue

        mask = (region_labels > 0).unsqueeze(0).float()
        if mask.sum().item() < 1.0:
            continue

        rgb_l1, region_loss_info = regionwise_weighted_l1_from_labels_with_info(
            pred, gt, region_labels, region_ids=region_ids
        )
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
            tv = base.env_tv_loss(gaussians)
            loss = loss + args.lambda_env_tv * tv

        plane_dist = torch.tensor(0.0, device=loss.device)
        plane_thick = torch.tensor(0.0, device=loss.device)
        plane_anchor = torch.tensor(0.0, device=loss.device)
        plane_align = torch.tensor(0.0, device=loss.device)
        plane_scale = torch.tensor(0.0, device=loss.device)
        plane_stats = {"pts": 0, "groups": 0, "unassigned": 0, "candidates": 0}
        if (
            args.lambda_plane_dist > 0
            or args.lambda_plane_thickness > 0
            or args.lambda_plane_anchor > 0
            or args.lambda_plane_normal_align > 0
            or args.lambda_plane_normal_scale > 0
        ):
            (
                plane_dist,
                plane_thick,
                plane_anchor,
                plane_align,
                plane_scale,
                plane_stats,
            ) = plane53_loss(gaussians, train_cams, plane_state, args, local_step)
            loss = (
                loss
                + args.lambda_plane_dist * plane_dist
                + args.lambda_plane_thickness * plane_thick
                + args.lambda_plane_anchor * plane_anchor
                + args.lambda_plane_normal_align * plane_align
                + args.lambda_plane_normal_scale * plane_scale
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
                    "anchor": f"{plane_anchor.item():.6f}",
                    "align": f"{plane_align.item():.6f}",
                    "nscale": f"{plane_scale.item():.6f}",
                    "groups": int(plane_stats["groups"]),
                    "pts": int(plane_stats["pts"]),
                }
            )

        if args.save_every > 0 and local_step % args.save_every == 0:
            scene.save(global_iter)

        if args.checkpoint_every > 0 and local_step % args.checkpoint_every == 0:
            torch.save((gaussians.capture(), global_iter), _ckpt_path(scene.model_path, global_iter))

    final_iter = first_iter + args.strict_iters
    scene.save(final_iter)
    final_ckpt = _ckpt_path(scene.model_path, final_iter)
    torch.save((gaussians.capture(), final_iter), final_ckpt)

    if args.render_compare and before_results is not None:
        after_results = base.collect_compare_results(compare_cams, gaussians, pipe, background)
        base.save_compare_pass(after_results, compare_root, "after")
        base.save_compare_panels(before_results, after_results, os.path.join(compare_root, "compare"))
        print(f"[INFO] saved AFTER and COMPARE panels -> {compare_root}")

    print(f"[DONE] saved final ply/checkpoint at iter {final_iter}")
    print(f"[DONE] checkpoint: {final_ckpt}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Strict mask-only plane finetuning 5.3")
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
    parser.add_argument("--compare_out_dir", type=str, default="strict_plane_5.3_before_after")
    parser.add_argument("--region_loss_debug", action="store_true", default=False)
    parser.add_argument("--region_loss_debug_every", type=int, default=1)

    parser.add_argument("--lambda_plane_dist", type=float, default=0.0)
    parser.add_argument("--lambda_plane_thickness", type=float, default=0.0)
    parser.add_argument("--lambda_plane_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_plane_normal_align", type=float, default=0.0)
    parser.add_argument("--lambda_plane_normal_scale", type=float, default=0.0)
    parser.add_argument("--plane_weight_mirror_thr", type=float, default=0.5)
    parser.add_argument("--plane_weight_refl_thr", type=float, default=0.05)
    parser.add_argument("--plane_max_points", type=int, default=50000)
    parser.add_argument("--plane_robust_eps", type=float, default=1e-6)
    parser.add_argument("--plane_optimize_geometry", action="store_true", default=False)
    parser.add_argument("--plane_geo_lr_scale", type=float, default=0.25)

    parser.add_argument("--plane53_mode", choices=["per_region", "global"], default="per_region")
    parser.add_argument("--plane53_assignment_min_votes", type=int, default=1)
    parser.add_argument("--plane53_assign_max_cameras", type=int, default=0)
    parser.add_argument("--plane53_projection_chunk", type=int, default=50000)
    parser.add_argument("--plane53_min_region_points", type=int, default=256)
    parser.add_argument("--plane53_refit_every", type=int, default=25)
    parser.add_argument("--plane53_reassign_every", type=int, default=0)
    parser.add_argument("--plane53_ema_decay", type=float, default=0.9)
    parser.add_argument("--plane53_sample_strategy", choices=["topk", "random"], default="topk")
    parser.add_argument("--plane53_region_balance", choices=["equal", "weight"], default="equal")
    parser.add_argument("--plane53_outlier_quantile", type=float, default=0.98)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    print("[Plane5.3] using main/train_mask_only_strict_plane_5.3.py")
    safe_state(args.quiet)
    strict_mask_training_5_3(lp.extract(args), op.extract(args), pp.extract(args), args)
