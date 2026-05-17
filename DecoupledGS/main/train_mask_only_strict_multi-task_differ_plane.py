#
# Strict mask-only finetuning:
# optimize reflection/env parameters using losses computed only inside spec masks.
#

import os
import re
import sys
import json
from argparse import ArgumentParser
from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, get_trans_color
from scene import Scene, GaussianModel
from train import prepare_output_and_logger
from utils.general_utils import safe_state
from utils.spec_mask_utils import erode_regions, get_spec_mask_regions, get_spec_mask_union
from min_norm_solvers import MinNormSolver, gradient_normalizers
from cubemapencoder import CubemapEncoder


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
    # 从 checkpoint 文件名解析迭代号，并恢复同迭代的 point_cloud.map。
    # 这是为了在 strict_multi 阶段延续前序反射环境，而不是随机初始化 env_map。
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


def load_checkpoint_stage3_bridge(path):
    """
    Bridge loader:
    - legacy checkpoint: (gaussians_state, first_iter)
    - stage3 checkpoint: {"gaussians", "first_iter"/"iteration", ...transmission extras...}

    设计意图：
    - 保持 strict_multi 的“原生输入习惯”（优先 tuple）
    - 仅额外兼容 stage3 的 dict 格式，便于无缝接入每窗透射场
    """
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, (tuple, list)) and len(ckpt) >= 2:
        return ckpt[0], int(ckpt[1]), None
    if isinstance(ckpt, dict) and "gaussians" in ckpt:
        first_iter = int(ckpt.get("first_iter", ckpt.get("iteration", 0)))
        extras = {
            "n_windows": ckpt.get("n_windows", None),
            "cubemap_resol": ckpt.get("cubemap_resol", None),
            "trans_maps_state": ckpt.get("trans_maps_state", None),
            "color_to_id": ckpt.get("color_to_id", None),
        }
        return ckpt["gaussians"], first_iter, extras
    raise RuntimeError(f"Unsupported checkpoint format for strict_multi: {path}")


# Read the geo_link_report.json file and extract the region IDs contained in each color mask image.
def parse_geo_link_report(json_path):
    """
    Parameters:
        json_path (str): Path to the JSON file
    Return:
        tuple: (region_num, mask_to_region, color_to_global_id)
               region_num: int，number of elements in id_to_color
               mask_to_region: dict，keys are image names, values are lists of global_ids contained in that image (ascending order)
               color_to_global_id: dict，keys are (R,G,B), values are global_ids
    """
    # Initialization return result
    region_num = 0
    mask_to_region = {}
    color_to_global_id = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)        
        if 'id_to_color' in data and isinstance(data['id_to_color'], dict):
            region_num = len(data['id_to_color'])        
            for gid_str, color in data['id_to_color'].items():
                if isinstance(color, list) and len(color) >= 3:
                    color_to_global_id[tuple(int(c) for c in color[:3])] = int(gid_str)
        if 'images' in data and isinstance(data['images'], dict):
            for img_name, img_annotations in data['images'].items():
                img_id = img_name.split(".")[0]
                global_ids = []
                for annot in img_annotations:
                    if 'global_id' in annot and isinstance(annot['global_id'], int):
                        global_ids.append(annot['global_id'])
                global_ids = sorted(list(set(global_ids)))
                mask_to_region[img_id] = global_ids     
        return region_num, mask_to_region, color_to_global_id    
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file doesn't exist: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"The specified file is not in valid JSON format: {json_path}")
    except Exception as e:
        raise Exception(f"An error occurred while parsing the JSON file: {str(e)}")


class WindowTransmissionMap(nn.Module):
    # 每个玻璃实例一个透射场：
    # - cubemap: 学习该玻璃对应的透射环境颜色
    # - strength_logit: 学习该玻璃透射强度（sigmoid 后在 0~1）
    def __init__(self, cubemap_resol: int = 128, init_strength: float = 0.3):
        super().__init__()
        self.cubemap = CubemapEncoder(output_dim=3, resolution=cubemap_resol)
        init_val = max(min(init_strength, 0.9999), 0.0001)
        init_logit = float(torch.log(torch.tensor(init_val / (1.0 - init_val))))
        self.strength_logit = nn.Parameter(torch.tensor(init_logit))

    @property
    def strength(self) -> torch.Tensor:
        return torch.sigmoid(self.strength_logit)

    def get_color(self, HWK, R, T) -> torch.Tensor:
        return get_trans_color(self.cubemap, HWK, R, T)


def trans_tv_loss(trans_map: WindowTransmissionMap) -> torch.Tensor:
    # 透射 cubemap 的 TV 正则，抑制高频噪声与块状伪影
    texture = trans_map.cubemap.params["Cubemap_texture"]  # (6, 3, R, R)
    tv_h = (texture[:, :, :, 1:] - texture[:, :, :, :-1]).abs().mean()
    tv_v = (texture[:, :, 1:, :] - texture[:, :, :-1, :]).abs().mean()
    return tv_h + tv_v


def build_trans_maps_from_checkpoint_extras(extras, device="cuda"):
    # 从 stage3 checkpoint 的 extras 重建透射场参数
    # 返回：
    # - trans_maps: 每窗透射场参数容器
    # - color_to_id: RGB 掩码颜色到实例 ID 的映射（用于跨脚本一致索引）
    # - cubemap_resol: 透射 cubemap 分辨率
    states = extras.get("trans_maps_state", None) if isinstance(extras, dict) else None
    if states is None:
        return None, None, None
    n_windows = int(extras.get("n_windows", len(states)))
    cubemap_resol = int(extras.get("cubemap_resol", 128))
    color_to_id = extras.get("color_to_id", None)

    trans_maps = nn.ModuleList(
        [WindowTransmissionMap(cubemap_resol=cubemap_resol, init_strength=0.3) for _ in range(n_windows)]
    ).to(device)
    for i in range(min(len(states), n_windows)):
        trans_maps[i].load_state_dict(states[i])
    return trans_maps, color_to_id, cubemap_resol


def _to_color_key(color):
    if isinstance(color, tuple):
        return tuple(int(c) for c in color)
    if isinstance(color, list):
        return tuple(int(c) for c in color)
    return color


def _resolve_trans_index(local_idx, region_color_ids, color_to_id, n_windows):
    # 将当前视图中的“局部区域索引”映射到全局透射场索引。
    # 场景存在 RGB 掩码时优先按 color_to_id 对齐，避免不同视图区域顺序不一致。
    if region_color_ids is not None and local_idx < len(region_color_ids):
        rid = region_color_ids[local_idx]
    else:
        rid = local_idx + 1

    trans_idx = None
    if color_to_id is not None:
        key = _to_color_key(rid)
        mapped = color_to_id.get(key, None)
        if mapped is not None:
            trans_idx = int(mapped) - 1
    else:
        if isinstance(rid, (int, np.integer)):
            trans_idx = int(rid) - 1
        else:
            trans_idx = local_idx

    if trans_idx is None or trans_idx < 0 or trans_idx >= n_windows:
        return None
    return trans_idx


def compose_with_transmission(pred_refl, cam, region_masks, region_color_ids, trans_maps, color_to_id):
    # 联合预测：将基础高斯+反射场渲染出的图像（pred_refl），叠加加上各个玻璃区域的透射颜色（$C_{trans} \times strength \times mask$）。意义：完成了完整的光照渲染等式：$Final = Base + Reflection + Transmission$
    # 返回 used_indices 供后续只对“本步实际可见的透射窗”施加 TV。
    if trans_maps is None:
        return pred_refl, []
    pred = pred_refl
    used_indices = []
    for i in range(region_masks.shape[0]):
        trans_idx = _resolve_trans_index(i, region_color_ids, color_to_id, len(trans_maps))
        if trans_idx is None:
            continue
        used_indices.append(trans_idx)
        mask3 = region_masks[i].unsqueeze(0).expand_as(pred).float()
        c_trans = trans_maps[trans_idx].get_color(cam.HWK, cam.R, cam.T)
        pred = pred + trans_maps[trans_idx].strength * c_trans * mask3
    return pred, used_indices


def render_mirror_mask(camera, gaussians, pipe):#利用冻结的高斯掩码权重 _mirror_weight，渲染出预测的玻璃区域掩码。
    mirror_weights = gaussians.get_mirror_weight
    mirror_colors = mirror_weights.repeat(1, 3)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    render_pkg = render(camera, gaussians, pipe, bg, override_color=mirror_colors)
    return render_pkg["render"][:1].clamp(0.0, 1.0)


def get_camera_regions(camera, mask_erode):#获取当前相机中每一块玻璃独立的掩码
    # Retrieve the “individual mask for each glass pane” from the camera along with the corresponding color for that mask.
    regions = get_spec_mask_regions(camera)
    region_color_ids = list(getattr(camera, "spec_mask_region_ids", []) or [])
    if regions is None:
        return None, None
    if mask_erode > 1:
        # Each area is etched separately to prevent adjacent glass from affecting each other during the union process.
        regions = erode_regions(regions, int(mask_erode))
        keep = regions.flatten(1).sum(dim=1) > 0.5
        if keep.any():
            regions = regions[keep]
            if region_color_ids:
                region_color_ids = [region_color_ids[idx] for idx, flag in enumerate(keep.tolist()) if flag]
        else:
            return None, None
    return regions, region_color_ids


def regionwise_weighted_l1_with_info(pred, gt, region_masks, region_color_ids=None, color_to_global_id=None, extra_weight=None):
    if region_masks is None:
        return None, []
    if region_masks.ndim == 2:
        region_masks = region_masks.unsqueeze(0)
    abs_diff = torch.abs(pred - gt)
    losses = []
    infos = []
    for idx, region in enumerate(region_masks):
        region = region.unsqueeze(0)
        if extra_weight is not None:
            region = region * extra_weight
        denom = pred.shape[0] * region.sum() + 1e-6
        if float(denom.detach().item()) <= 1e-6:
            continue
        loss_i = (abs_diff * region.expand_as(pred)).sum() / denom
        losses.append(loss_i)
        info = {
            "local_region_index": idx + 1,
            "region_index": idx + 1,
            "pixel_count": int(region.sum().detach().item()),
            "loss": float(loss_i.detach().item()),
            "region_color": None,
            "global_id": None,
        }
        if region_color_ids is not None and idx < len(region_color_ids):
            color = region_color_ids[idx]
            if isinstance(color, list):
                color = tuple(int(c) for c in color)
            elif isinstance(color, tuple):
                color = tuple(int(c) for c in color)
            elif isinstance(color, (int, np.integer)):
                color = int(color)
            info["region_color"] = color
            if color_to_global_id is not None and color is not None:
                if isinstance(color, tuple):
                    gid = color_to_global_id.get(color, None)
                elif isinstance(color, (int, np.integer)):
                    # Some datasets provide region IDs directly instead of RGB tuples.
                    gid = int(color)
                else:
                    gid = None
                info["global_id"] = gid
                if gid is not None:
                    # region_index directly aligns with the global ID in geo_link_report.json
                    info["region_index"] = gid
        infos.append(info)
    if len(losses) == 0:
        return None, []
    return torch.stack(losses, dim=0).mean(), infos


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


def maybe_enable_geometry_for_plane(gaussians, args):
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
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # checkpoint 输入桥接：
    # 1) 可读旧 tuple（前序脚本）
    # 2) 可读 stage3 dict（含透射场）
    model_params, first_iter, ckpt_extras = load_checkpoint_stage3_bridge(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    # 若 checkpoint 含 stage3 透射状态，则在 strict_multi 中继续联合优化
    trans_maps, trans_color_to_id, trans_cubemap_resol = build_trans_maps_from_checkpoint_extras(ckpt_extras, device="cuda")
    trans_optimizer = None
    if trans_maps is not None and not args.freeze_transmission:
        trans_optimizer = torch.optim.Adam(trans_maps.parameters(), lr=args.trans_lr, eps=1e-15)
        print(f"[INFO] Loaded {len(trans_maps)} transmission maps from checkpoint and enabled joint optimization.")
    elif trans_maps is not None:
        print(f"[INFO] Loaded {len(trans_maps)} transmission maps from checkpoint (frozen).")
        for p in trans_maps.parameters():
            p.requires_grad_(False)

    # NOTE:
    # training checkpoints (.pth) in this repo do not reliably carry env_map weights.
    # To avoid random/rainbow reflections, recover env_map from companion point_cloud.map
    # of the same iteration if available.
    # 恢复前序反射环境图，保证反射基底连续
    restore_env_map_from_checkpoint_sidecar(gaussians, args.start_checkpoint)

    set_strict_trainable(gaussians, opt, args.strict_train_refl)
    maybe_enable_geometry_for_plane(gaussians, args)

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
    
    # MultiTask part
    region_nums, image_to_regions, color_to_global_id = parse_geo_link_report(args.json_path)
    loss_last_moment = [torch.tensor(0.0) for _ in range(region_nums+1)]
    loss_now_moment = [torch.tensor(0.0) for _ in range(region_nums+1)]
    region_last_train = np.zeros(region_nums + 1, dtype=int)
    region_this_train = np.zeros(region_nums + 1, dtype=int)    

    for local_step in progress:#核心训练主循环
        global_iter = first_iter + local_step
        gaussians.update_learning_rate(global_iter)
        region_this_train = np.zeros(region_nums + 1, dtype=int)

        if not view_stack:
            view_stack = train_cams.copy()
        cam = view_stack.pop(randint(0, len(view_stack) - 1))
        
        render_pkg = render(cam, gaussians, pipe, background, initial_stage=False, use_mirror_gate=False)
        pred_refl = render_pkg["render"]
        gt = cam.original_image.cuda()
        

        # No longer calculate the loss by combining the entire mirrored area into one large mask,
        # but instead calculate each glass panel separately.
        # 获取当前相机中每一块玻璃独立的掩码
        region_masks, region_color_ids = get_camera_regions(cam, args.strict_mask_erode)
        if region_masks is None:
            continue

        # Union mask remains in place for:
        # 1. Non-specular keep loss
        # 2. Reflection leakage penalty
        mask = (region_masks.sum(dim=0, keepdim=True) > 0).float()
        if mask.sum().item() < 1.0:
            continue

        # Compose reflection + per-window transmission (if loaded from stage3 checkpoint).
        # 反射 + 透射联合前向
        pred, used_trans_indices = compose_with_transmission(
            pred_refl, cam, region_masks, region_color_ids, trans_maps, trans_color_to_id
        )

        # Primary Loss:
        # Calculate L1 for each glass pane individually, then average across all panes.
        # This ensures more balanced weighting for each pane in the primary loss calculation, preventing large-area panes from dominating the result.
        rgb_l1, losses = regionwise_weighted_l1_with_info(pred, gt, region_masks, region_color_ids=region_color_ids, color_to_global_id=color_to_global_id)
        if rgb_l1 is None:
            continue
        # loss = rgb_l1
        
        
        #MultiTask part
        if (local_step == 1):
            # 1st iter: loss is the average value, and the value in loss_last_moment is updated with losses.
            # region_last_train is a binary array representing regions that appeared in the last iteration.
            # such as region_last_train[i] = 1, indicating that the ith region appeared in the last iteration
            # Similarly, region_this_train represents regions that appeared in this iteration.
            for item in losses:
                gid = item.get("global_id", None)
                if gid is None or gid < 1 or gid > region_nums:
                    continue
                loss_last_moment[gid] = pred.new_tensor(item["loss"])
                region_last_train[gid] = 1
            loss = rgb_l1
        else:
            # 2nd iter: loss is the weighted average value, and the value in loss_last_moment is updated with losses.
            for item in losses:
                gid = item.get("global_id", None)
                if gid is None or gid < 1 or gid > region_nums:
                    continue
                loss_now_moment[gid] = pred.new_tensor(item["loss"])
                region_this_train[gid] = 1

            # Only regions present in both the last and this iterations can be used to compute differences for weight optimization.
            # We store the corresponding region IDs in loss_to_backward.
            loss_to_backward = []
            for gid in range(1, region_nums + 1):
                if region_this_train[gid] == 1 and region_last_train[gid] == 1:
                    loss_to_backward.append(gid)

            if len(loss_to_backward) == 0:
                loss = rgb_l1
                loss_last_moment = [t.clone() for t in loss_now_moment]
                region_last_train = region_this_train.copy()
            else:
                # Calculate the gradient using the differential method.
                # When i == j, the partial derivative is 1; otherwise, it is computed by (L^i_t-L^i_{t-1})/(L^j_t-L^j_{t-1})
                # To avoid a zero division, if (L^j_t - L^j_{t-1}) is too small, add eps.
                grads = {}
                eps = 1e-8
                for i in loss_to_backward:
                    grads[i] = []
                    for j in loss_to_backward:
                        # Compute the partial derivative for the gradient.
                        if i==j:
                            grads[i].append(torch.tensor(1.0))
                        else:
                            diff_j = loss_now_moment[j] - loss_last_moment[j]
                            diff_j_value = float(diff_j.detach().item())
                            if abs(diff_j_value) < 1e-9:
                                denominator = diff_j + pred.new_tensor(eps)
                            else:
                                denominator = diff_j
                            grads[i].append((loss_now_moment[i] - loss_last_moment[i]) / denominator)

                # Normalize all gradients.
                gn = gradient_normalizers(grads, loss_now_moment, 'loss+')
                for i in loss_to_backward:
                    gn_i = float(gn[i]) if gn[i] != 0 else 1.0
                    for gr_i in range(len(grads[i])):
                        grads[i][gr_i] = grads[i][gr_i] / gn_i

                # Frank-Wolfe iteration to compute scales
                if (len(loss_to_backward) == 1):
                    loss = loss_now_moment[loss_to_backward[0]]
                else:
                    sol, min_norm = MinNormSolver.find_min_norm_element_FW([grads[i] for i in loss_to_backward])                    
                    loss = pred.new_tensor(0.0)
                    for sol_idx, gid in enumerate(loss_to_backward):
                        loss = loss + float(sol[sol_idx]) * loss_now_moment[gid]
                    
                # Update loss_last_moment with losses.
                loss_last_moment = [t.clone() for t in loss_now_moment] 
                region_last_train = region_this_train.copy() 

                              

        keep_loss = torch.tensor(0.0, device=loss.device)
        if args.lambda_nonmask_keep > 0:
            ref = ref_renders[cam.image_name].to(device=pred.device, dtype=pred.dtype)
            mask3 = mask.expand_as(pred)
            # keep_loss: Still using the union mask, as goal is to “keep everything outside the glass unchanged.”
            nonmask3 = 1.0 - mask3
            keep_loss = (torch.abs(pred - ref) * nonmask3).sum() / (nonmask3.sum() + 1e-6)
            loss = loss + args.lambda_nonmask_keep * keep_loss

        leak_loss = torch.tensor(0.0, device=loss.device)
        if args.lambda_mirror_leak > 0 and "refl_strength_map" in render_pkg:
            # leak_loss: Still using the union mask, as it penalizes reflections occurring in non-mirrored areas.
            leak_loss = (render_pkg["refl_strength_map"] * (1.0 - mask)).mean()
            loss = loss + args.lambda_mirror_leak * leak_loss

        tv = torch.tensor(0.0, device=loss.device)
        if args.lambda_env_tv > 0:
            tv = env_tv_loss(gaussians)
            loss = loss + args.lambda_env_tv * tv
        # 只对本步可见的透射窗施加 TV，降低无效正则开销
        if trans_maps is not None and args.lambda_trans_tv > 0 and len(used_trans_indices) > 0:
            tv_t = torch.tensor(0.0, device=loss.device)
            for tidx in sorted(set(used_trans_indices)):
                tv_t = tv_t + trans_tv_loss(trans_maps[tidx])
            tv_t = tv_t / max(1, len(set(used_trans_indices)))
            loss = loss + args.lambda_trans_tv * tv_t

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
        # 联合优化：
        # - gaussians.optimizer 更新反射相关参数
        # - trans_optimizer 更新透射 cubemap 与强度参数
        if trans_optimizer is not None:
            trans_optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        if trans_optimizer is not None:
            trans_optimizer.zero_grad(set_to_none=True)

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
            extras = None
            if trans_maps is not None:
                extras = {
                    "n_windows": int(len(trans_maps)),
                    "cubemap_resol": int(trans_cubemap_resol if trans_cubemap_resol is not None else args.trans_cubemap_resol),
                    "trans_maps_state": [tm.state_dict() for tm in trans_maps],
                    "color_to_id": trans_color_to_id,
                }
            # 保存策略：
            # - 无透射场：保持 legacy tuple 兼容
            # - 有透射场：保存 dict，携带 trans_maps_state/color_to_id
            if extras is None:
                torch.save((gaussians.capture(), global_iter), ckpt_path)
            else:
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "first_iter": global_iter,
                        "n_windows": extras["n_windows"],
                        "cubemap_resol": extras["cubemap_resol"],
                        "trans_maps_state": extras["trans_maps_state"],
                        "color_to_id": extras["color_to_id"],
                    },
                    ckpt_path,
                )

    final_iter = first_iter + args.strict_iters
    scene.save(final_iter)
    final_ckpt = os.path.join(scene.model_path, f"strict_mask_chkpnt{final_iter}.pth")
    extras = None
    if trans_maps is not None:
        extras = {
            "n_windows": int(len(trans_maps)),
            "cubemap_resol": int(trans_cubemap_resol if trans_cubemap_resol is not None else args.trans_cubemap_resol),
            "trans_maps_state": [tm.state_dict() for tm in trans_maps],
            "color_to_id": trans_color_to_id,
        }
    # 最终 checkpoint 同上，按是否含透射场决定格式
    if extras is None:
        torch.save((gaussians.capture(), final_iter), final_ckpt)
    else:
        torch.save(
            {
                "gaussians": gaussians.capture(),
                "first_iter": final_iter,
                "n_windows": extras["n_windows"],
                "cubemap_resol": extras["cubemap_resol"],
                "trans_maps_state": extras["trans_maps_state"],
                "color_to_id": extras["color_to_id"],
            },
            final_ckpt,
        )

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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json_path", type=str, default="./output1/geo_link_report.json")
    parser.add_argument("--trans_lr", type=float, default=0.01)
    parser.add_argument("--lambda_trans_tv", type=float, default=1e-5)
    parser.add_argument("--trans_cubemap_resol", type=int, default=128)
    parser.add_argument("--freeze_transmission", action="store_true", default=False)
    parser.add_argument("--lambda_plane_dist", type=float, default=0.0)
    parser.add_argument("--lambda_plane_thickness", type=float, default=0.0)
    parser.add_argument("--plane_weight_mirror_thr", type=float, default=0.5)
    parser.add_argument("--plane_weight_refl_thr", type=float, default=0.05)
    parser.add_argument("--plane_max_points", type=int, default=50000)
    parser.add_argument("--plane_robust_eps", type=float, default=1e-6)
    parser.add_argument("--plane_optimize_geometry", action="store_true", default=False)
    parser.add_argument("--plane_geo_lr_scale", type=float, default=0.25)

    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    strict_mask_training(lp.extract(args), op.extract(args), pp.extract(args), args)
