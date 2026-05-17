import ast
import os
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from cubemapencoder import CubemapEncoder
from gaussian_renderer import GaussianModel, get_trans_color, render, render_env_map
from lpipsPyTorch import get_lpips_model
from scene import Scene
from utils.checkpoint_utils import load_training_checkpoint, restore_env_map_from_sidecar
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.spec_mask_utils import erode_region_labels, get_spec_mask_region_labels


class WindowTransmissionMap(nn.Module):
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


def _infer_iter_from_checkpoint(checkpoint_path):
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def _to_color_key(color):
    if isinstance(color, tuple):
        return tuple(int(c) for c in color)
    if isinstance(color, list):
        return tuple(int(c) for c in color)
    if isinstance(color, str):
        try:
            parsed = ast.literal_eval(color)
            if isinstance(parsed, (tuple, list)):
                return tuple(int(c) for c in parsed)
        except Exception:
            return color
    return color


def _normalize_color_to_id(color_to_id):
    if color_to_id is None:
        return None
    return {_to_color_key(k): int(v) for k, v in color_to_id.items()}


def build_trans_maps_from_extras(extras, device="cuda"):
    states = extras.get("trans_maps_state", None) if isinstance(extras, dict) else None
    if states is None:
        raise RuntimeError(
            "Checkpoint has no trans_maps_state. Please pass the Stage3 final_chkpnt*.pth, "
            "not a plain point_cloud iteration or reflection-only checkpoint."
        )

    n_windows = int(extras.get("n_windows", len(states)))
    cubemap_resol = int(extras.get("cubemap_resol", 128))
    trans_maps = nn.ModuleList(
        [WindowTransmissionMap(cubemap_resol=cubemap_resol, init_strength=0.3) for _ in range(n_windows)]
    ).to(device)
    for idx in range(min(len(states), n_windows)):
        trans_maps[idx].load_state_dict(states[idx])
    trans_maps.eval()
    signed_transmission = bool(extras.get("signed_transmission", False))
    weaken_eval_ratio = float(extras.get("weaken_eval_ratio", 0.0))
    return trans_maps, _normalize_color_to_id(extras.get("color_to_id", None)), signed_transmission, weaken_eval_ratio


def find_instance_mask(mask_dir, image_name):
    if not mask_dir:
        return None
    for ext in (".png", ".PNG", ".jpg", ".jpeg"):
        path = os.path.join(mask_dir, image_name + ext)
        if os.path.exists(path):
            return path
    return None


def rgb_mask_to_labels(img_bgr, color_to_id):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    labels = np.zeros(img_rgb.shape[:2], dtype=np.int64)
    for color, cid in color_to_id.items():
        if not isinstance(color, tuple):
            continue
        match = np.all(img_rgb == np.array(color, dtype=np.uint8)[None, None, :], axis=2)
        labels[match] = int(cid)
    return labels


def load_instance_labels_from_dir(mask_dir, image_name, image_height, image_width, color_to_id, device):
    path = find_instance_mask(mask_dir, image_name)
    if path is None:
        return None, None

    if color_to_id is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        if img.shape[0] != image_height or img.shape[1] != image_width:
            img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        labels_np = rgb_mask_to_labels(img, color_to_id)
        region_ids = list(range(1, int(labels_np.max()) + 1))
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        if img.shape[0] != image_height or img.shape[1] != image_width:
            img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        labels_np = img.astype(np.int64)
        region_ids = sorted(int(v) for v in np.unique(labels_np) if int(v) != 0)

    labels = torch.from_numpy(labels_np).long().to(device)
    return labels, region_ids


def get_camera_instance_labels(cam, args, color_to_id):
    device = cam.original_image.device
    if args.instance_mask_dir:
        return load_instance_labels_from_dir(
            args.instance_mask_dir,
            cam.image_name,
            int(cam.image_height),
            int(cam.image_width),
            color_to_id,
            device,
        )

    labels, region_ids = get_spec_mask_region_labels(
        cam, erode_kernel=0, return_region_ids=True
    )
    if labels is None:
        return None, None
    return labels.to(device), region_ids


def resolve_trans_index(region_id, color_to_id, n_windows):
    if color_to_id is not None and not isinstance(region_id, (int, np.integer)):
        mapped = color_to_id.get(_to_color_key(region_id), None)
        if mapped is None:
            return None
        idx = int(mapped) - 1
    elif isinstance(region_id, (int, np.integer)):
        idx = int(region_id) - 1
    else:
        return None
    return idx if 0 <= idx < n_windows else None


def weaken_stage2_render(render_image, mask3, background, ratio):
    if ratio <= 0:
        return render_image
    bg = background[:, None, None].to(device=render_image.device, dtype=render_image.dtype)
    return render_image * (1.0 - ratio * mask3) + bg * (ratio * mask3)


def compose_with_transmission(pred_refl, cam, trans_maps, color_to_id, signed_transmission, background, weaken_ratio, args):
    labels, region_ids = get_camera_instance_labels(cam, args, color_to_id)
    if labels is None or labels.max() <= 0:
        return pred_refl, pred_refl, 0, None

    if args.mask_erode > 1:
        labels, kept = erode_region_labels(labels.cpu(), args.mask_erode, return_kept=True)
        labels = labels.to(pred_refl.device)
        if region_ids is not None:
            region_ids = [region_ids[i - 1] for i in kept]

    pred_stage2 = pred_refl
    pred = pred_refl
    used_mask = torch.zeros_like(labels, dtype=torch.bool, device=pred_refl.device)
    used = 0
    max_label = int(labels.max().item())
    if region_ids is None:
        region_ids = list(range(1, max_label + 1))

    for local_label, region_id in enumerate(region_ids, start=1):
        trans_idx = resolve_trans_index(region_id, color_to_id, len(trans_maps))
        if trans_idx is None:
            continue
        mask = labels == local_label
        if mask.sum().item() < args.min_pixels:
            continue
        mask3 = mask.float().unsqueeze(0).expand_as(pred)
        pred_stage2 = weaken_stage2_render(pred_stage2, mask3, background, weaken_ratio)
        pred = weaken_stage2_render(pred, mask3, background, weaken_ratio)
        c_trans = trans_maps[trans_idx].get_color(cam.HWK, cam.R, cam.T)
        if signed_transmission:
            c_trans = 2.0 * c_trans - 1.0
        pred = pred + trans_maps[trans_idx].strength * c_trans * mask3
        used_mask = torch.logical_or(used_mask, mask)
        used += 1
    return pred_stage2, pred, used, used_mask if used > 0 else None


def masked_l1_and_psnr(pred, gt, mask):
    if mask is None or mask.sum().item() <= 0:
        return None, None, 0
    mask3 = mask.float().unsqueeze(0).expand_as(pred)
    denom = mask3.sum().clamp_min(1.0)
    diff = (pred - gt) * mask3
    l1_value = diff.abs().sum() / denom
    mse_value = (diff * diff).sum() / denom
    psnr_value = -10.0 * torch.log10(mse_value.clamp_min(1e-10))
    return l1_value.item(), psnr_value.item(), int(mask.sum().item())


def pick_views(scene, split):
    if split == "train":
        return scene.getTrainCameras(), "train"
    if split == "all":
        return scene.getTrainCameras() + scene.getTestCameras(), "all"
    return scene.getTestCameras(), "test"


@torch.no_grad()
def render_set(model_path, name, iteration, views, gaussians, trans_maps, color_to_id, signed_transmission, weaken_ratio, pipeline, background, args):
    if args.save_images:
        render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders_transmission")
        gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
        color_path = os.path.join(render_path, "rgb")
        normal_path = os.path.join(render_path, "normal")
        os.makedirs(color_path, exist_ok=True)
        os.makedirs(normal_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)

        ltres = render_env_map(gaussians)
        torchvision.utils.save_image(ltres["env_cood1"], os.path.join(model_path, "light1.png"))
        torchvision.utils.save_image(ltres["env_cood2"], os.path.join(model_path, "light2.png"))

    lpips_model = get_lpips_model(net_type="vgg").cuda()
    ssims, psnrs, lpipss, render_times = [], [], [], []
    masked_l1_refls, masked_psnr_refls = [], []
    masked_l1_transs, masked_psnr_transs = [], []
    masked_view_count = 0
    used_region_count = 0
    masked_pixel_count = 0

    for idx, view in enumerate(tqdm(views, desc="Transmission eval rendering")):
        view.refl_mask = None
        t1 = time.time()
        rendering = render(view, gaussians, pipeline, background)
        pred_refl = rendering["render"].clamp(0.0, 1.0)
        pred_stage2, pred_trans, used, used_mask = compose_with_transmission(
            pred_refl, view, trans_maps, color_to_id, signed_transmission, background, weaken_ratio, args
        )
        pred_stage2 = pred_stage2.clamp(0.0, 1.0)
        pred_trans = pred_trans.clamp(0.0, 1.0)
        render_times.append(time.time() - t1)

        gt_image = view.original_image[0:3, :, :]
        if used > 0:
            masked_view_count += 1
            used_region_count += used
            refl_l1, refl_psnr, pixels = masked_l1_and_psnr(pred_stage2, gt_image, used_mask)
            trans_l1, trans_psnr, _ = masked_l1_and_psnr(pred_trans, gt_image, used_mask)
            if refl_l1 is not None:
                masked_l1_refls.append(refl_l1)
                masked_psnr_refls.append(refl_psnr)
                masked_l1_transs.append(trans_l1)
                masked_psnr_transs.append(trans_psnr)
                masked_pixel_count += pixels

        render_color = pred_trans[None]
        gt = gt_image[None]
        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(lpips_model(render_color, gt).item())

        if args.save_images:
            normal_map = rendering["normal_map"] * 0.5 + 0.5
            torchvision.utils.save_image(render_color, os.path.join(color_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(normal_map, os.path.join(normal_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

    metric = {
        "psnr": float(np.array(psnrs).mean()),
        "ssim": float(np.array(ssims).mean()),
        "lpips": float(np.array(lpipss).mean()),
        "fps": float(1.0 / np.array(render_times).mean()),
        "masked_views": int(masked_view_count),
        "used_regions": int(used_region_count),
        "masked_pixels": int(masked_pixel_count),
        "masked_l1_refl": float(np.array(masked_l1_refls).mean()) if masked_l1_refls else float("nan"),
        "masked_l1_trans": float(np.array(masked_l1_transs).mean()) if masked_l1_transs else float("nan"),
        "masked_psnr_refl": float(np.array(masked_psnr_refls).mean()) if masked_psnr_refls else float("nan"),
        "masked_psnr_trans": float(np.array(masked_psnr_transs).mean()) if masked_psnr_transs else float("nan"),
    }
    metric_line = (
        f"psnr:{metric['psnr']},ssim:{metric['ssim']},lpips:{metric['lpips']},"
        f"fps:{metric['fps']},masked_views:{metric['masked_views']},"
        f"used_regions:{metric['used_regions']},masked_pixels:{metric['masked_pixels']},"
        f"masked_l1_refl:{metric['masked_l1_refl']},"
        f"masked_l1_trans:{metric['masked_l1_trans']},"
        f"masked_psnr_refl:{metric['masked_psnr_refl']},"
        f"masked_psnr_trans:{metric['masked_psnr_trans']}"
    )
    print(metric_line)
    with open(os.path.join(model_path, args.metric_name), "w") as f:
        f.write(metric_line)


def render_sets(dataset, opt, iteration, pipeline, args):
    checkpoint_path = args.trans_checkpoint
    if checkpoint_path is None:
        if iteration is None or iteration < 0:
            raise ValueError("--trans_checkpoint is required for Stage3 transmission evaluation.")
        checkpoint_path = os.path.join(dataset.model_path, f"final_chkpnt{iteration}.pth")

    load_iter = _infer_iter_from_checkpoint(checkpoint_path)
    if load_iter is None:
        load_iter = iteration

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=False)
    gaussians.training_setup(opt)

    model_params, first_iter, extras = load_training_checkpoint(checkpoint_path)
    gaussians.restore(model_params, opt)
    restore_env_map_from_sidecar(gaussians, checkpoint_path)
    trans_maps, color_to_id, signed_transmission, ckpt_weaken_ratio = build_trans_maps_from_extras(extras, device="cuda")
    weaken_ratio = ckpt_weaken_ratio if args.weaken_stage2_ratio < 0 else args.weaken_stage2_ratio
    weaken_ratio = float(max(0.0, min(1.0, weaken_ratio)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views, split_name = pick_views(scene, args.eval_split)
    if len(views) == 0:
        raise RuntimeError("No views selected for evaluation.")

    print(f"[Transmission Eval] checkpoint={checkpoint_path}")
    print(f"[Transmission Eval] checkpoint_iter={first_iter}, render_iter={scene.loaded_iter}, windows={len(trans_maps)}")
    print(f"[Transmission Eval] signed_transmission={signed_transmission}")
    print(f"[Transmission Eval] weaken_stage2_ratio={weaken_ratio}")
    if args.instance_mask_dir:
        print(f"[Transmission Eval] using instance masks from: {args.instance_mask_dir}")
    else:
        print(f"[Transmission Eval] using camera spec masks from --spec_mask_dir={dataset.spec_mask_dir}")

    render_set(
        dataset.model_path,
        split_name,
        scene.loaded_iter,
        views,
        gaussians,
        trans_maps,
        color_to_id,
        signed_transmission,
        weaken_ratio,
        pipeline,
        background,
        args,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate Stage3 transmission checkpoints.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--trans_checkpoint", default=None, type=str)
    parser.add_argument("--instance_mask_dir", default=None, type=str)
    parser.add_argument("--mask_erode", default=0, type=int)
    parser.add_argument(
        "--weaken_stage2_ratio",
        default=-1.0,
        type=float,
        help="评估时玻璃区域 Stage2 弱化比例；负值表示读取 checkpoint 中保存的 weaken_eval_ratio。"
    )
    parser.add_argument("--min_pixels", default=1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--metric_name", default="metric_transmission.txt", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Transmission rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args)
