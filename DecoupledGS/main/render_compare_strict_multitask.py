import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from cubemapencoder import CubemapEncoder
from gaussian_renderer import render, get_trans_color
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.checkpoint_utils import load_training_checkpoint, restore_env_map_from_sidecar
from utils.spec_mask_utils import get_spec_mask_regions, erode_regions


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


def build_trans_maps_from_checkpoint_extras(extras, device="cuda"):
    states = extras.get("trans_maps_state", None) if isinstance(extras, dict) else None
    if states is None:
        return None, None
    n_windows = int(extras.get("n_windows", len(states)))
    cubemap_resol = int(extras.get("cubemap_resol", 128))
    color_to_id = extras.get("color_to_id", None)

    trans_maps = nn.ModuleList(
        [WindowTransmissionMap(cubemap_resol=cubemap_resol, init_strength=0.3) for _ in range(n_windows)]
    ).to(device)
    for i in range(min(len(states), n_windows)):
        trans_maps[i].load_state_dict(states[i])
    trans_maps.eval()
    return trans_maps, color_to_id


def _to_color_key(color):
    if isinstance(color, tuple):
        return tuple(int(c) for c in color)
    if isinstance(color, list):
        return tuple(int(c) for c in color)
    return color


def _resolve_trans_index(local_idx, region_color_ids, color_to_id, n_windows):
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


def get_camera_regions(camera, mask_erode):
    regions = get_spec_mask_regions(camera)
    region_color_ids = list(getattr(camera, "spec_mask_region_ids", []) or [])
    if regions is None:
        return None, None
    if mask_erode > 1:
        regions = erode_regions(regions, int(mask_erode))
        keep = regions.flatten(1).sum(dim=1) > 0.5
        if keep.any():
            regions = regions[keep]
            if region_color_ids:
                region_color_ids = [region_color_ids[idx] for idx, flag in enumerate(keep.tolist()) if flag]
        else:
            return None, None
    return regions, region_color_ids


def compose_with_transmission(pred_refl, cam, trans_maps, color_to_id, mask_erode):
    if trans_maps is None:
        return pred_refl
    region_masks, region_color_ids = get_camera_regions(cam, mask_erode)
    if region_masks is None:
        return pred_refl

    pred = pred_refl
    for i in range(region_masks.shape[0]):
        trans_idx = _resolve_trans_index(i, region_color_ids, color_to_id, len(trans_maps))
        if trans_idx is None:
            continue
        mask3 = region_masks[i].unsqueeze(0).expand_as(pred).float()
        c_trans = trans_maps[trans_idx].get_color(cam.HWK, cam.R, cam.T)
        pred = pred + trans_maps[trans_idx].strength * c_trans * mask3
    return pred


def pick_cameras(scene, split, max_views, only_masked):
    if split == "train":
        cams = scene.getTrainCameras()
    elif split == "test":
        cams = scene.getTestCameras()
    else:
        cams = scene.getTrainCameras() + scene.getTestCameras()
    if only_masked:
        masked = [cam for cam in cams if getattr(cam, "spec_mask", None) is not None]
        if len(masked) == 0:
            print("[WARN] --compare_only_masked enabled but no camera has spec_mask. Fallback to all selected cameras.")
        else:
            cams = masked
    cams = sorted(cams, key=lambda c: c.image_name)
    if max_views > 0:
        cams = cams[:max_views]
    return cams


def _infer_iter_from_checkpoint(checkpoint_path):
    base = os.path.basename(checkpoint_path)
    stem = os.path.splitext(base)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits == "":
        return None
    return int(digits)


def load_gaussians_from_checkpoint(dataset, opt, checkpoint_path, model_path, load_test_cameras):
    # Allow running without -m by inferring model_path from checkpoint directory.
    dataset.model_path = model_path
    if getattr(dataset, "resolution", None) is None:
        # Keep consistent with training default branch in camera_utils.loadCam
        # where HWK is initialized.
        dataset.resolution = -1
    if getattr(dataset, "data_device", None) in (None, "", "None"):
        dataset.data_device = "cuda"
    if getattr(dataset, "sh_degree", None) is None:
        dataset.sh_degree = 3

    gaussians = GaussianModel(dataset.sh_degree)
    load_iter = _infer_iter_from_checkpoint(checkpoint_path)
    scene = Scene(dataset, gaussians, load_iteration=load_iter, load_test_cameras=load_test_cameras)
    gaussians.training_setup(opt)
    model_params, _first_iter, extras = load_training_checkpoint(checkpoint_path)
    gaussians.restore(model_params, opt)
    restore_env_map_from_sidecar(gaussians, checkpoint_path)
    return scene, gaussians, extras


@torch.no_grad()
def run_compare(scene, gaussians_strict, gaussians_multi, trans_maps, trans_color_to_id, pipe, args):
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cams = pick_cameras(scene, args.compare_split, args.compare_max_views, args.compare_only_masked)
    if len(cams) == 0:
        raise RuntimeError("No cameras selected for comparison.")

    os.makedirs(args.out_dir, exist_ok=True)
    strict_dir = os.path.join(args.out_dir, "strict")
    multi_dir = os.path.join(args.out_dir, "multi_task")
    panel_dir = os.path.join(args.out_dir, "panel")
    gt_dir = os.path.join(args.out_dir, "gt")
    os.makedirs(strict_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for cam in tqdm(cams, desc="Offline compare rendering"):
        strict_pkg = render(cam, gaussians_strict, pipe, background, initial_stage=False, use_mirror_gate=False)
        multi_pkg = render(cam, gaussians_multi, pipe, background, initial_stage=False, use_mirror_gate=False)

        strict_rgb = strict_pkg["render"].clamp(0.0, 1.0)
        multi_refl = multi_pkg["render"].clamp(0.0, 1.0)
        multi_rgb = compose_with_transmission(
            multi_refl, cam, trans_maps, trans_color_to_id, args.strict_mask_erode
        ).clamp(0.0, 1.0)
        gt = cam.original_image.cuda().clamp(0.0, 1.0)

        diff = ((multi_rgb - strict_rgb).abs() * args.diff_gain).clamp(0.0, 1.0)
        panel = torch.cat([gt, strict_rgb, multi_rgb, diff], dim=2)

        name = cam.image_name
        save_image(gt, os.path.join(gt_dir, f"{name}_gt.png"))
        save_image(strict_rgb, os.path.join(strict_dir, f"{name}_strict.png"))
        save_image(multi_rgb, os.path.join(multi_dir, f"{name}_multi.png"))
        save_image(panel, os.path.join(panel_dir, f"{name}_gt_strict_multi_diffx{args.diff_gain:.1f}.png"))

    print(f"[DONE] Compare renders saved to: {args.out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Offline compare renderer: strict vs multi-task")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--strict_checkpoint", type=str, required=True)
    parser.add_argument("--multi_checkpoint", type=str, required=True)
    parser.add_argument("--strict_model_path", type=str, default=None)
    parser.add_argument("--multi_model_path", type=str, default=None)
    parser.add_argument("--compare_split", type=str, choices=["train", "test", "all"], default="train")
    parser.add_argument("--compare_max_views", type=int, default=20)
    parser.add_argument("--compare_only_masked", action="store_true", default=False)
    parser.add_argument("--strict_mask_erode", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="output/compare_strict_vs_multi")
    parser.add_argument("--diff_gain", type=float, default=4.0)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Defensive defaults for standalone offline rendering.
    if getattr(dataset, "data_device", None) in (None, "", "None"):
        dataset.data_device = "cuda"
    if getattr(dataset, "resolution", None) is None:
        dataset.resolution = -1
    if getattr(dataset, "sh_degree", None) is None:
        dataset.sh_degree = 3

    strict_model_path = args.strict_model_path or args.model_path or os.path.dirname(args.strict_checkpoint)
    multi_model_path = args.multi_model_path or os.path.dirname(args.multi_checkpoint)

    need_test_cameras = bool(args.compare_split in ["test", "all"])
    scene, gaussians_strict, _ = load_gaussians_from_checkpoint(
        dataset, opt, args.strict_checkpoint, strict_model_path, load_test_cameras=need_test_cameras
    )
    _, gaussians_multi, extras = load_gaussians_from_checkpoint(
        dataset, opt, args.multi_checkpoint, multi_model_path, load_test_cameras=need_test_cameras
    )
    trans_maps, trans_color_to_id = build_trans_maps_from_checkpoint_extras(extras, device="cuda")
    if trans_maps is None:
        print("[WARN] multi checkpoint has no transmission extras; compare will be reflection-only.")

    run_compare(scene, gaussians_strict, gaussians_multi, trans_maps, trans_color_to_id, pipe, args)
