import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def find_spec_mask_path(source_path, spec_mask_dir, image_name, image_path=None):
    if not spec_mask_dir:
        return None
    mask_dir = os.path.join(source_path, spec_mask_dir)
    candidates = [
        os.path.join(mask_dir, f"{image_name}.png"),
        os.path.join(mask_dir, f"mask_{image_name}.png"),
        os.path.join(mask_dir, f"IMG_{image_name}.png"),
        os.path.join(mask_dir, f"mask_IMG_{image_name}.png"),
    ]
    if image_path is not None:
        basename = os.path.basename(image_path)
        stem, _ext = os.path.splitext(basename)
        candidates.extend(
            [
                os.path.join(mask_dir, basename),
                os.path.join(mask_dir, f"{stem}.png"),
                os.path.join(mask_dir, f"mask_{basename}"),
                os.path.join(mask_dir, f"mask_{stem}.png"),
                os.path.join(mask_dir, f"IMG_{stem}.png"),
                os.path.join(mask_dir, f"mask_IMG_{stem}.png"),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _cache_path(mask_path, resolution):
    return f"{mask_path}.{int(resolution[0])}x{int(resolution[1])}.region_cache.pt"


def _build_compact_labels_from_gray(mask_np):
    label_np = mask_np.astype(np.int32)
    unique_vals = np.unique(label_np)
    unique_vals = unique_vals[unique_vals != 0]
    if unique_vals.size == 0:
        return None, []
    compact = np.zeros_like(label_np, dtype=np.int32)
    # One-time mapping, no longer requiring full-map scans for each region
    pos = np.searchsorted(unique_vals, label_np)
    valid = label_np != 0
    compact[valid] = pos[valid] + 1
    region_ids = [int(v) for v in unique_vals.tolist()]
    return compact, region_ids


def _build_compact_labels_from_rgb(mask_np):
    if mask_np.shape[2] > 3:
        mask_np = mask_np[:, :, :3]
    # Pack RGB into a single uint32 to avoid performing a slow unique operation on [N,3].
    packed = (
        (mask_np[:, :, 0].astype(np.uint32) << 16)
        | (mask_np[:, :, 1].astype(np.uint32) << 8)
        | mask_np[:, :, 2].astype(np.uint32)
    )
    unique_vals = np.unique(packed)
    unique_vals = unique_vals[unique_vals != 0]
    if unique_vals.size == 0:
        return None, []
    compact = np.zeros_like(packed, dtype=np.int32)
    pos = np.searchsorted(unique_vals, packed)
    valid = packed != 0
    compact[valid] = pos[valid] + 1
    region_ids = [
        (int(v >> 16) & 255, int(v >> 8) & 255, int(v) & 255)
        for v in unique_vals.tolist()
    ]
    return compact, region_ids


def load_spec_mask_with_regions(mask_path, resolution):
    # Split an annotated image into:
    # 1. union_mask: The combined mask for all glass regions, shape [1,H,W]
    # 2. region_labels: Which glass region each pixel belongs to, shape [H,W], where 0 represents background and 1..K represents the Kth glass region
    #
    # We no longer store the dense region mask [K,H,W] here, as it consumes significant memory when resolutions are high and regions are numerous.
    # During training, frequent transfers between CPU/GPU would cause “Loading Cameras” delays and slow down each training step.
    cache_path = _cache_path(mask_path, resolution)
    if os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path, weights_only=False)
        except TypeError:
            cached = torch.load(cache_path)
        union_mask = cached.get("union_mask", None)
        region_labels = cached.get("region_labels", None)
        region_ids = cached.get("region_ids", [])
        if union_mask is not None and region_labels is not None:
            return union_mask.float(), region_labels.long(), region_ids

    mask_image = Image.open(mask_path)
    mask_image = mask_image.resize(resolution, resample=Image.NEAREST)
    mask_np = np.array(mask_image)

    if mask_np.ndim == 2:
        compact, region_ids = _build_compact_labels_from_gray(mask_np)
    else:
        compact, region_ids = _build_compact_labels_from_rgb(mask_np)

    if compact is None:
        return None, None, []

    region_labels = torch.from_numpy(compact).long()
    union_mask = (region_labels > 0).unsqueeze(0).float()
    try:
        torch.save(
            {
                "union_mask": union_mask.cpu(),
                "region_labels": region_labels.cpu(),
                "region_ids": region_ids,
            },
            cache_path,
        )
    except Exception:
        pass
    return union_mask, region_labels, region_ids


def get_spec_mask_union(camera):
    mask = getattr(camera, "spec_mask", None)
    if mask is None:
        return None
    return (mask > 0.5).float()


def _build_region_masks_from_labels(region_labels, num_regions):
    return (region_labels.unsqueeze(0) == torch.arange(1, num_regions + 1, device=region_labels.device).view(-1, 1, 1)).float()


def erode_region_labels(region_labels, kernel_size, return_kept=False):
    if region_labels is None:
        return (None, []) if return_kept else None
    if kernel_size <= 1:
        if return_kept:
            num_regions = int(region_labels.max().item()) if region_labels.numel() > 0 else 0
            return region_labels, list(range(1, num_regions + 1))
        return region_labels
    if kernel_size % 2 == 0:
        kernel_size += 1
    if region_labels.ndim != 2:
        raise ValueError("region_labels must have shape [H,W].")
    num_regions = int(region_labels.max().item())
    if num_regions <= 0:
        return (region_labels, []) if return_kept else region_labels
    masks = _build_region_masks_from_labels(region_labels, num_regions)
    inv = 1.0 - masks.unsqueeze(1)
    inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = (1.0 - inv[:, 0]).clamp(0.0, 1.0) > 0.5
    compact = torch.zeros_like(region_labels)
    kept = 0
    kept_original_ids = []
    for idx in range(num_regions):
        region = eroded[idx]
        if region.any():
            kept += 1
            compact[region] = kept
            kept_original_ids.append(idx + 1)
    if return_kept:
        return compact, kept_original_ids
    return compact


def get_spec_mask_region_labels(camera, erode_kernel=0, return_region_ids=False):
    labels = getattr(camera, "spec_mask_region_labels", None)
    raw_region_ids = getattr(camera, "spec_mask_region_ids", None)
    if labels is None:
        union_mask = get_spec_mask_union(camera)
        if union_mask is None:
            return (None, None) if return_region_ids else None
        labels = union_mask[0].long().cpu()
        raw_region_ids = None
    else:
        labels = labels.cpu()

    cache_name = f"_spec_mask_region_labels_erode_{int(erode_kernel)}"
    cached = getattr(camera, cache_name, None)
    cache_ids_name = f"{cache_name}_ids"
    cached_region_ids = getattr(camera, cache_ids_name, None)
    if cached is None:
        if erode_kernel and erode_kernel > 1:
            cached, kept_original_ids = erode_region_labels(labels, int(erode_kernel), return_kept=True)
            cached = cached.cpu()
            if raw_region_ids is not None:
                cached_region_ids = [raw_region_ids[i - 1] for i in kept_original_ids]
            else:
                cached_region_ids = kept_original_ids
        else:
            cached = labels
            num_regions = int(labels.max().item()) if labels.numel() > 0 else 0
            if raw_region_ids is not None:
                cached_region_ids = list(raw_region_ids)
            else:
                cached_region_ids = list(range(1, num_regions + 1))
        setattr(camera, cache_name, cached)
        setattr(camera, cache_ids_name, cached_region_ids)

    device = getattr(camera, "data_device", None)
    if device is None:
        device = cached.device
    labels_dev = cached.to(device=device, non_blocking=True)
    if return_region_ids:
        return labels_dev, cached_region_ids
    return labels_dev


def get_spec_mask_regions(camera):
    region_labels = get_spec_mask_region_labels(camera, erode_kernel=0)
    if region_labels is None:
        return None
    num_regions = int(region_labels.max().item())
    if num_regions <= 0:
        return None
    return _build_region_masks_from_labels(region_labels, num_regions)


def erode_regions(region_masks, kernel_size):
    if region_masks is None:
        return None
    if kernel_size <= 1:
        return region_masks
    if region_masks.ndim == 2:
        region_masks = region_masks.unsqueeze(0)
    if kernel_size % 2 == 0:
        kernel_size += 1
    eroded = []
    for region in region_masks:
        inv = 1.0 - region.unsqueeze(0)
        inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        eroded.append((1.0 - inv[0]).clamp(0.0, 1.0))
    return torch.stack(eroded, dim=0)


def regionwise_weighted_l1(pred, gt, region_masks, extra_weight=None):
    if region_masks is None:
        return None
    if region_masks.ndim == 2:
        region_masks = region_masks.unsqueeze(0)
    abs_diff = torch.abs(pred - gt)
    losses = []
    for region in region_masks:
        region = region.unsqueeze(0)
        if extra_weight is not None:
            region = region * extra_weight
        denom = pred.shape[0] * region.sum() + 1e-6
        if float(denom.detach().item()) <= 1e-6:
            continue
        losses.append((abs_diff * region.expand_as(pred)).sum() / denom)
    if len(losses) == 0:
        return None
    return torch.stack(losses, dim=0).mean()


def regionwise_weighted_l1_from_labels(pred, gt, region_labels, extra_weight=None):
    if region_labels is None:
        return None
    if region_labels.ndim != 2:
        raise ValueError("region_labels must have shape [H,W].")

    abs_diff = torch.abs(pred - gt).mean(dim=0)
    labels_flat = region_labels.reshape(-1)
    valid = labels_flat > 0
    if not bool(valid.any()):
        return None

    values = abs_diff.reshape(-1)[valid]
    labels = labels_flat[valid] - 1
    if extra_weight is not None:
        values = values * extra_weight.reshape(-1)[valid]

    num_regions = int(region_labels.max().item())
    region_sum = torch.zeros(num_regions, device=pred.device, dtype=values.dtype)
    region_sum.scatter_add_(0, labels, values)

    ones = torch.ones_like(values)
    region_count = torch.zeros(num_regions, device=pred.device, dtype=values.dtype)
    region_count.scatter_add_(0, labels, ones)

    keep = region_count > 0
    if not bool(keep.any()):
        return None
    return (region_sum[keep] / (region_count[keep] + 1e-6)).mean()


def regionwise_weighted_l1_from_labels_with_info(pred, gt, region_labels, region_ids=None, extra_weight=None):
    # return：
    # 1. Region-wise average L1 (consistent with regionwise_weighted_l1_from_labels)
    # 2. Individual loss per region / pixel_count / corresponding region_id (RGB or grayscale ID)
    if region_labels is None:
        return None, []
    if region_labels.ndim != 2:
        raise ValueError("region_labels must have shape [H,W].")

    abs_diff = torch.abs(pred - gt).mean(dim=0)
    labels_flat = region_labels.reshape(-1)
    valid = labels_flat > 0
    if not bool(valid.any()):
        return None, []

    values = abs_diff.reshape(-1)[valid]
    labels = labels_flat[valid] - 1
    if extra_weight is not None:
        values = values * extra_weight.reshape(-1)[valid]

    num_regions = int(region_labels.max().item())
    region_sum = torch.zeros(num_regions, device=pred.device, dtype=values.dtype)
    region_sum.scatter_add_(0, labels, values)

    ones = torch.ones_like(values)
    region_count = torch.zeros(num_regions, device=pred.device, dtype=values.dtype)
    region_count.scatter_add_(0, labels, ones)

    keep = region_count > 0
    if not bool(keep.any()):
        return None, []

    region_mean = region_sum / (region_count + 1e-6)
    info = []
    for idx in range(num_regions):
        if not bool(keep[idx].item()):
            continue
        item = {
            "region_index": idx + 1,
            "pixel_count": int(region_count[idx].detach().item()),
            "loss": float(region_mean[idx].detach().item()),
        }
        if region_ids is not None and idx < len(region_ids):
            item["region_id"] = region_ids[idx]
        info.append(item)
    return region_mean[keep].mean(), info
