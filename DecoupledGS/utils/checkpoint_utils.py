import os
import re
import torch


def _torch_load_any(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def load_training_checkpoint(path):
    """
    Unified loader for both legacy and extended checkpoint formats.

    Returns:
        gaussians_state: state payload consumable by gaussians.restore(...)
        first_iter: int
        extras: dict with optional auxiliary states (e.g. transmission maps)
    """
    payload = _torch_load_any(path)

    # Legacy format: (gaussians.capture(), iteration)
    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        return payload[0], int(payload[1]), {}

    # Extended format: {"gaussians": ..., "first_iter": ..., ...}
    if isinstance(payload, dict) and "gaussians" in payload:
        first_iter = int(payload.get("first_iter", payload.get("iteration", 0)))
        extras = {k: v for k, v in payload.items() if k not in ("gaussians", "first_iter", "iteration")}
        return payload["gaussians"], first_iter, extras

    raise ValueError(f"Unsupported checkpoint format: {path}")


def save_training_checkpoint(path, gaussians_state, iteration, extras=None):
    """
    Save in legacy tuple format when no extras are provided; otherwise save dict.
    """
    if extras is None or len(extras) == 0:
        torch.save((gaussians_state, int(iteration)), path)
        return
    payload = {"gaussians": gaussians_state, "first_iter": int(iteration)}
    payload.update(extras)
    torch.save(payload, path)


def restore_env_map_from_sidecar(gaussians, checkpoint_path):
    """
    Restore env_map from companion point_cloud.map using checkpoint iteration id.
    """
    if checkpoint_path is None or gaussians.env_map is None:
        return
    ckpt_name = os.path.basename(checkpoint_path)
    match = re.search(r"(\d+)\.pth$", ckpt_name)
    if match is None:
        return
    iter_id = int(match.group(1))
    model_dir = os.path.dirname(checkpoint_path)
    map_path = os.path.join(model_dir, "point_cloud", f"iteration_{iter_id}", "point_cloud.map")
    if not os.path.exists(map_path):
        print(f"[WARN] point_cloud.map not found for start checkpoint iteration: {map_path}")
        return
    env_state = _torch_load_any(map_path)
    gaussians.env_map.load_state_dict(env_state)
    print(f"[INFO] Restored env_map from {map_path}")
