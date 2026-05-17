"""
Stage 3: 逐窗口透射训练 (Per-Window Transmission Training)

从 Stage 2 检查点 (refl_chkpnt.pth) 加载，冻结所有高斯参数（包括
反射 env_map），引入 N 个独立的室内透射 CubemapEncoder，每个窗口
各自学习其对应的室内环境贴图。

实例掩码格式 (masks_instance/ 目录)
--------------------------------------
每张训练图像对应一张灰度 PNG，与图像同名（或 stem 相同）：
  像素值 0   → 背景（无透射）
  像素值 1   → 窗户 A
  像素值 2   → 窗户 B
  ...
  像素值 N   → 第 N 个窗户

训练原理（残差学习）
---------------------
1. 用冻结的高斯渲染基础+反射合成图：
       C_wo_trans = (1 - r) * C_base + r * C_refl
2. 计算残差（GT 减去已学好的部分）：
       Residual = GT - C_wo_trans
3. 对每个窗口 i，用其专属透射贴图拟合残差：
       Loss_i = L1(strength_i * C_trans_i, Residual) * mask_i
   其中 strength_i 是每个窗口可学习的透射强度标量。
4. 最终混合公式：
       C_final = C_wo_trans + Σ_i (strength_i * C_trans_i * mask_i)

未来扩展: 可给高斯增加 _trans_strength 参数（per-gaussian），通过
c3 光栅化得到逐像素透射强度图，替换现有的 per-window 标量强度。

用法:
    python train_stage3_transmission.py \\
        -s <数据集路径> -m <输出路径> \\
        --start_checkpoint <输出路径>/refl_chkpnt<iter>.pth \\
        --instance_mask_dir <数据集路径>/masks_instance \\
        --stage3_iters 10000

输出: <输出路径>/final_chkpnt<iter>.pth
      （包含冻结的高斯参数 + N 个透射贴图状态）
"""

import os
import sys
import cv2
import numpy as np
from argparse import ArgumentParser
from random import randint

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, get_trans_color
from scene import Scene, GaussianModel
from train import prepare_output_and_logger
from utils.general_utils import safe_state
from cubemapencoder import CubemapEncoder
from utils.checkpoint_utils import load_training_checkpoint, restore_env_map_from_sidecar

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ---------------------------------------------------------------------------
# 透射贴图模块
# ---------------------------------------------------------------------------

class WindowTransmissionMap(nn.Module):
    """
    单个窗口的透射环境贴图。
    包含：
      - CubemapEncoder (output_dim=3)：室内环境颜色
      - strength_logit：可学习的透射强度（logit 空间，sigmoid 输出 [0,1]）
    透射方向使用直通视线方向（不是反射方向）。
    """

    def __init__(self, cubemap_resol: int = 128, init_strength: float = 0.3):
        super().__init__()
        self.cubemap = CubemapEncoder(output_dim=3, resolution=cubemap_resol)
        # 用 logit 初始化，使 sigmoid(strength_logit) ≈ init_strength
        init_val = max(min(init_strength, 0.9999), 0.0001)
        init_logit = float(torch.log(torch.tensor(init_val / (1.0 - init_val))))
        self.strength_logit = nn.Parameter(torch.tensor(init_logit))

    @property
    def strength(self) -> torch.Tensor:
        """返回 [0,1] 的透射强度标量。"""
        return torch.sigmoid(self.strength_logit)

    def get_color(self, HWK, R, T) -> torch.Tensor:
        """
        用直通视线方向查询透射 Cubemap。
        返回: (3, H, W) float tensor
        """
        return get_trans_color(self.cubemap, HWK, R, T)


# ---------------------------------------------------------------------------
# 辅助函数：支持灰度掩码（像素值 1/2/3...）和 RGB 彩色掩码（颜色→窗口ID）
# ---------------------------------------------------------------------------

def _find_mask_file(mask_dir: str, image_name: str):
    """查找掩码文件路径，返回路径或 None。"""
    for ext in ('.png', '.PNG', '.jpg', '.jpeg'):
        path = os.path.join(mask_dir, image_name + ext)
        if os.path.exists(path):
            return path
    return None


def _is_rgb_mask_dir(mask_dir: str) -> bool:
    """
    检测掩码目录是灰度模式还是 RGB 彩色模式。
    读取第一张找到的掩码，若图像通道数 == 3 则为 RGB 模式。
    """
    for fname in sorted(os.listdir(mask_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(mask_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.ndim == 3 and img.shape[2] >= 3
    return False


def build_color_to_id(mask_dir: str) -> dict:
    """
    扫描所有 RGB 彩色掩码，提取所有非背景颜色，建立全局 颜色→窗口ID 映射。
    背景定义：(0, 0, 0)。
    映射按颜色排序以保证不同运行结果一致。

    返回: { (R, G, B): int_id }，int_id 从 1 开始。
    同一颜色在所有图像中映射到相同 ID → 对应同一个 trans_maps[ID-1]。
    """
    colors = set()
    for fname in sorted(os.listdir(mask_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(mask_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 提取所有唯一颜色（排除纯黑背景）
            pixels = img_rgb.reshape(-1, 3)
            unique = np.unique(pixels, axis=0)
            for c in unique:
                if not (c[0] == 0 and c[1] == 0 and c[2] == 0):
                    colors.add((int(c[0]), int(c[1]), int(c[2])))

    # 按颜色值排序确保映射稳定
    color_to_id = {color: idx + 1 for idx, color in enumerate(sorted(colors))}
    return color_to_id


def rgb_mask_to_instance(img_bgr: np.ndarray, color_to_id: dict) -> np.ndarray:
    """
    将 BGR 彩色掩码图转换为整数实例掩码（H, W），int64。
    背景像素保持 0，各颜色像素填入对应 ID。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    instance = np.zeros((H, W), dtype=np.int64)
    for color, cid in color_to_id.items():
        c = np.array(color, dtype=np.uint8)
        match = np.all(img_rgb == c[None, None, :], axis=2)
        instance[match] = cid
    return instance


def discover_windows(mask_dir: str):
    """
    自动检测掩码格式（灰度 or RGB），扫描所有掩码，
    返回 (N, color_to_id)：
      - N          : 窗口数量
      - color_to_id: RGB 模式下的颜色→ID 字典（灰度模式返回 None）
    同时打印发现的窗口映射信息供用户核验。
    """
    is_rgb = _is_rgb_mask_dir(mask_dir)
    has_jpeg = any(
        fname.lower().endswith((".jpg", ".jpeg"))
        for fname in os.listdir(mask_dir)
    )
    if has_jpeg:
        print(
            "[Stage 3][警告] 检测到 JPEG 掩码文件。JPEG 有损压缩可能引入颜色噪声，"
            "RGB 实例掩码建议使用 PNG。"
        )

    if is_rgb:
        color_to_id = build_color_to_id(mask_dir)
        N = len(color_to_id)
        print(f"[Stage 3] 检测到 RGB 彩色掩码，发现 {N} 个窗口实例：")
        for color, cid in sorted(color_to_id.items(), key=lambda x: x[1]):
            r, g, b = color
            print(f"  窗口 {cid:2d}  ←  颜色 RGB({r:3d}, {g:3d}, {b:3d})")
        return N, color_to_id
    else:
        # 灰度模式：像素值 1/2/3... 直接作为窗口 ID
        max_val = 0
        for fname in sorted(os.listdir(mask_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(mask_dir, fname)
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    max_val = max(max_val, int(mask.max()))
        N = max_val
        print(f"[Stage 3] 检测到灰度掩码，发现 {N} 个窗口实例（像素值 1~{N}）。")
        return N, None


def load_instance_mask(mask_dir: str, image_name: str,
                       color_to_id=None,
                       device: str = 'cuda'):
    """
    加载单张图像对应的实例掩码。
    - 灰度模式（color_to_id=None）：像素值直接作为窗口 ID。
    - RGB 模式（color_to_id 为字典）：颜色映射到窗口 ID。
    返回: (H, W) long tensor，或 None（未找到）。
    """
    path = _find_mask_file(mask_dir, image_name)
    if path is None:
        return None

    if color_to_id is not None:
        # RGB 彩色掩码
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        instance = rgb_mask_to_instance(img, color_to_id)
        return torch.from_numpy(instance).to(device)
    else:
        # 灰度掩码
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        return torch.from_numpy(mask.astype(np.int64)).to(device)


def preload_instance_masks(mask_dir: str, cameras,
                           color_to_id=None,
                           device: str = 'cuda') -> dict:
    """
    预加载所有相机对应的实例掩码到字典（避免训练时反复 I/O）。
    返回: {image_name: (H, W) long tensor}
    """
    masks = {}
    for cam in cameras:
        m = load_instance_mask(mask_dir, cam.image_name,
                               color_to_id=color_to_id, device=device)
        if m is not None and m.max() > 0:
            masks[cam.image_name] = m
    return masks


def freeze_gaussians(gaussians: GaussianModel):
    """
    冻结所有高斯几何参数及反射 env_map（Stage 2 已学好，保持不变）。
    """
    params_to_freeze = [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians._refl_strength,
        gaussians._mirror_weight,
    ]
    for param in params_to_freeze:
        param.requires_grad_(False)

    if gaussians.env_map is not None:
        for param in gaussians.env_map.parameters():
            param.requires_grad_(False)

    # 将 optimizer 中所有参数的 lr 置零（防止意外更新）
    for group in gaussians.optimizer.param_groups:
        group["lr"] = 0.0

    print("[Stage 3] 所有高斯参数（含反射 env_map）已冻结。")


def trans_tv_loss(trans_map: WindowTransmissionMap) -> torch.Tensor:
    """透射 Cubemap 的 Total Variation 正则化（抑制噪声纹理）。"""
    texture = trans_map.cubemap.params["Cubemap_texture"]  # (6, 3, R, R)
    tv_h = (texture[:, :, :, 1:] - texture[:, :, :, :-1]).abs().mean()
    tv_v = (texture[:, :, 1:, :] - texture[:, :, :-1, :]).abs().mean()
    return tv_h + tv_v


def trans_contribution(trans_map: WindowTransmissionMap, cam, signed: bool) -> torch.Tensor:
    """Return this window's additive transmission contribution before masking."""
    trans_color = trans_map.get_color(cam.HWK, cam.R, cam.T)
    if signed:
        trans_color = 2.0 * trans_color - 1.0
    return trans_map.strength * trans_color


def get_stage2_weaken_ratio(step: int, args) -> float:
    if not args.weaken_stage2:
        return 0.0
    if args.weaken_period <= 0:
        ratio = args.weaken_max
    else:
        phase = (step % args.weaken_period) / float(args.weaken_period)
        ratio = args.weaken_min + (args.weaken_max - args.weaken_min) * 0.5 * (
            1.0 + np.cos(2.0 * np.pi * phase)
        )
    if args.weaken_warmup > 0:
        ratio *= min(1.0, step / float(args.weaken_warmup))
    return float(max(0.0, min(1.0, ratio)))


def get_stage2_eval_weaken_ratio(args) -> float:
    if not args.weaken_stage2:
        return 0.0
    if args.weaken_eval_ratio >= 0:
        return float(max(0.0, min(1.0, args.weaken_eval_ratio)))
    return float(max(0.0, min(1.0, args.weaken_max)))


def weaken_stage2_render(render_image: torch.Tensor, mask3: torch.Tensor, background: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0:
        return render_image
    bg = background[:, None, None].to(device=render_image.device, dtype=render_image.dtype)
    return render_image * (1.0 - ratio * mask3) + bg * (ratio * mask3)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def stage3_training(dataset, opt, pipe, args):
    if args.start_checkpoint is None:
        raise ValueError("Stage 3 需要 --start_checkpoint（Stage 2 输出的 refl_chkpnt.pth）。")

    if not os.path.isdir(args.instance_mask_dir):
        raise ValueError(
            f"--instance_mask_dir 目录不存在: {args.instance_mask_dir}\n"
            "请提供包含实例掩码 PNG 的目录（像素值 1=窗户A, 2=窗户B...）。"
        )

    prepare_output_and_logger(dataset)

    # ---- 加载 Stage 2 高斯模型 ----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    model_params, first_iter, _extras = load_training_checkpoint(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    restore_env_map_from_sidecar(gaussians, args.start_checkpoint)
    print(f"[Stage 3] 已从第 {first_iter} 次迭代的检查点加载。")
    print(f"[Stage 3] 高斯数量: {gaussians.get_xyz.shape[0]}")

    # ---- 冻结高斯（含 env_map）----
    freeze_gaussians(gaussians)

    # ---- 发现窗口数量 N（自动识别灰度 or RGB 彩色掩码）----
    N, color_to_id = discover_windows(args.instance_mask_dir)
    if N == 0:
        raise RuntimeError(
            f"在 {args.instance_mask_dir} 中未找到任何窗口实例。\n"
            "灰度模式：确认掩码像素值包含 1、2、3... 等非零值。\n"
            "RGB 模式：确认掩码包含非纯黑色（0,0,0）的彩色区域。"
        )

    # ---- 创建 N 个透射贴图 ----
    trans_maps = nn.ModuleList([
        WindowTransmissionMap(
            cubemap_resol=args.cubemap_resol,
            init_strength=args.init_trans_strength
        )
        for _ in range(N)
    ]).cuda()
    print(f"[Stage 3] 已创建 {N} 个 WindowTransmissionMap (分辨率={args.cubemap_resol})。")

    # ---- 仅对透射贴图建立优化器 ----
    optimizer = torch.optim.Adam(
        trans_maps.parameters(), lr=args.trans_lr, eps=1e-15
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- 预加载实例掩码（传入 color_to_id 以支持 RGB 模式）----
    print("[Stage 3] 预加载实例掩码...")
    all_cams = scene.getTrainCameras()
    instance_masks = preload_instance_masks(
        args.instance_mask_dir, all_cams,
        color_to_id=color_to_id
    )

    train_cams = [c for c in all_cams if c.image_name in instance_masks]
    if not train_cams:
        raise RuntimeError(
            f"没有训练相机的实例掩码文件匹配。\n"
            f"请确认 {args.instance_mask_dir} 中文件名与图像名一致。"
        )
    print(f"[Stage 3] 有实例掩码的训练相机: {len(train_cams)}/{len(all_cams)}")

    # ---- 训练循环 ----
    view_stack = []
    progress = tqdm(range(1, args.stage3_iters + 1), desc="Stage 3: 透射训练")
    ema_loss = 0.0

    for step in progress:
        if not view_stack:
            view_stack = train_cams.copy()
        cam = view_stack.pop(randint(0, len(view_stack) - 1))

        instance_mask = instance_masks[cam.image_name]   # (H, W) long
        gt = cam.original_image.cuda()                    # (3, H, W)

        # 冻结的高斯渲染（无需梯度，节省显存和计算）
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe, background, initial_stage=False)
            # C_wo_trans = (1-r)*C_base + r*C_refl，即 Stage 2 的完整合成结果
            C_wo_trans = render_pkg["render"]   # (3, H, W)
            union_mask3 = (instance_mask > 0).float().unsqueeze(0).expand_as(C_wo_trans)
            weaken_ratio = get_stage2_weaken_ratio(step, args)
            C_stage2 = weaken_stage2_render(C_wo_trans, union_mask3, background, weaken_ratio)

        # Residual is still available for ablations, but the default loss below
        # directly supervises the final composited image.
        residual = (gt - C_stage2).detach()   # (3, H, W)

        # 对所有活跃窗口累加损失
        total_loss = torch.zeros(1, device="cuda", requires_grad=False)
        loss_list = []
        active_windows = 0

        for i in range(N):
            pixel_mask = (instance_mask == (i + 1))     # (H, W) bool
            n_pixels = pixel_mask.sum().item()
            if n_pixels < args.min_pixels:
                continue                                  # 该窗口在此视角不可见，跳过

            # 掩码 L1 损失（仅在该窗口区域）
            mask3 = pixel_mask.float().unsqueeze(0).expand(3, -1, -1)    # (3, H, W)
            C_contrib_i = trans_contribution(trans_maps[i], cam, args.signed_transmission)
            if args.trans_loss_mode == "residual":
                target_i = residual
                loss_i = (C_contrib_i - target_i).abs() * mask3
            elif args.trans_loss_mode == "positive_residual":
                target_i = residual.clamp_min(0.0)
                loss_i = (C_contrib_i - target_i).abs() * mask3
            else:
                pred_i = C_stage2 + C_contrib_i * mask3
                loss_i = (pred_i - gt).abs() * mask3
            loss_i = loss_i.sum() / (mask3.sum() + 1e-6)

            if args.lambda_trans_mag > 0:
                loss_i = loss_i + args.lambda_trans_mag * (C_contrib_i.abs() * mask3).sum() / (mask3.sum() + 1e-6)

            # TV 正则化（平滑 Cubemap 纹理）
            if args.lambda_trans_tv > 0:
                loss_i = loss_i + args.lambda_trans_tv * trans_tv_loss(trans_maps[i])

            loss_list.append(loss_i)
            active_windows += 1

        if active_windows == 0:
            continue

        # 所有活跃窗口损失取平均后反传
        total_loss = sum(loss_list) / active_windows
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_loss = 0.4 * total_loss.item() + 0.6 * ema_loss
        if step % args.log_interval == 0:
            strengths_str = "[" + ", ".join(
                f"win{i+1}={trans_maps[i].strength.item():.3f}" for i in range(N)
            ) + "]"
            progress.set_postfix({
                "loss":     f"{ema_loss:.6f}",
                "active":   active_windows,
                "weaken":   f"{weaken_ratio:.3f}",
                "strength": strengths_str,
            })

        global_iter = first_iter + step
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(scene.model_path, f"trans_chkpnt{global_iter}.pth")
            _save_stage3_checkpoint(
                gaussians, trans_maps, N, args, global_iter, ckpt_path, color_to_id=color_to_id
            )
            print(f"\n[Stage 3] 检查点已保存: {ckpt_path}")

        if args.debug_every > 0 and step % args.debug_every == 0:
            _save_stage3_debug(scene.model_path, step, cam, C_wo_trans, C_stage2, gt, instance_mask, trans_maps, args)

    # ---- 保存最终检查点 ----
    final_iter = first_iter + args.stage3_iters
    scene.save(final_iter)
    final_ckpt = os.path.join(scene.model_path, f"final_chkpnt{final_iter}.pth")
    _save_stage3_checkpoint(
        gaussians, trans_maps, N, args, final_iter, final_ckpt, color_to_id=color_to_id
    )

    print(f"\n[Stage 3 完成] 最终检查点: {final_ckpt}")
    print("各窗口透射强度:")
    for i in range(N):
        print(f"  窗户 {i+1}: strength = {trans_maps[i].strength.item():.4f}")

    return final_ckpt


def _save_stage3_checkpoint(gaussians, trans_maps, N, args, iteration, path, color_to_id=None):
    """
    保存 Stage 3 检查点：高斯模型 + N 个透射贴图。
    加载方式示例（渲染或评估时）:
        ckpt = torch.load(path)
        gaussians.restore(ckpt["gaussians"], opt)
        for i, state in enumerate(ckpt["trans_maps_state"]):
            trans_maps[i].load_state_dict(state)
    """
    torch.save(
        {
            "gaussians":        gaussians.capture(),
            "first_iter":       iteration,
            "n_windows":        N,
            "cubemap_resol":    args.cubemap_resol,
            "signed_transmission": bool(args.signed_transmission),
            "trans_loss_mode":  args.trans_loss_mode,
            "weaken_stage2":    bool(args.weaken_stage2),
            "weaken_eval_ratio": get_stage2_eval_weaken_ratio(args),
            "weaken_bg_color":  "white" if args.white_background else "black",
            "trans_maps_state": [tm.state_dict() for tm in trans_maps],
            # 保存颜色映射，RGB 模式下用于推理时还原窗口归属
            "color_to_id":      color_to_id,
        },
        path,
    )


@torch.no_grad()
def _save_stage3_debug(model_path, step, cam, C_wo_trans, C_stage2, gt, instance_mask, trans_maps, args):
    debug_dir = os.path.join(model_path, "stage3_trans_debug")
    os.makedirs(debug_dir, exist_ok=True)
    pred = C_stage2.clone()
    contrib = torch.zeros_like(pred)
    for i in range(len(trans_maps)):
        pixel_mask = instance_mask == (i + 1)
        if pixel_mask.sum().item() < args.min_pixels:
            continue
        mask3 = pixel_mask.float().unsqueeze(0).expand_as(pred)
        c_i = trans_contribution(trans_maps[i], cam, args.signed_transmission) * mask3
        pred = pred + c_i
        contrib = contrib + c_i
    pred = pred.clamp(0.0, 1.0)
    contrib_vis = contrib
    if args.signed_transmission:
        contrib_vis = contrib_vis * 0.5 + 0.5
    else:
        contrib_vis = contrib_vis.clamp(0.0, 1.0)
    residual_vis = ((gt - C_stage2) * 0.5 + 0.5).clamp(0.0, 1.0)
    panel = torch.cat([
        gt.clamp(0.0, 1.0),
        C_wo_trans.clamp(0.0, 1.0),
        C_stage2.clamp(0.0, 1.0),
        pred,
        contrib_vis.clamp(0.0, 1.0),
        residual_vis,
    ], dim=2)
    save_image(panel, os.path.join(debug_dir, f"{step:06d}_{cam.image_name}_gt_refl_weaken_pred_trans_residual.png"))


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 3: 逐窗口透射训练")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--start_checkpoint", type=str, required=True,
        help="Stage 2 输出的检查点路径（refl_chkpnt<iter>.pth）"
    )
    parser.add_argument(
        "--instance_mask_dir", type=str, required=True,
        help="实例掩码目录（灰度 PNG，像素值 1=窗户A, 2=窗户B...）"
    )
    parser.add_argument(
        "--stage3_iters", type=int, default=10000,
        help="Stage 3 训练迭代次数（默认 10000）"
    )
    parser.add_argument(
        "--cubemap_resol", type=int, default=128,
        help="每个透射 Cubemap 的分辨率（默认 128）"
    )
    parser.add_argument(
        "--trans_lr", type=float, default=0.05,
        help="透射贴图的学习率（默认 0.05）"
    )
    parser.add_argument(
        "--init_trans_strength", type=float, default=0.3,
        help="各窗口初始透射强度，范围 (0,1)，默认 0.3"
    )
    parser.add_argument(
        "--lambda_trans_tv", type=float, default=1e-5,
        help="透射 Cubemap TV 正则化权重（默认 1e-5，设 0 禁用）"
    )
    parser.add_argument(
        "--lambda_trans_mag", type=float, default=0.0,
        help="透射贡献幅度正则，防止局部过强补偿（默认 0）"
    )
    parser.add_argument(
        "--trans_loss_mode",
        choices=["final", "residual", "positive_residual"],
        default="final",
        help="透射训练目标：final 直接监督最终合成图；residual 为旧逻辑；positive_residual 只拟合正残差。"
    )
    parser.add_argument(
        "--signed_transmission",
        action="store_true",
        help="允许透射贡献为负，用于修正 Stage2 已经过亮的玻璃区域。"
    )
    parser.add_argument(
        "--weaken_stage2",
        action="store_true",
        help="在玻璃 mask 内弱化 Stage2 渲染，再训练透射补偿。评估脚本会读取 checkpoint 中的弱化比例。"
    )
    parser.add_argument("--weaken_min", type=float, default=0.0)
    parser.add_argument("--weaken_max", type=float, default=0.4)
    parser.add_argument("--weaken_period", type=int, default=1000)
    parser.add_argument("--weaken_warmup", type=int, default=1000)
    parser.add_argument(
        "--weaken_eval_ratio",
        type=float,
        default=-1.0,
        help="保存到 checkpoint 供评估使用的固定弱化比例；负值表示使用 weaken_max。"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=100,
        help="计算损失所需的窗口最小像素数（默认 100）"
    )
    parser.add_argument("--debug_every", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--log_interval",     type=int, default=100)
    parser.add_argument("--quiet",            action="store_true")

    args = parser.parse_args(sys.argv[1:])
    print("Stage 3 逐窗口透射训练: " + args.model_path)
    safe_state(args.quiet)
    stage3_training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nStage 3 训练完成。")
