
import os
import re
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_env_map
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import cv2, time
import numpy as np
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
from utils.image_utils import psnr
import torchvision
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

'''
Add decay opacity for refl gaussians (banned)
Add reset refl ratio
Add refl smooth loss (banned)
SH0, and no densify (banned)
INIT_ITER 5000, cbmp_lr = 0.01 $

densify -> 30k
densify_intv in prop -> 100
'''

def set_stage4_trainable(gaussians, opt):
    if not opt.stage4_env_finetune:
        return
    for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                  gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians._mirror_weight]:
        param.requires_grad_(False)
    gaussians._refl_strength.requires_grad_(bool(opt.stage4_train_refl))
    for param in gaussians.env_map.parameters():
        param.requires_grad_(True)

def set_refl_lr(gaussians, lr):
    for group in gaussians.optimizer.param_groups:
        if group["name"] == "refl":
            group["lr"] = lr
            break


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

def edge_aware_depth_smooth_loss(depth, image, valid, edge_weight):
    dx_depth = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
    dy_depth = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
    dx_image = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    dy_image = (image[:, 1:, :] - image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    wx = torch.exp(-edge_weight * dx_image)
    wy = torch.exp(-edge_weight * dy_image)
    vx = valid[:, :, 1:] * valid[:, :, :-1]
    vy = valid[:, 1:, :] * valid[:, :-1, :]
    return ((dx_depth * wx * vx).sum() + (dy_depth * wy * vy).sum()) / (vx.sum() + vy.sum() + 1e-6)

def edge_aware_normal_smooth_loss(normal, image, valid, edge_weight):
    dx_normal = (normal[:, :, 1:] - normal[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    dy_normal = (normal[:, 1:, :] - normal[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    dx_image = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    dy_image = (image[:, 1:, :] - image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    wx = torch.exp(-edge_weight * dx_image)
    wy = torch.exp(-edge_weight * dy_image)
    vx = valid[:, :, 1:] * valid[:, :, :-1]
    vy = valid[:, 1:, :] * valid[:, :-1, :]
    return ((dx_normal * wx * vx).sum() + (dy_normal * wy * vy).sum()) / (vx.sum() + vy.sum() + 1e-6)

def global_surface_scale_loss(gaussians):
    scales = gaussians.get_scaling
    sorted_scales, _ = torch.sort(scales, dim=-1)
    return (sorted_scales[:, 0] / (sorted_scales[:, 1] + 1e-6)).mean()

def save_envmap_preview(scene, iteration):
    env_dir = os.path.join(scene.model_path, "envmap_preview")
    os.makedirs(env_dir, exist_ok=True)
    env_res = render_env_map(scene.gaussians)
    for env_name, env_im in env_res.items():
        save_path = os.path.join(env_dir, f"{env_name}_{iteration:06d}.png")
        torchvision.utils.save_image(env_im.clamp(0.0, 1.0), save_path)


def pick_preview_cameras(scene, split, names, max_views):
    if split == "test":
        cams = scene.getTestCameras()
    elif split == "all":
        cams = scene.getTrainCameras() + scene.getTestCameras()
    else:
        cams = scene.getTrainCameras()

    cams = sorted(cams, key=lambda c: c.image_name)
    if names:
        wanted = set(names)
        cams = [c for c in cams if c.image_name in wanted]
    if max_views > 0:
        cams = cams[:max_views]
    return cams


@torch.no_grad()
def save_rgb_previews(scene, pipe, background, iteration, split="train", names=None, max_views=3, out_dir_name="save_preview"):
    cams = pick_preview_cameras(scene, split=split, names=names, max_views=max_views)
    if len(cams) == 0:
        print("[WARN] No preview cameras selected; skip preview rendering.")
        return

    out_dir = os.path.join(scene.model_path, out_dir_name, f"iter_{iteration:06d}")
    os.makedirs(out_dir, exist_ok=True)
    for cam in cams:
        res = render(cam, scene.gaussians, pipe, background, initial_stage=False)
        pred = res["render"].clamp(0.0, 1.0)
        gt = cam.original_image.cuda().clamp(0.0, 1.0)
        diff = (pred - gt).abs().clamp(0.0, 1.0)
        panel = torch.cat([gt, pred, diff], dim=2)
        torchvision.utils.save_image(panel, os.path.join(out_dir, f"{cam.image_name}_gt_pred_diff.png"))
    print(f"[INFO] Saved preview renders to {out_dir}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, preview_cfg=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    INIT_UNITIL_ITER = opt.init_until_iter #3000
    FR_OPTIM_FROM_ITER = opt.feature_rest_from_iter
    NORMAL_PROP_UNTIL_ITER = opt.normal_prop_until_iter + opt.longer_prop_iter #24_000
    OPAC_LR0_INTERVAL = opt.opac_lr0_interval # 200
    DENSIFIDATION_INTERVAL_WHEN_PROP = opt.densification_interval_when_prop #500
    
    TOT_ITER = opt.iterations + opt.longer_prop_iter + 1
    DENSIFY_UNTIL_ITER = opt.densify_until_iter + opt.longer_prop_iter

    # for real scenes
    USE_ENV_SCOPE = opt.use_env_scope # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians) # init all parameters(pos,scale,rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        restore_env_map_from_checkpoint_sidecar(gaussians, checkpoint)
    if opt.stage4_env_finetune and not checkpoint:
        raise ValueError("stage4_env_finetune requires --start_checkpoint.")
    set_stage4_trainable(gaussians, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print('propagation until: {}'.format(NORMAL_PROP_UNTIL_ITER))
    print('densify until: {}'.format(DENSIFY_UNTIL_ITER))
    print('total iter: {}'.format(TOT_ITER))

    initial_stage = True
    if opt.stage4_env_finetune:
        initial_stage = False
        print("[INFO] stage4_env_finetune enabled: freezing geometry/appearance, refining envmap.")
        if opt.stage4_train_refl:
            set_refl_lr(gaussians, 0.0)

    # Toycar
    #ENV_CENTER = torch.tensor([0.6810, 0.8080, 4.4550], device='cuda') # None
    #ENV_RANGE = 2.707

    # Garden
    #ENV_CENTER = torch.tensor([-0.2270,  1.9700,  1.7740], device='cuda') # None
    #ENV_RANGE = 0.974

    # Sedan
    #ENV_CENTER = torch.tensor([-0.032,0.808,0.751], device='cuda') # None
    #ENV_RANGE = 2.138

    while iteration < TOT_ITER:        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (not opt.stage4_env_finetune) and iteration > FR_OPTIM_FROM_ITER and iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if (not opt.stage4_env_finetune) and iteration > INIT_UNITIL_ITER:
            initial_stage = False
        if opt.stage4_env_finetune and opt.stage4_train_refl:
            if iteration >= opt.stage4_refl_start_iter:
                set_refl_lr(gaussians, opt.refl_lr)
            else:
                set_refl_lr(gaussians, 0.0)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        need_geometry_maps = (not initial_stage) and (
            opt.lambda_global_depth_smooth > 0 or opt.lambda_global_normal_smooth > 0
        )
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            initial_stage=initial_stage,
            use_mirror_gate=(opt.use_mirror_gate or opt.stage4_env_finetune),
            render_geometry=need_geometry_maps,
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # GT
        gt_image = viewpoint_cam.original_image.cuda()
        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else \
                torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

        gt_mask = None
        if hasattr(viewpoint_cam, "spec_mask") and viewpoint_cam.spec_mask is not None:
            gt_mask = (viewpoint_cam.spec_mask > 0.5).float()
            if opt.stage4_env_finetune and opt.stage4_mask_erode > 1:
                gt_mask = erode_binary_mask(gt_mask, int(opt.stage4_mask_erode))

        if opt.stage4_env_finetune and opt.lambda_stage4_mirror > 0 and gt_mask is not None:
            conf_m = torch.ones_like(gt_mask)
            if "refl_strength_map" in render_pkg and opt.stage4_refl_conf_thr > 0:
                conf_m = (render_pkg["refl_strength_map"].detach() > opt.stage4_refl_conf_thr).float()
            mirror_w = gt_mask * conf_m
            mirror_l1 = (torch.abs(image - gt_image) * mirror_w).sum() / (image.shape[0] * mirror_w.sum() + 1e-6)
            loss = loss + opt.lambda_stage4_mirror * mirror_l1

        if (opt.use_mirror_gate or opt.stage4_env_finetune) and opt.lambda_mirror_leak > 0 and (not initial_stage) and gt_mask is not None:
            # 2026-02-16: suppress reflection leakage in non-mirror regions.
            if "refl_strength_map" in render_pkg:
                mirror_leak_loss = (render_pkg["refl_strength_map"] * (1.0 - gt_mask)).mean()
                loss = loss + opt.lambda_mirror_leak * mirror_leak_loss

        if opt.stage4_env_finetune and opt.lambda_env_tv > 0:
            loss = loss + opt.lambda_env_tv * env_tv_loss(gaussians)

        if (not initial_stage) and "alpha_map" in render_pkg:
            valid_w = (render_pkg["alpha_map"].detach() > opt.geom_alpha_thr).float()
            if opt.lambda_global_depth_smooth > 0 and "depth_map" in render_pkg:
                loss = loss + opt.lambda_global_depth_smooth * edge_aware_depth_smooth_loss(
                    render_pkg["depth_map"], gt_image.detach(), valid_w, opt.geom_edge_weight
                )
            if opt.lambda_global_normal_smooth > 0 and "normal_map" in render_pkg:
                loss = loss + opt.lambda_global_normal_smooth * edge_aware_normal_smooth_loss(
                    render_pkg["normal_map"], gt_image.detach(), valid_w, opt.geom_edge_weight
                )

        if (not initial_stage) and opt.lambda_global_surface_scale > 0:
            loss = loss + opt.lambda_global_surface_scale * global_surface_scale_loss(gaussians)

        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += REFL_MSK_LOSS_W * refl_msk_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations or iteration == TOT_ITER-1):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if preview_cfg is not None and preview_cfg.get("enable", False):
                    save_rgb_previews(
                        scene,
                        pipe,
                        background,
                        iteration,
                        split=preview_cfg.get("split", "train"),
                        names=preview_cfg.get("names", None),
                        max_views=preview_cfg.get("max_views", 3),
                        out_dir_name=preview_cfg.get("out_dir", "save_preview"),
                    )

            if opt.save_envmap_every > 0 and (iteration % opt.save_envmap_every == 0 or iteration == TOT_ITER - 1):
                save_envmap_preview(scene, iteration)

            # Densification
            if (not opt.stage4_env_finetune) and iteration < DENSIFY_UNTIL_ITER:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= INIT_UNITIL_ITER:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= NORMAL_PROP_UNTIL_ITER:
                    opacity_reset_intval = 3000 # 2:1 (reset 1: reset 0)
                    densification_interval = DENSIFIDATION_INTERVAL_WHEN_PROP
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 
                        opt.prune_opacity_threshold, 
                        scene.cameras_extent, size_threshold, 
                    )

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    gaussians.reset_refl(exclusive_msk=outside_msk) ###
                if  OPAC_LR0_INTERVAL > 0 and (INIT_UNITIL_ITER < iteration <= NORMAL_PROP_UNTIL_ITER) and iteration % OPAC_LR0_INTERVAL == 0: ## 200->50
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if  (INIT_UNITIL_ITER < iteration <= NORMAL_PROP_UNTIL_ITER) and iteration % 1000 == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        gaussians.dist_color(exclusive_msk=outside_msk) #
                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if OPAC_LR0_INTERVAL > 0 and iteration != NORMAL_PROP_UNTIL_ITER:
                            gaussians.set_opacity_lr(0.0)

            # Optimizer step
            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        iteration += 1

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        #args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration % 10_000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        env_res = render_env_map(scene.gaussians)
        for env_name in env_res.keys():
            if tb_writer:
                tb_writer.add_image("#envmap/{}".format(env_name), env_res[env_name], global_step=iteration)
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    res = renderFunc(viewpoint, scene.gaussians, more_debug_infos = True, *renderArgs)
                    image = torch.clamp(res["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        for maps_name in res.keys():
                            if 'map' in maps_name:
                                if 'normal' in maps_name:
                                     res[maps_name] = res[maps_name]*0.5+0.5
                                tb_writer.add_image(config['name'] + "_view_{}/{}".format(viewpoint.image_name, maps_name), res[maps_name], global_step=iteration)    
                        tb_writer.add_image(config['name'] + "_view_{}/2_render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == 10_000:
                            tb_writer.add_image(config['name'] + "_view_{}/1_ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #tb_writer.add_scalar("refl_gauss_ratio", scene.gaussians.get_refl_strength_to_total.item(), iteration)
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000, 60_000, 100_000, 150_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--save_preview_on_save", action="store_true", default=False)
    parser.add_argument("--preview_split", type=str, choices=["train", "test", "all"], default="train")
    parser.add_argument("--preview_max_views", type=int, default=3)
    parser.add_argument("--preview_camera_names", nargs="+", type=str, default=[])
    parser.add_argument("--preview_out_dir", type=str, default="save_preview")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    preview_cfg = {
        "enable": bool(args.save_preview_on_save),
        "split": args.preview_split,
        "max_views": int(args.preview_max_views),
        "names": args.preview_camera_names if len(args.preview_camera_names) > 0 else None,
        "out_dir": args.preview_out_dir,
    }
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        preview_cfg=preview_cfg,
    )

    # All done
    print("\nTraining complete.")
