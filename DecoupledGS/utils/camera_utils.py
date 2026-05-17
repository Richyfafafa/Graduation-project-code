#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
from PIL import Image
import numpy as np
import os, cv2, torch, time
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from utils.spec_mask_utils import find_spec_mask_path, load_spec_mask_with_regions

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    t0 = time.time()
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if False and orig_w > 1600: ###
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        HWK = None
        if cam_info.K is not None:
            K = cam_info.K.copy()
            K[:2] = K[:2] * scale
            HWK = (resolution[1], resolution[0], K)

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # 2026-01-14 / 2026-03-11：
                                   
                
                                      
                                                           
    spec_mask = None
    spec_mask_region_labels = None
    spec_mask_region_ids = None
    mask_path = None
    if getattr(args, "spec_mask_dir", None):
        mask_path = find_spec_mask_path(args.source_path, args.spec_mask_dir, cam_info.image_name, getattr(cam_info, "image_path", None))
        if mask_path is not None:
            spec_mask, spec_mask_region_labels, spec_mask_region_ids = load_spec_mask_with_regions(mask_path, resolution)

                                        
            if getattr(args, "spec_mask_debug", False):
                debug_every = max(1, int(getattr(args, "spec_mask_debug_every", 50)))
                if id % debug_every == 0:
                    debug_dir = os.path.join(args.model_path, getattr(args, "spec_mask_debug_dir", "debug_masks"))
                    os.makedirs(debug_dir, exist_ok=True)
                    rgb = gt_image[:3].clone()
                    overlay = rgb * (1.0 - 0.5 * spec_mask) + torch.tensor([1.0, 0.0, 0.0], device=rgb.device)[:, None, None] * (0.5 * spec_mask)
                    overlay = (overlay.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(overlay).save(os.path.join(debug_dir, f"{cam_info.image_name}.png"))

    refl_path = os.path.join(
        os.path.dirname(os.path.dirname(cam_info.image_path)), 'image_msk')
    refl_path = os.path.join(refl_path, os.path.basename(cam_info.image_path))
    if not os.path.exists(refl_path):
        refl_path = refl_path.replace('.JPG', '.jpg')
    if os.path.exists(refl_path):
        refl_msk = cv2.imread(refl_path) != 0 # max == 1
        refl_msk = torch.tensor(refl_msk).permute(2,0,1).float()
    else: refl_msk = None

    camera = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, spec_mask=spec_mask, spec_mask_region_labels=spec_mask_region_labels, spec_mask_region_ids=spec_mask_region_ids,
                  image_name=cam_info.image_name, uid=id, 
                  data_device=args.data_device, HWK=HWK, gt_refl_mask=refl_msk)
    if getattr(args, "camera_load_debug", False):
        every = max(1, int(getattr(args, "camera_load_debug_every", 1)))
        if id % every == 0:
            num_regions = 0 if spec_mask_region_labels is None else int(spec_mask_region_labels.max().item())
            has_union = spec_mask is not None
            elapsed = time.time() - t0
            print(
                f"[CAM_LOAD] idx={id} name={cam_info.image_name} "
                f"size={camera.image_width}x{camera.image_height} "
                f"mask={'yes' if has_union else 'no'} regions={num_regions} "
                f"mask_path={mask_path if mask_path is not None else 'None'} "
                f"time={elapsed:.3f}s",
                flush=True,
            )
    return camera

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    t0 = time.time()
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    if getattr(args, "camera_load_debug", False):
        print(
            f"[CAM_LOAD] completed {len(camera_list)} cameras at scale={resolution_scale} "
            f"in {time.time() - t0:.3f}s",
            flush=True,
        )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
