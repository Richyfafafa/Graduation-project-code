import argparse
from collections import deque
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration

from ellipsoid_debug import build_ellipsoid_debug_cache, render_ellipsoid_debug_cpu, render_ellipsoid_debug_gpu
from viewer.controller import OrbitCamera
from viewer.modes import KEY_TO_MODE, MODE_NAME, MODE_ORDER, RenderMode
from viewer.viewer_camera import ViewerCamera, build_camera_from_c2w


def _cam_to_c2w(cam) -> np.ndarray:
    w2c = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    return np.linalg.inv(w2c).astype(np.float32)


def _build_cam_from_pose(c2w: np.ndarray, fovx: float, fovy: float, hwk, data_device: str):
    return build_camera_from_c2w(c2w, fovx, fovy, hwk, data_device=data_device)


def _fit_render_scale(hwk, max_width: int, max_height: int) -> float:
    h, w, _ = hwk
    scale = 1.0
    if max_width > 0:
        scale = min(scale, max_width / float(w))
    if max_height > 0:
        scale = min(scale, max_height / float(h))
    return float(min(1.0, max(scale, 1e-3)))


def _scale_hwk(hwk, scale: float):
    h, w, k = hwk
    scale = float(min(1.0, max(scale, 1e-3)))
    if abs(scale - 1.0) < 1e-6:
        return (int(h), int(w), k.copy()), (int(w), int(h)), 1.0

    new_w = max(64, int(round(w * scale)))
    actual_scale = new_w / float(w)
    new_h = max(64, int(round(h * actual_scale)))
    actual_scale = new_h / float(h)
    new_w = max(64, int(round(w * actual_scale)))

    new_k = k.copy()
    new_k[0, :] *= actual_scale
    new_k[1, :] *= actual_scale
    return (new_h, new_w, new_k), (new_w, new_h), actual_scale


def _load_views_from_cameras_json(model_path: Path, data_device: str):
    cameras_path = model_path / "cameras.json"
    with open(cameras_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    views = []
    for idx, cam in enumerate(payload):
        width = int(cam["width"])
        height = int(cam["height"])
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = width * 0.5
        cy = height * 0.5

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.asarray(cam["rotation"], dtype=np.float32)
        c2w[:3, 3] = np.asarray(cam["position"], dtype=np.float32)
        w2c = np.linalg.inv(c2w)

        k = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        views.append(
            ViewerCamera(
                colmap_id=int(cam.get("id", idx)),
                R=np.transpose(w2c[:3, :3]).astype(np.float32),
                T=w2c[:3, 3].astype(np.float32),
                FoVx=focal2fov(fx, width),
                FoVy=focal2fov(fy, height),
                image_name=str(cam.get("img_name", idx)),
                uid=idx,
                data_device=data_device,
                HWK=(height, width, k),
            )
        )
    return views


def _load_model_only_view(dataset, iteration):
    model_path = Path(dataset.model_path)
    cameras_path = model_path / "cameras.json"
    point_cloud_root = model_path / "point_cloud"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_path}")
    if not point_cloud_root.exists():
        raise FileNotFoundError(f"point_cloud directory not found: {point_cloud_root}")

    load_iteration = searchForMaxIteration(str(point_cloud_root)) if iteration == -1 else iteration
    ply_path = point_cloud_root / f"iteration_{load_iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(ply_path))
    views = _load_views_from_cameras_json(model_path, dataset.data_device)
    return gaussians, views, load_iteration


def _render_mode_image(
    view,
    gaussians: GaussianModel,
    pipe,
    background,
    mode: RenderMode,
    ellipsoid_cache,
    ellipsoid_sigma: float,
    ellipsoid_debug_limit: int,
    ellipsoid_debug_min_radius: float,
    ellipsoid_debug_min_opacity: float,
    ellipsoid_debug_scale_mult: float,
    ellipsoid_debug_canvas_scale: float,
    ellipsoid_debug_backend: str,
    ellipsoid_debug_preview_active: bool,
    use_mirror_gate: bool,
) -> torch.Tensor:
    if mode == RenderMode.GAUSSIAN:
        out = render(view, gaussians, pipe, background, initial_stage=True)
        return out["render"]
    if mode == RenderMode.ELLIPSOID:
        backend = ellipsoid_debug_backend
        if backend == "hybrid":
            backend = "gpu" if ellipsoid_debug_preview_active else "cpu"
        if backend == "cpu":
            return render_ellipsoid_debug_cpu(
                view,
                background,
                ellipsoid_cache,
                max_gaussians=ellipsoid_debug_limit,
                sigma_scale=ellipsoid_sigma,
                min_major_radius=ellipsoid_debug_min_radius,
                min_opacity=ellipsoid_debug_min_opacity,
                scale_mult=ellipsoid_debug_scale_mult,
                canvas_scale=ellipsoid_debug_canvas_scale,
            )
        return render_ellipsoid_debug_gpu(
            view,
            background,
            ellipsoid_cache,
            max_gaussians=ellipsoid_debug_limit,
            sigma_scale=ellipsoid_sigma,
            min_major_radius=ellipsoid_debug_min_radius,
            min_opacity=ellipsoid_debug_min_opacity,
            scale_mult=ellipsoid_debug_scale_mult,
            canvas_scale=ellipsoid_debug_canvas_scale,
        )
    out = render(view, gaussians, pipe, background, use_mirror_gate=use_mirror_gate)
    if mode == RenderMode.COLOR:
        return out["render"]
    if mode == RenderMode.STRENGTH:
        return out["refl_strength_map"].expand(3, -1, -1)
    if mode == RenderMode.BASE_COLOR:
        return out["base_color_map"]
    if mode == RenderMode.REFL_COLOR:
        return out["refl_color_map"]
    if mode == RenderMode.NORMAL:
        return out["normal_map"] * 0.5 + 0.5
    return out["render"]


TOOLBAR_BG = (28, 33, 40)
TOOLBAR_HEADER = (44, 52, 63)
TOOLBAR_BORDER = (76, 89, 106)
TOOLBAR_BTN = (57, 68, 82)
TOOLBAR_BTN_ACTIVE = (80, 132, 192)
TOOLBAR_TEXT = (232, 238, 245)
TOOLBAR_TEXT_SUB = (185, 194, 206)


def _point_in_rect(x: int, y: int, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _draw_button(canvas: np.ndarray, rect, label: str, active: bool = False):
    x0, y0, x1, y1 = rect
    fill = TOOLBAR_BTN_ACTIVE if active else TOOLBAR_BTN
    border = (138, 169, 209) if active else TOOLBAR_BORDER
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = x0 + max(8, (x1 - x0 - tw) // 2)
    ty = y0 + max(20, (y1 - y0 + th) // 2)
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TOOLBAR_TEXT, 1, cv2.LINE_AA)


def _compose_viewer_layout(
    render_frame_bgr: np.ndarray,
    toolbar_width: int,
    render_time_ms: float,
    mode: RenderMode,
    cam_idx: int,
    cam_count: int,
    render_wh,
    render_scale: float,
    preview_active: bool,
    ellipsoid_debug_backend: str,
):
    render_h, render_w = render_frame_bgr.shape[:2]
    toolbar_w = max(220, int(toolbar_width))
    canvas = np.zeros((render_h, render_w + toolbar_w, 3), dtype=np.uint8)
    canvas[:, :render_w] = render_frame_bgr

    panel_x0 = render_w
    panel_x1 = render_w + toolbar_w - 1
    cv2.rectangle(canvas, (panel_x0, 0), (panel_x1, render_h - 1), TOOLBAR_BG, -1)
    cv2.line(canvas, (panel_x0, 0), (panel_x0, render_h - 1), TOOLBAR_BORDER, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (panel_x0 + 1, 0), (panel_x1, 54), TOOLBAR_HEADER, -1)

    buttons = []
    left = panel_x0 + 14
    right = panel_x1 - 14
    y = 32

    cv2.putText(canvas, "DecoupledGS Viewer", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TOOLBAR_TEXT, 1, cv2.LINE_AA)
    y = 76
    cv2.putText(canvas, f"Render time: {render_time_ms:.2f} ms", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TOOLBAR_TEXT_SUB, 1, cv2.LINE_AA)
    y += 22
    cv2.putText(
        canvas,
        f"Render: {render_wh[0]}x{render_wh[1]} ({render_scale:.2f}x)",
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        TOOLBAR_TEXT_SUB,
        1,
        cv2.LINE_AA,
    )
    y += 22
    cv2.putText(
        canvas,
        f"Camera: {cam_idx + 1}/{cam_count}",
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        TOOLBAR_TEXT_SUB,
        1,
        cv2.LINE_AA,
    )
    y += 24
    if preview_active:
        cv2.putText(
            canvas,
            f"Ellipsoid preview ({ellipsoid_debug_backend})",
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (155, 203, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    def add_section(title: str):
        nonlocal y
        y += 10
        cv2.putText(canvas, title, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TOOLBAR_TEXT, 1, cv2.LINE_AA)
        y += 10

    def add_full_button(label: str, action, active: bool = False, height: int = 32):
        nonlocal y
        y += 8
        rect = (left, y, right, y + height)
        _draw_button(canvas, rect, label, active=active)
        buttons.append({"rect": rect, "action": action})
        y += height

    def add_row_buttons(items, height: int = 32, gap: int = 8):
        nonlocal y
        y += 8
        n = len(items)
        btn_w = max(60, (right - left - gap * (n - 1)) // n)
        x = left
        for label, action, active in items:
            rect = (x, y, x + btn_w, y + height)
            _draw_button(canvas, rect, label, active=active)
            buttons.append({"rect": rect, "action": action})
            x += btn_w + gap
        y += height

    add_section("Render Mode")
    for m in MODE_ORDER:
        add_full_button(MODE_NAME[m], ("mode", m), active=(mode == m))

    add_section("Resolution")
    add_row_buttons(
        [
            ("-", ("res", "dec"), False),
            ("+", ("res", "inc"), False),
        ]
    )

    add_section("Camera")
    add_row_buttons(
        [
            ("Prev", ("cam", "prev"), False),
            ("Next", ("cam", "next"), False),
        ]
    )

    add_section("Actions")
    add_full_button("Reset View", ("action", "reset"), active=False)
    add_full_button("Save Screenshot", ("action", "screenshot"), active=False)

    hint_y = render_h - 110
    cv2.putText(canvas, "Shortcuts:", (left, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TOOLBAR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "M / 1..7 switch mode", (left, hint_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TOOLBAR_TEXT_SUB, 1, cv2.LINE_AA)
    cv2.putText(canvas, "WASDEF + JUIHKL move", (left, hint_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TOOLBAR_TEXT_SUB, 1, cv2.LINE_AA)
    cv2.putText(canvas, "R reset  C capture  Q quit", (left, hint_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TOOLBAR_TEXT_SUB, 1, cv2.LINE_AA)

    return canvas, buttons


def _next_mode(cur: RenderMode) -> RenderMode:
    idx = MODE_ORDER.index(cur)
    return MODE_ORDER[(idx + 1) % len(MODE_ORDER)]


def _select_views(camera_set: str, train_views, test_views):
    camera_set = str(camera_set).strip().lower()
    if camera_set == "train":
        return list(train_views), "train"
    if camera_set == "test":
        return list(test_views), "test"
    if camera_set == "all":
        merged = []
        if len(test_views) > 0:
            merged.extend(test_views)
        if len(train_views) > 0:
            merged.extend(train_views)
        return merged, "all"
    chosen = list(test_views) if len(test_views) > 0 else list(train_views)
    label = "test(auto)" if len(test_views) > 0 else "train(auto)"
    return chosen, label


def _estimate_scene_center_from_views(views) -> np.ndarray:
    if len(views) == 0:
        return np.zeros(3, dtype=np.float32)
    centers = []
    for view in views:
        c2w = _cam_to_c2w(view)
        centers.append(c2w[:3, 3].astype(np.float32))
    return np.mean(np.stack(centers, axis=0), axis=0).astype(np.float32)


def run_local_viewer(args):
    dataset = args.model_extract.extract(args)
    pipe = args.pipeline_extract.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    selected_camera_set = "auto"

    scene = None
    try:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        test_views = scene.getTestCameras()
        train_views = scene.getTrainCameras()
        views, selected_camera_set = _select_views(args.camera_set, train_views, test_views)
    except Exception as exc:
        print(f"[Viewer] Scene load failed, falling back to model-only mode: {exc}")
        gaussians, views, loaded_iter = _load_model_only_view(dataset, args.iteration)
        print(f"[Viewer] Loaded model-only viewer at iteration {loaded_iter}")
        if args.camera_set != "auto":
            print("[Viewer] camera_set is ignored in model-only mode (using cameras.json order).")

    if len(views) == 0:
        raise RuntimeError("No cameras found in scene.")
    print(f"[Viewer] camera set: {selected_camera_set}, count: {len(views)}")
    print(f"[Viewer] mirror gate: {'on' if args.use_mirror_gate else 'off'}")

    cam_idx = int(np.clip(args.start_camera, 0, len(views) - 1))
    base_view = views[cam_idx]
    base_hwk = base_view.HWK
    fovx, fovy = float(base_view.FoVx), float(base_view.FoVy)
    auto_scale = _fit_render_scale(base_hwk, args.max_render_width, args.max_render_height)
    render_scale = auto_scale if args.render_scale <= 0 else float(args.render_scale)
    render_hwk, img_wh, render_scale = _scale_hwk(base_hwk, render_scale)
    display_wh = (img_wh[0], img_wh[1])

    pts = gaussians.get_xyz.detach().cpu().numpy()
    radius = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    radius = max(radius, 1.0)
    if scene is not None and hasattr(scene, "cameras_center"):
        orbit_center = np.asarray(scene.cameras_center, dtype=np.float32)
    else:
        orbit_center = _estimate_scene_center_from_views(views)

    controller = OrbitCamera(img_wh, center=orbit_center, radius=radius)
    controller.set_pose(_cam_to_c2w(base_view), keep_radius=True)
    controller.set_orbit_pivot(orbit_center, preserve_view=True)
    ellipsoid_cache = build_ellipsoid_debug_cache(gaussians)

    default_center = controller.center.copy()
    default_radius = float(controller.radius)
    default_rot = controller.rot.copy()

    mode = RenderMode.COLOR
    mode_arg = args.start_mode.strip().lower()
    for m in MODE_ORDER:
        if mode_arg == MODE_NAME[m] or mode_arg == m.name.lower():
            mode = m
            break

    save_dir = Path(dataset.model_path) / "viewer_captures"
    save_dir.mkdir(parents=True, exist_ok=True)

    window = "DecoupledGS Local Viewer"
    toolbar_width = int(args.toolbar_width)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_wh[0] + toolbar_width, display_wh[1])

    mouse_state = {"x": 0, "y": 0, "left": False, "mid": False, "right": False}
    render_state = {"dirty": True, "preview_until": 0.0}
    ui_state = {"panel_x": display_wh[0], "buttons": []}

    def request_render(interactive: bool = False):
        render_state["dirty"] = True
        if interactive:
            render_state["preview_until"] = time.time() + float(args.ellipsoid_debug_preview_hold)

    def apply_render_scale(scale_mult: float):
        nonlocal render_hwk, img_wh, render_scale
        target_scale = render_scale * float(scale_mult)
        target_scale = min(1.0, target_scale)
        render_hwk, img_wh, render_scale = _scale_hwk(base_hwk, target_scale)
        controller.W, controller.H = display_wh
        ui_state["panel_x"] = display_wh[0]
        print(f"[Viewer] render scale -> {render_scale:.2f} ({img_wh[0]}x{img_wh[1]})")
        request_render()

    def switch_camera(step: int):
        nonlocal cam_idx
        cam_idx = (cam_idx + int(step)) % len(views)
        controller.set_pose(_cam_to_c2w(views[cam_idx]), keep_radius=True)
        controller.set_orbit_pivot(orbit_center, preserve_view=True)
        request_render()

    def reset_view():
        controller.center = default_center.copy()
        controller.radius = float(default_radius)
        controller.rot = default_rot.copy()
        request_render()

    def save_capture():
        if last_frame_bgr is None:
            return
        save_path = save_dir / f"viewer_{frame_id:06d}_{mode.name.lower()}.png"
        cv2.imwrite(str(save_path), last_frame_bgr)
        print(f"[Viewer] saved screenshot: {save_path}")

    def apply_ui_action(action):
        nonlocal mode
        kind, value = action
        if kind == "mode":
            mode = value
            request_render()
        elif kind == "res":
            apply_render_scale(0.8 if value == "dec" else 1.25)
        elif kind == "cam":
            switch_camera(-1 if value == "prev" else 1)
        elif kind == "action":
            if value == "reset":
                reset_view()
            elif value == "screenshot":
                save_capture()

    def on_mouse(event, x, y, flags, _):
        dx = x - mouse_state["x"]
        dy = y - mouse_state["y"]
        mouse_state["x"] = x
        mouse_state["y"] = y

        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= ui_state["panel_x"]:
                for item in ui_state["buttons"]:
                    if _point_in_rect(x, y, item["rect"]):
                        apply_ui_action(item["action"])
                        break
                return
            controller.set_orbit_pivot(orbit_center, preserve_view=True)
            mouse_state["left"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state["left"] = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            if x >= ui_state["panel_x"]:
                return
            mouse_state["mid"] = True
        elif event == cv2.EVENT_MBUTTONUP:
            mouse_state["mid"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if x >= ui_state["panel_x"]:
                return
            mouse_state["right"] = True
        elif event == cv2.EVENT_RBUTTONUP:
            mouse_state["right"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if x >= ui_state["panel_x"]:
                return
            controller.zoom(1.5 if flags > 0 else -1.5)
            request_render(interactive=True)
        elif event == cv2.EVENT_MOUSEMOVE:
            if x >= ui_state["panel_x"]:
                return
            if mouse_state["left"]:
                controller.orbit(dx, dy, sensitivity_deg=float(args.orbit_sensitivity_deg))
                request_render(interactive=True)
            pan_scale = None
            if mouse_state["mid"]:
                pan_scale = float(args.pan_sensitivity_scale)
            elif mouse_state["right"]:
                pan_scale = float(args.right_pan_sensitivity_scale)
            if pan_scale is not None:
                controller.pan_pixels(dx, dy, scale=pan_scale)
                request_render(interactive=True)

    cv2.setMouseCallback(window, on_mouse)

    move_step = float(args.move_speed)
    rot_step = float(args.rot_speed_deg)
    frame_id = 0
    last_time = time.time()
    render_time_ms = 0.0
    last_frame_bgr = None
    last_preview_active = False

    with torch.inference_mode():
        while True:
            preview_active = mode == RenderMode.ELLIPSOID and time.time() < render_state["preview_until"]
            if last_preview_active and not preview_active:
                render_state["dirty"] = True
            last_preview_active = preview_active

            if render_state["dirty"] or last_frame_bgr is None:
                t0 = time.time()
                view = _build_cam_from_pose(controller.pose, fovx, fovy, render_hwk, dataset.data_device)

                ellipsoid_debug_limit = args.ellipsoid_debug_limit
                ellipsoid_debug_min_radius = args.ellipsoid_debug_min_radius
                ellipsoid_debug_min_opacity = args.ellipsoid_debug_min_opacity
                ellipsoid_debug_canvas_scale = args.ellipsoid_debug_canvas_scale
                if preview_active:
                    if ellipsoid_debug_limit > 0:
                        preview_limit = max(
                            2000,
                            int(round(ellipsoid_debug_limit * float(args.ellipsoid_debug_preview_limit_scale))),
                        )
                        ellipsoid_debug_limit = min(ellipsoid_debug_limit, preview_limit)
                    ellipsoid_debug_min_radius = max(
                        ellipsoid_debug_min_radius,
                        float(args.ellipsoid_debug_preview_min_radius),
                    )
                    ellipsoid_debug_min_opacity = max(
                        ellipsoid_debug_min_opacity,
                        float(args.ellipsoid_debug_preview_min_opacity),
                    )
                    ellipsoid_debug_canvas_scale = min(
                        ellipsoid_debug_canvas_scale,
                        float(args.ellipsoid_debug_preview_canvas_scale),
                    )

                image = _render_mode_image(
                    view,
                    gaussians,
                    pipe,
                    background,
                    mode,
                    ellipsoid_cache,
                    args.ellipsoid_sigma,
                    ellipsoid_debug_limit,
                    ellipsoid_debug_min_radius,
                    ellipsoid_debug_min_opacity,
                    args.ellipsoid_debug_scale_mult,
                    ellipsoid_debug_canvas_scale,
                    args.ellipsoid_debug_backend,
                    preview_active,
                    args.use_mirror_gate,
                )
                frame = torch.clamp(image, 0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
                frame = (frame * 255.0).astype(np.uint8)
                last_frame_bgr = frame[..., ::-1].copy()

                render_dt = max(time.time() - t0, 1e-6)
                render_time_ms = float(render_dt * 1000.0)
                render_state["dirty"] = False

            if last_frame_bgr.shape[1] != display_wh[0] or last_frame_bgr.shape[0] != display_wh[1]:
                display_frame_bgr = cv2.resize(last_frame_bgr, display_wh, interpolation=cv2.INTER_LINEAR)
            else:
                display_frame_bgr = last_frame_bgr

            frame_with_ui, ui_buttons = _compose_viewer_layout(
                display_frame_bgr,
                toolbar_width,
                render_time_ms,
                mode,
                cam_idx,
                len(views),
                img_wh,
                render_scale,
                preview_active=preview_active,
                ellipsoid_debug_backend=args.ellipsoid_debug_backend,
            )
            ui_state["buttons"] = ui_buttons
            ui_state["panel_x"] = display_wh[0]
            cv2.imshow(window, frame_with_ui)

            key = cv2.waitKey(15 if not render_state["dirty"] else 1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mode = _next_mode(mode)
                request_render()
            elif key in KEY_TO_MODE:
                mode = KEY_TO_MODE[key]
                request_render()
            elif key == ord(" "):
                switch_camera(1)
            elif key == ord("r"):
                reset_view()
            elif key == ord("c"):
                save_capture()
            elif key in (ord("-"), ord("_")):
                apply_render_scale(0.8)
            elif key in (ord("="), ord("+")):
                apply_render_scale(1.25)
            elif key == ord("a"):
                controller.translate_local(+move_step, 0.0, 0.0)
                request_render(interactive=True)
            elif key == ord("d"):
                controller.translate_local(-move_step, 0.0, 0.0)
                request_render(interactive=True)
            elif key == ord("w"):
                controller.translate_local(0.0, 0.0, -move_step)
                request_render(interactive=True)
            elif key == ord("s"):
                controller.translate_local(0.0, 0.0, +move_step)
                request_render(interactive=True)
            elif key == ord("e"):
                controller.translate_local(0.0, +move_step, 0.0)
                request_render(interactive=True)
            elif key == ord("f"):
                controller.translate_local(0.0, -move_step, 0.0)
                request_render(interactive=True)
            elif key == ord("j"):
                controller.yaw(+rot_step)
                request_render(interactive=True)
            elif key == ord("u"):
                controller.yaw(-rot_step)
                request_render(interactive=True)
            elif key == ord("k"):
                controller.pitch(+rot_step)
                request_render(interactive=True)
            elif key == ord("h"):
                controller.pitch(-rot_step)
                request_render(interactive=True)
            elif key == ord("l"):
                controller.roll(+rot_step)
                request_render(interactive=True)
            elif key == ord("i"):
                controller.roll(-rot_step)
                request_render(interactive=True)

            frame_id += 1
            now = time.time()
            if now - last_time > 1.5:
                last_time = now

    cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="DecoupledGS local realtime viewer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera_set", type=str, choices=["auto", "test", "train", "all"], default="all")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_camera", type=int, default=0)
    parser.add_argument("--start_mode", type=str, default="color")
    parser.add_argument("--use_mirror_gate", dest="use_mirror_gate", action="store_true")
    parser.add_argument("--disable_mirror_gate", dest="use_mirror_gate", action="store_false")
    parser.add_argument("--move_speed", type=float, default=0.03)
    parser.add_argument("--rot_speed_deg", type=float, default=1.5)
    parser.add_argument("--orbit_sensitivity_deg", type=float, default=0.25)
    parser.add_argument("--pan_sensitivity_scale", type=float, default=1.5e-4)
    parser.add_argument("--right_pan_sensitivity_scale", type=float, default=2.4e-4)
    parser.add_argument("--render_scale", type=float, default=1.0)
    parser.add_argument("--ellipsoid_sigma", type=float, default=2.2)
    parser.add_argument("--ellipsoid_debug_limit", type=int, default=40000)
    parser.add_argument("--ellipsoid_debug_min_radius", type=float, default=0.04)
    parser.add_argument("--ellipsoid_debug_min_opacity", type=float, default=0.0002)
    parser.add_argument("--ellipsoid_debug_scale_mult", type=float, default=1.0)
    parser.add_argument("--ellipsoid_debug_canvas_scale", type=float, default=0.60)
    parser.add_argument("--ellipsoid_debug_backend", type=str, choices=["hybrid", "gpu", "cpu"], default="hybrid")
    parser.add_argument("--ellipsoid_debug_preview_hold", type=float, default=0.25)
    parser.add_argument("--ellipsoid_debug_preview_limit_scale", type=float, default=0.28)
    parser.add_argument("--ellipsoid_debug_preview_min_radius", type=float, default=0.12)
    parser.add_argument("--ellipsoid_debug_preview_min_opacity", type=float, default=0.0010)
    parser.add_argument("--ellipsoid_debug_preview_canvas_scale", type=float, default=0.35)
    parser.add_argument("--max_render_width", type=int, default=1280)
    parser.add_argument("--max_render_height", type=int, default=960)
    parser.add_argument("--toolbar_width", type=int, default=320)
    parser.set_defaults(model_extract=model, pipeline_extract=pipeline, use_mirror_gate=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = get_combined_args(parser)
    print("Launching DecoupledGS local viewer for", args.model_path)
    safe_state(args.quiet)
    run_local_viewer(args)
