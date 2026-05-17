import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arguments import get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from viewer.controller import OrbitCamera
from viewer.ellipsoid_debug import build_ellipsoid_debug_cache
from viewer.main import (
    _build_cam_from_pose,
    _cam_to_c2w,
    _fit_render_scale,
    _load_model_only_view,
    _render_mode_image,
    _scale_hwk,
    _select_views,
    build_parser as build_viewer_parser,
)
from viewer.modes import MODE_NAME, MODE_ORDER, RenderMode


def _resolve_mode(mode_arg: str) -> RenderMode:
    mode_arg = str(mode_arg).strip().lower()
    for mode in MODE_ORDER:
        if mode_arg == MODE_NAME[mode] or mode_arg == mode.name.lower():
            return mode
    raise ValueError(f"Unknown render mode: {mode_arg}")


def _sync_cuda_if_needed(device_name: str):
    if torch.cuda.is_available() and str(device_name).startswith("cuda"):
        torch.cuda.synchronize()


def _load_runtime(args):
    dataset = args.model_extract.extract(args)
    pipe = args.pipeline_extract.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    selected_camera_set = "auto"

    try:
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        test_views = scene.getTestCameras()
        train_views = scene.getTrainCameras()
        views, selected_camera_set = _select_views(args.camera_set, train_views, test_views)
    except Exception as exc:
        print(f"[Benchmark] Scene load failed, falling back to model-only mode: {exc}")
        gaussians, views, loaded_iter = _load_model_only_view(dataset, args.iteration)
        print(f"[Benchmark] Loaded model-only viewer at iteration {loaded_iter}")
        scene = None
        if args.camera_set != "auto":
            print("[Benchmark] camera_set is ignored in model-only mode (using cameras.json order).")

    if len(views) == 0:
        raise RuntimeError("No cameras found for benchmark.")

    print(f"[Benchmark] camera set: {selected_camera_set}, count: {len(views)}")
    return dataset, pipe, gaussians, views, background, scene


def _build_orbit_controller(gaussians, base_view, img_wh):
    pts = gaussians.get_xyz.detach().cpu().numpy()
    scene_center = pts.mean(axis=0).astype(np.float32)
    radius = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    radius = max(radius, 1.0)

    controller = OrbitCamera(img_wh, center=scene_center, radius=radius)
    controller.set_pose(_cam_to_c2w(base_view), keep_radius=True)
    controller.set_orbit_pivot(scene_center, preserve_view=True)
    return controller


def _camera_benchmark_view(src_view, dataset, args):
    auto_scale = _fit_render_scale(src_view.HWK, args.max_render_width, args.max_render_height)
    render_scale = auto_scale if args.render_scale <= 0 else float(args.render_scale)
    render_hwk, img_wh, render_scale = _scale_hwk(src_view.HWK, render_scale)
    view = _build_cam_from_pose(
        _cam_to_c2w(src_view),
        float(src_view.FoVx),
        float(src_view.FoVy),
        render_hwk,
        dataset.data_device,
    )
    return view, img_wh, render_scale


def _orbit_benchmark_view(controller, base_view, dataset, args):
    auto_scale = _fit_render_scale(base_view.HWK, args.max_render_width, args.max_render_height)
    render_scale = auto_scale if args.render_scale <= 0 else float(args.render_scale)
    render_hwk, img_wh, render_scale = _scale_hwk(base_view.HWK, render_scale)
    view = _build_cam_from_pose(
        controller.pose,
        float(base_view.FoVx),
        float(base_view.FoVy),
        render_hwk,
        dataset.data_device,
    )
    return view, img_wh, render_scale


def _summarize_times(times):
    arr = np.asarray(times, dtype=np.float64)
    mean_s = float(arr.mean())
    mean_ms = mean_s * 1000.0
    fps = float(1.0 / mean_s) if mean_s > 0 else 0.0
    return {
        "frames": int(arr.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
        "fps": fps,
    }


def run_benchmark(args):
    dataset, pipe, gaussians, views, background, _scene = _load_runtime(args)
    mode = _resolve_mode(args.start_mode)
    ellipsoid_cache = build_ellipsoid_debug_cache(gaussians)

    start_idx = int(np.clip(args.start_camera, 0, len(views) - 1))
    base_view = views[start_idx]
    controller = _build_orbit_controller(gaussians, base_view, (base_view.HWK[1], base_view.HWK[0]))

    render_times = []
    e2e_times = []
    total_frames = int(max(1, args.benchmark_frames))
    warmup_frames = int(max(0, args.warmup_frames))
    report_every = int(max(1, args.report_every))
    benchmark_preview_active = bool(args.benchmark_interactive_preview)

    print(
        f"[Benchmark] mode={mode.name.lower()} path={args.benchmark_path} "
        f"warmup={warmup_frames} frames={total_frames}"
    )

    with torch.inference_mode():
        for frame_idx in range(warmup_frames + total_frames):
            measure_frame = frame_idx >= warmup_frames

            if args.benchmark_path == "cameras":
                src_view = views[(start_idx + frame_idx) % len(views)]
                view, img_wh, render_scale = _camera_benchmark_view(src_view, dataset, args)
            else:
                if frame_idx > 0:
                    controller.yaw(float(args.orbit_yaw_deg))
                    if abs(float(args.orbit_pitch_deg)) > 1e-8:
                        controller.pitch(float(args.orbit_pitch_deg))
                view, img_wh, render_scale = _orbit_benchmark_view(controller, base_view, dataset, args)

            _sync_cuda_if_needed(dataset.data_device)
            t0 = time.perf_counter()
            image = _render_mode_image(
                view,
                gaussians,
                pipe,
                background,
                mode,
                ellipsoid_cache,
                args.ellipsoid_sigma,
                args.ellipsoid_debug_limit,
                args.ellipsoid_debug_min_radius,
                args.ellipsoid_debug_min_opacity,
                args.ellipsoid_debug_scale_mult,
                args.ellipsoid_debug_canvas_scale,
                args.ellipsoid_debug_backend,
                benchmark_preview_active,
            )
            _sync_cuda_if_needed(dataset.data_device)
            t1 = time.perf_counter()

            if args.copy_to_cpu:
                frame = torch.clamp(image, 0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
                if args.to_uint8:
                    frame = (frame * 255.0).astype(np.uint8)
                del frame
            t2 = time.perf_counter()

            if not measure_frame:
                continue

            render_dt = max(t1 - t0, 1e-9)
            e2e_dt = max(t2 - t0, render_dt)
            render_times.append(render_dt)
            e2e_times.append(e2e_dt)

            measured_idx = frame_idx - warmup_frames + 1
            if measured_idx % report_every == 0 or measured_idx == total_frames:
                render_stats = _summarize_times(render_times)
                e2e_stats = _summarize_times(e2e_times)
                print(
                    f"[Benchmark] {measured_idx:4d}/{total_frames} "
                    f"res={img_wh[0]}x{img_wh[1]} scale={render_scale:.2f} "
                    f"render={render_stats['mean_ms']:.2f}ms {render_stats['fps']:.2f}fps "
                    f"end2end={e2e_stats['mean_ms']:.2f}ms {e2e_stats['fps']:.2f}fps"
                )

    render_stats = _summarize_times(render_times)
    e2e_stats = _summarize_times(e2e_times)
    summary = {
        "model_path": dataset.model_path,
        "mode": mode.name.lower(),
        "benchmark_path": args.benchmark_path,
        "frames": total_frames,
        "warmup_frames": warmup_frames,
        "copy_to_cpu": bool(args.copy_to_cpu),
        "to_uint8": bool(args.to_uint8),
        "render": render_stats,
        "end_to_end": e2e_stats,
    }

    print("[Benchmark] Summary")
    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Benchmark] wrote summary to {out_path}")


def build_parser():
    parser = build_viewer_parser()
    parser.description = "DecoupledGS headless benchmark"
    parser.add_argument("--benchmark_path", choices=["cameras", "orbit"], default="cameras")
    parser.add_argument("--benchmark_frames", type=int, default=240)
    parser.add_argument("--warmup_frames", type=int, default=20)
    parser.add_argument("--report_every", type=int, default=30)
    parser.add_argument("--orbit_yaw_deg", type=float, default=1.5)
    parser.add_argument("--orbit_pitch_deg", type=float, default=0.0)
    parser.add_argument("--copy_to_cpu", action="store_true")
    parser.add_argument("--to_uint8", action="store_true")
    parser.add_argument("--benchmark_interactive_preview", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = get_combined_args(parser)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print("Launching DecoupledGS headless benchmark for", args.model_path)
    safe_state(args.quiet)
    run_benchmark(args)
