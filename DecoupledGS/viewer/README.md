# DecoupledGS Local Viewer

This viewer is a local realtime renderer for `DecoupledGS`, combining:

- the realtime presentation-mode switching idea from the network viewer (`net_viewer`)
- the keyboard interaction workflow from `visualize.py`

## Run

From `Decoupled GS/DecoupledGS`:

```bash
python viewer/main.py --model_path output/your_scene
```

On Windows, you can also use the one-click launcher in the repo root:

```bat
launch_viewer.cmd
launch_viewer.cmd bookcase_strict_mask_ft sphere
```

Optional:

```bash
python viewer/main.py --model_path output/your_scene --iteration -1 --start_mode gaussian --move_speed 0.03 --rot_speed_deg 1.5
python viewer/main.py --model_path output/your_scene --start_mode ellipsoid --ellipsoid_debug_limit 40000 --ellipsoid_debug_canvas_scale 0.60
python viewer/main.py --model_path output/your_scene --start_mode ellipsoid --ellipsoid_debug_limit 40000 --ellipsoid_debug_canvas_scale 0.60 --ellipsoid_debug_preview_canvas_scale 0.35
python viewer/main.py --model_path output/your_scene --start_mode ellipsoid --ellipsoid_debug_backend hybrid
python viewer/main.py --model_path output/your_scene --start_mode ellipsoid --ellipsoid_debug_backend gpu
```

## Controls

- `m`: cycle render mode
- `1..8`: direct render mode
- `space`: switch to next dataset camera preset
- `w/a/s/d/e/f`: translate camera
- `j/u/k/h/l/i`: rotate camera
- Mouse left drag: orbit
- Mouse middle/right drag: pan
- Mouse wheel: zoom
- `r`: reset camera
- `c`: save screenshot to `model_path/viewer_captures`
- `q` or `esc`: quit

## Render Modes

1. `color(final)` - deferred final image
2. `gaussian(c3)` - pure Gaussian mode (no deferred reflection composition)
3. `ellipsoid(debug)` - anisotropic Gaussian ellipsoid debug view; by default this now uses a hybrid backend: GPU while moving for responsiveness, CPU after motion stops to restore the explicit projected ellipse look
4. `refl_strength`
5. `base_color`
6. `refl_color`
7. `normal`

## Ellipsoid Debug Tuning

- `--ellipsoid_debug_backend`: choose `hybrid`, `gpu`, or `cpu`. `hybrid` is now the default: it uses GPU while moving and CPU when static. `gpu` keeps the faster subset-rasterized debug view, while `cpu` keeps the older OpenCV-drawn ellipse fallback all the time.
- `--ellipsoid_debug_limit`: maximum number of ellipsoids drawn in `ellipsoid(debug)`; set higher to retain more, set `0` or negative to disable the cap. Current default: `40000`
- `--ellipsoid_debug_min_radius`: skip only the tiniest projected ellipsoids; lowering it retains more detail but costs FPS. Current default: `0.04`
- `--ellipsoid_debug_min_opacity`: skip only the weakest ellipsoids; lowering it retains more detail but costs FPS. Current default: `0.0002`
- `--ellipsoid_debug_scale_mult`: extra size multiplier used only by `ellipsoid(debug)`. In GPU mode this is passed to the Gaussian rasterizer scale modifier; in CPU mode it scales the drawn ellipse after projection. Current default: `1.0`
- `--ellipsoid_debug_canvas_scale`: render `ellipsoid(debug)` on a smaller internal canvas and upscale it for speed. Lower values are faster in both backends. Current default: `0.60`
- `--ellipsoid_debug_preview_hold`: keep `ellipsoid(debug)` in lightweight preview mode briefly after camera motion, then restore the full-quality debug view. Current default: `0.25`
- `--ellipsoid_debug_preview_limit_scale`: use only a fraction of the normal ellipsoid count while moving. Current default: `0.28`
- `--ellipsoid_debug_preview_min_radius`: ignore tiny projected ellipsoids while moving. Current default: `0.12`
- `--ellipsoid_debug_preview_min_opacity`: ignore very faint ellipsoids while moving. Current default: `0.0010`
- `--ellipsoid_debug_preview_canvas_scale`: use an even smaller internal canvas while moving. Current default: `0.35`

When the camera is static, the viewer now reuses the last rendered frame instead of recomputing identical `ellipsoid(debug)` frames. This does not change the final image, but it avoids wasting CPU on idle redraws.
