# Global Geometry Constraint Modifications

This note records the code changes that add scene-wide geometry regularization to the current DecoupledGS training pipeline.

## Goal

The geometry constraints are applied to the whole rendered scene instead of only the mirror/specular mask. The design follows a conservative first step:

- Use weak global smoothness on rendered depth and normal maps.
- Preserve object boundaries with RGB edge-aware weights.
- Regularize Gaussian scales so Gaussians stay more surface-like.
- Avoid forcing global planar constraints, because the whole scene can contain curved surfaces, depth discontinuities, thin structures, and occlusion boundaries.

## Modified Files

- `gaussian_renderer/__init__.py`
- `main/train.py`
- `arguments/__init__.py`

## Renderer Changes

`gaussian_renderer.render()` now additionally outputs:

```python
"depth_map": depth_map,
"alpha_map": alpha_map[:1],
```

The current depth is alpha-weighted camera-space depth:

```python
depth_map = depth_num[:1] / alpha_map[:1].clamp_min(1e-6)
```

This is a lightweight version that can be used immediately for regularization. A future PGSR-style upgrade can replace this with an unbiased ray-plane depth from each Gaussian's local plane.

## Training Loss Changes

Three helper losses were added to `main/train.py`.

### Edge-Aware Depth Smoothness

```python
L_depth = |Dx depth| * exp(-w * |Dx image|)
        + |Dy depth| * exp(-w * |Dy image|)
```

The valid region is selected from rendered alpha:

```python
valid_w = (render_pkg["alpha_map"].detach() > opt.geom_alpha_thr).float()
```

This encourages depth continuity inside visible surfaces while reducing smoothing across RGB edges.

### Edge-Aware Normal Smoothness

```python
L_normal = |Dx normal| * exp(-w * |Dx image|)
         + |Dy normal| * exp(-w * |Dy image|)
```

This stabilizes the rendered normal field while respecting image boundaries.

### Global Surface-Like Scale Regularization

```python
L_scale = mean(s_min / s_mid)
```

where `s_min <= s_mid <= s_max` are each Gaussian's activated scales. Minimizing this encourages one Gaussian axis to remain thinner than the tangent axes, making Gaussians more surface-like and reducing thick blob geometry.

## Loss Integration

The total training loss now includes:

```python
loss = rgb_ssim_loss
     + lambda_global_depth_smooth * L_depth
     + lambda_global_normal_smooth * L_normal
     + lambda_global_surface_scale * L_scale
```

These losses are enabled only after the initial stage:

```python
if not initial_stage:
    ...
```

This avoids constraining geometry too early while colors and coarse structure are still forming.

## New Arguments

The following defaults were added in `arguments/OptimizationParams`:

```python
self.lambda_global_depth_smooth = 0.005
self.lambda_global_normal_smooth = 0.001
self.lambda_global_surface_scale = 0.0005
self.geom_alpha_thr = 0.1
self.geom_edge_weight = 10.0
```

Recommended search ranges:

```text
lambda_global_depth_smooth: 0.002 - 0.01
lambda_global_normal_smooth: 0.0005 - 0.003
lambda_global_surface_scale: 0.0002 - 0.001
geom_alpha_thr: 0.05 - 0.2
geom_edge_weight: 5.0 - 20.0
```

## Relation To The Papers

- PGSR: The added depth regularization is a lightweight first step toward depth consistency. The current code does not yet implement PGSR's unbiased Gaussian-plane depth.
- GeoSplat: The scale regularization follows the idea that Gaussians should behave like local surface elements instead of volumetric blobs.
- G3Splat: The global scale prior helps reduce geometry-degenerate Gaussians that can fit images but have unstable 3D structure.

## Suggested Next Upgrade

The next stronger version should add a rendered normal-depth consistency loss:

```text
depth gradient normal should agree with rendered normal
```

For local planar regions or masks, the existing strict plane code in `main/train_mask_only_strict_plane_5.3.py` can be extracted into a shared geometry loss utility and reused as an optional stronger region-level constraint.
