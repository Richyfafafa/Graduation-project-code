# Mirror B-Spline Stage（Stage4）说明

## 目标
- 在镜面/玻璃区域，将高斯中心点约束到一个平滑 B 样条曲面上。
- 减少训练后玻璃表面前后抖动和凹凸不平。
- 不修改原有训练脚本，仅新增阶段脚本与工具文件。

## 新增文件
- `main/train_stage4_mirror_bspline.py`
- `utils/bspline_surface.py`
- `utils/bspline_fit.py`
- `utils/bspline_projection.py`
- `utils/mirror_bspline_adapter.py`

## 训练逻辑
1. 从 `--start_checkpoint` 恢复模型（并尝试恢复同迭代 `.map` 的 env map）。
2. 根据 `mirror_weight` 和 `refl` 阈值选取镜面高斯。
3. 用选中的点做 B 样条曲面拟合（加平滑正则）。
4. 将镜面点投影到 B 样条曲面，得到每个高斯目标点。
5. 只优化 `xyz`，最小化：
   - `lambda_bspline * 点到曲面目标距离`
   - `lambda_anchor * 与初始镜面点偏移`
6. 周期性重拟合曲面（`--refit_every`），提升稳定性。

## 推荐命令
```bash
python -u main/train_stage4_mirror_bspline.py \
  -s <dataset_path> \
  -m <model_path> \
  --start_checkpoint <stage3_or_strict_checkpoint> \
  --bspline_iters 3000 \
  --bspline_xyz_lr 3e-5 \
  --mirror_weight_thr 0.5 \
  --refl_thr 0.02 \
  --num_ctrl_u 6 --num_ctrl_v 6 \
  --degree_u 3 --degree_v 3 \
  --fit_lambda_smooth 5e-4 \
  --lambda_bspline 1.0 \
  --lambda_anchor 0.02 \
  --refit_every 200
```

## 关键参数建议
- `--mirror_weight_thr`：
  - 高一点（0.5~0.7）更保守，只约束高置信镜面点。
  - 低一点（0.3~0.5）覆盖更多区域，但可能引入非镜面点。
- `--num_ctrl_u / --num_ctrl_v`：
  - 控制网格越大，曲面表达越灵活，但可能过拟合。
  - 常用范围：`5~8`。
- `--fit_lambda_smooth`：
  - 越大越平滑，越不容易出现局部波纹。
- `--lambda_anchor`：
  - 防止几何整体漂移，建议 `0.01~0.05`。

## 输出
- 常规：
  - `point_cloud/iteration_xxx/point_cloud.ply`
  - `mirror_bspline_chkpntxxx.pth`
- B 样条中间结果：
  - `mirror_bspline/bspline_init.pt`
  - `mirror_bspline/bspline_iter_xxx.pt`
  - `mirror_bspline/bspline_final.pt`
