# DecoupledGS Viewer 实现文档

本文档面向代码阅读和二次开发，系统说明 `DecoupledGS` 本地 viewer 的功能、实现方式、参数来源、依赖库、核心函数以及整体设计思路。

## 1. Viewer 的定位

`viewer` 是一个本地实时查看器，用来在训练输出目录上直接浏览 `DecoupledGS` 模型结果。它不是训练脚本的一部分，而是一个独立的离线交互工具，核心目标有三点：

1. 直接加载训练结果并可视化。
2. 统一查看不同渲染中间结果和最终结果。
3. 在交互流畅性和调试可读性之间做平衡，尤其是椭球调试模式。

它综合了两类思路：

- 借鉴网络版 viewer 的“渲染模式切换”能力。
- 借鉴本地可视化脚本的“键鼠相机控制”能力。

## 2. 代码结构

`viewer` 目录下与实现相关的文件如下：

- `viewer/main.py`
  - viewer 主入口，负责参数解析、模型加载、窗口/UI、事件循环、模式切换和截图。
- `viewer/controller.py`
  - 相机控制器，负责 orbit / pan / zoom / 本地坐标系平移和旋转。
- `viewer/modes.py`
  - 渲染模式枚举、显示名称、快捷键映射。
- `viewer/viewer_camera.py`
  - 构建 viewer 使用的相机对象，补齐 `world_view_transform`、投影矩阵、相机中心等渲染所需字段。
- `viewer/ellipsoid_debug.py`
  - 椭球调试视图实现，提供 CPU 和 GPU 两种后端。
- `viewer/README.md`
  - 简要使用说明。

viewer 还依赖项目主干中的模块：

- `arguments/__init__.py`
  - 项目公共参数系统。
- `scene/__init__.py`
  - 场景和相机加载。
- `scene/cameras.py`
  - 原始训练/测试相机定义。
- `gaussian_renderer/__init__.py`
  - Gaussian 渲染和 DecoupledGS 的 deferred shading 主逻辑。
- `utils.graphics_utils`
  - `focal2fov`、相机矩阵相关函数。
- `utils.general_utils`
  - `safe_state`、旋转构造等。
- `utils.system_utils`
  - 自动搜索最新 iteration。

## 3. 入口与执行流程

### 3.1 程序入口

`viewer/main.py` 的执行入口是：

1. `build_parser()`
2. `get_combined_args(parser)`
3. `safe_state(args.quiet)`
4. `run_local_viewer(args)`

这里的设计不是只解析 viewer 自己的命令行参数，而是把 viewer 参数和项目通用参数系统合并。

### 3.2 参数合并方式

`get_combined_args(parser)` 的行为是：

1. 先解析命令行参数。
2. 再尝试读取 `model_path/cfg_args`。
3. 用命令行参数覆盖配置文件中的同名项。

因此 viewer 的参数来源有两层：

- 项目训练/推理通用配置，例如 `model_path`、`source_path`、`white_background`、`sh_degree`、`data_device`。
- viewer 自己新增的交互和调试参数。

这使得 viewer 可以直接复用训练时的模型配置，不需要单独维护一套完整配置文件。

### 3.3 主流程

`run_local_viewer(args)` 的核心流程如下：

1. 从 `args` 中提取 `dataset` 和 `pipe`。
2. 根据 `white_background` 构造背景颜色张量。
3. 尝试通过 `Scene` 加载高层场景对象。
4. 如果失败，则回退到“只加载模型 + cameras.json”的模式。
5. 选定初始相机和初始分辨率。
6. 从高斯点云估计场景中心和半径，初始化 `OrbitCamera`。
7. 预构建椭球调试缓存。
8. 创建 OpenCV 窗口和鼠标/键盘回调。
9. 进入循环：按需渲染、绘制侧边栏、显示图像、响应交互。

## 4. Viewer 支持的全部功能

### 4.1 加载训练结果并显示

功能说明：

- 可以直接对 `output/...` 下的模型结果启动 viewer。
- 支持按 iteration 加载指定 checkpoint。
- 支持自动选择最新 iteration。

实现方式：

- 优先使用 `Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)`。
- `Scene` 成功时，可获得：
  - `gaussians`
  - `test_views`
  - `train_views`
- 如果 `Scene` 加载失败，则调用 `_load_model_only_view()`：
  - 在 `model_path/point_cloud/iteration_xxx/point_cloud.ply` 中读取高斯模型。
  - 在 `model_path/cameras.json` 中读取相机参数。

这样设计的原因是 viewer 既能在“数据集元信息完整”的情况下工作，也能在只保留训练产物时工作。

### 4.2 相机集合选择

功能说明：

- 支持 `auto`、`train`、`test`、`all` 四种相机集合选择。

实现方式：

- `_select_views(camera_set, train_views, test_views)` 负责选择。
- 规则：
  - `train` 只使用训练相机。
  - `test` 只使用测试相机。
  - `all` 先 test 再 train 合并。
  - `auto` 优先 test，没有 test 时退回 train。

这样做的动机是：

- 测试相机通常更符合“查看泛化结果”的需求。
- 又不能要求所有场景都必须有测试集。

### 4.3 从 `cameras.json` 回退加载

功能说明：

- 即使 `source_path` 下缺少原始 Colmap/Blender 数据，viewer 仍可运行。

实现方式：

- `_load_views_from_cameras_json()` 从 `cameras.json` 逐项读取：
  - `width`
  - `height`
  - `fx`
  - `fy`
  - `rotation`
  - `position`
  - `id`
  - `img_name`
- 代码将 `rotation + position` 视为 C2W。
- 然后求逆得到 W2C，再构造 `ViewerCamera`。

这个回退路径很关键，因为很多时候用户只保留训练输出目录，而不会保留原始数据集结构。

### 4.4 多种渲染模式切换

viewer 提供 7 种渲染模式：

1. `color(final)`
2. `gaussian(c3)`
3. `ellipsoid(debug)`
4. `refl_strength`
5. `base_color`
6. `refl_color`
7. `normal`

模式定义在 `viewer/modes.py` 中，通过 `RenderMode`、`MODE_ORDER`、`MODE_NAME`、`KEY_TO_MODE` 管理。

#### 4.4.1 `color(final)`

功能说明：

- 显示 DecoupledGS 的最终合成图像。

实现方式：

- 在 `_render_mode_image()` 中调用：
  - `render(view, gaussians, pipe, background, use_mirror_gate=True)`
- 从返回结果中取 `out["render"]`。

底层逻辑位于 `gaussian_renderer.render()`：

1. 先渲染 `base_color`。
2. 同时输出每像素法线和反射强度。
3. 使用环境图与法线计算 `refl_color`。
4. 用 `(1 - refl_strength) * base_color + refl_strength * refl_color` 得到最终结果。

其中 `use_mirror_gate=True` 表示反射强度会被 `pc.get_mirror_weight` 进一步门控，只在镜面区域生效。

#### 4.4.2 `gaussian(c3)`

功能说明：

- 显示纯 Gaussian splatting 结果，不走 deferred reflection 合成。

实现方式：

- 调用：
  - `render(view, gaussians, pipe, background, initial_stage=True)`
- 返回 `out["render"]`。

底层使用 `diff_gaussian_rasterization_c3`，只输出 3 通道颜色图。

这个模式的意义是：

- 用来对比“纯高斯底色”和“最终含反射的结果”。
- 适合检查基础颜色拟合质量。

#### 4.4.3 `ellipsoid(debug)`

功能说明：

- 以椭球或高斯体的方式调试可视化每个 Gaussian 的空间形状和投影范围。

实现方式：

- 预处理阶段调用 `build_ellipsoid_debug_cache(gaussians)`，缓存：
  - `xyz_h`
  - `xyz`
  - `scales`
  - `quat`
  - `rots`
  - `cov3d`
  - `opacity`
  - `dc_color`
- 运行时在 `_render_mode_image()` 中根据后端选择：
  - `render_ellipsoid_debug_cpu()`
  - `render_ellipsoid_debug_gpu()`

viewer 支持三种后端策略：

- `cpu`
  - 用 OpenCV 在 2D 画布上逐个画椭圆。
- `gpu`
  - 把筛选后的 Gaussian 子集重新喂给 rasterizer，在 GPU 上快速绘制。
- `hybrid`
  - 交互时使用 GPU 预览，停止移动后切回 CPU 精细显示。

这是 viewer 设计里最典型的“交互性和可读性折中”。

#### 4.4.4 `refl_strength`

功能说明：

- 显示每像素反射强度。

实现方式：

- 从 `render(..., use_mirror_gate=True)` 的输出中取 `out["refl_strength_map"]`。
- 再用 `.expand(3, -1, -1)` 扩成 3 通道显示。

#### 4.4.5 `base_color`

功能说明：

- 显示不含反射合成的底色图。

实现方式：

- 返回 `out["base_color_map"]`。

#### 4.4.6 `refl_color`

功能说明：

- 显示环境图反射颜色图。

实现方式：

- 返回 `out["refl_color_map"]`。

#### 4.4.7 `normal`

功能说明：

- 显示法线图。

实现方式：

- 底层输出是 `[-1, 1]` 范围法线。
- viewer 中做 `out["normal_map"] * 0.5 + 0.5` 映射到 `[0, 1]` 便于显示。

### 4.5 鼠标交互控制相机

支持的操作：

- 左键拖动：orbit
- 中键或右键拖动：pan
- 滚轮：zoom

实现方式：

- OpenCV 通过 `cv2.setMouseCallback(window, on_mouse)` 注册回调。
- 在 `on_mouse()` 中维护 `mouse_state`：
  - `x`
  - `y`
  - `left`
  - `mid`
  - `right`
- 根据事件调用 `OrbitCamera`：
  - `orbit(dx, dy)`
  - `pan_pixels(dx, dy)`
  - `zoom(delta)`

### 4.6 键盘交互控制相机和 viewer

支持的快捷键：

- `m`：循环切换渲染模式
- `1..7`：直接切换模式
- `space`：切换到下一个数据集相机
- `r`：重置视图
- `c`：截图
- `q` / `esc`：退出
- `-` / `+`：降低/提高内部渲染分辨率
- `w/a/s/d/e/f`：局部坐标系平移
- `j/u/k/h/l/i`：局部坐标系旋转

实现方式：

- 主循环中用 `cv2.waitKey(...) & 0xFF` 读键盘。
- 根据按键调用：
  - `_next_mode()`
  - `switch_camera()`
  - `reset_view()`
  - `save_capture()`
  - `apply_render_scale()`
  - `OrbitCamera.translate_local()`
  - `OrbitCamera.yaw() / pitch() / roll()`

### 4.7 数据集相机切换

功能说明：

- 可以在场景中快速切换到其它预设相机位姿。

实现方式：

- `switch_camera(step)` 更新 `cam_idx`。
- 再用 `controller.set_pose(_cam_to_c2w(views[cam_idx]), keep_radius=True)` 同步控制器姿态。

这里的思路不是简单替换 viewer 相机对象，而是把“数据集相机姿态”映射到控制器状态中，这样切换后依旧能继续交互。

### 4.8 分辨率动态调整

功能说明：

- viewer 可以动态调整内部渲染分辨率，以换取更高帧率。

实现方式：

- `_fit_render_scale()` 根据：
  - `max_render_width`
  - `max_render_height`
  自动给出初始缩放。
- `_scale_hwk()` 按缩放比例生成新的 `(H, W, K)`。
- `apply_render_scale()` 在运行时调整 `render_hwk`、`img_wh`、`render_scale`。

注意：

- 窗口显示尺寸 `display_wh` 默认保持为初始尺寸。
- 当内部渲染尺寸变小后，会使用 `cv2.resize` 放大到显示尺寸。

这是一种典型的“内部低分辨率渲染 + 外部固定窗口显示”的实时 viewer 方案。

### 4.9 侧边栏 UI

功能说明：

- viewer 不是纯图像窗口，还会绘制一个右侧操作面板。

UI 内容包括：

- 标题
- 实时 FPS
- 当前内部渲染分辨率和缩放比例
- 当前相机编号
- 椭球预览状态提示
- 渲染模式按钮
- 分辨率按钮
- 相机切换按钮
- Reset / Screenshot 按钮
- 快捷键提示

实现方式：

- `_compose_viewer_layout()` 负责构建最终画布。
- 用 OpenCV 的 2D 绘制接口：
  - `cv2.rectangle`
  - `cv2.line`
  - `cv2.putText`
- `_draw_button()` 负责按钮绘制。
- 每个按钮会记录一个：
  - `rect`
  - `action`
- 鼠标点击右侧面板时，用 `_point_in_rect()` 判断命中，再调用 `apply_ui_action()`。

设计上，UI 是“渲染图像 + OpenCV 手绘控制面板”的轻量方案，不引入额外 GUI 框架，部署简单。

### 4.10 截图保存

功能说明：

- 可以保存当前模式下的 viewer 图像。

实现方式：

- 保存目录固定为：
  - `model_path/viewer_captures`
- `save_capture()` 把 `last_frame_bgr` 写到：
  - `viewer_{frame_id:06d}_{mode.name.lower()}.png`

注意这里保存的是纯渲染画面，不包含右侧面板。

### 4.11 按需渲染和静止复用

功能说明：

- 相机不动时不重复渲染，避免空转浪费。

实现方式：

- `render_state` 中维护：
  - `dirty`
  - `preview_until`
- 只有在以下场景才把 `dirty=True`：
  - 键鼠交互
  - 模式切换
  - 分辨率变化
  - 相机切换
  - 椭球预览状态切换
- 若 `dirty=False`，直接复用 `last_frame_bgr`。

这是整个 viewer 能保持较轻资源占用的关键机制。

### 4.12 椭球模式的预览机制

功能说明：

- 在 `ellipsoid(debug)` 模式中，用户拖动相机时使用轻量预览，停止后恢复高质量显示。

实现方式：

- 只要发生交互，就在 `request_render(interactive=True)` 中把：
  - `preview_until = 当前时间 + ellipsoid_debug_preview_hold`
- 主循环中判断：
  - `preview_active = mode == ELLIPSOID and now < preview_until`
- 若处于 preview：
  - 降低最多绘制的高斯数量
  - 提高最小半径阈值
  - 提高最小透明度阈值
  - 降低内部画布尺寸

这几项共同降低 CPU/GPU 负担，使交互更顺畅。

## 5. 核心实现细节

### 5.1 `OrbitCamera` 的设计

`OrbitCamera` 在 `viewer/controller.py` 中实现，其状态只有三个核心量：

- `center`
  - 观察中心
- `radius`
  - 相机离观察中心的距离
- `rot`
  - 当前相机方向旋转矩阵

`pose` 属性返回 C2W 矩阵，生成方式是：

1. 在局部坐标里把相机沿 z 方向后移 `radius`。
2. 应用旋转。
3. 结合 `center` 得到最终位姿。

这样设计比直接维护完整 4x4 位姿矩阵更适合交互：

- orbit 只改旋转
- zoom 只改半径
- pan / translate 只改中心

交互逻辑更清晰，不容易出现状态耦合。

### 5.2 viewer 使用专门的 `ViewerCamera`

viewer 没有直接复用训练时带图像和 mask 的 `Camera`，而是单独实现了 `ViewerCamera`。

原因是：

- viewer 不需要原始图像。
- viewer 只关心渲染所需几何参数。
- viewer 要支持运行时从自由相机姿态快速构造相机对象。

`ViewerCamera` 持有：

- `world_view_transform`
- `projection_matrix`
- `full_proj_transform`
- `camera_center`
- `R`
- `T`
- `HWK`

`build_camera_from_c2w()` 可以把当前控制器姿态即时变成 renderer 可接受的相机对象。

### 5.3 为什么要自己构造投影矩阵

`build_projection_matrix_correct()` 使用的是基于内参矩阵 `K` 的投影构造方式，而不是只依赖 `FoVx/FoVy`。

优势是：

- 可以保留主点偏移 `cx/cy`。
- 更适合从 `cameras.json` 或任意内参构造精确相机。

这对 viewer 很重要，因为它需要尽量忠实复现训练相机的投影关系。

### 5.4 `gaussian_renderer.render()` 如何支持 viewer 各模式

这个函数实际上提供了 viewer 所需的大多数中间结果：

- `render`
- `refl_strength_map`
- `normal_map`
- `refl_color_map`
- `base_color_map`
- `viewspace_points`
- `visibility_filter`
- `radii`

模式复用关系如下：

- `gaussian(c3)` 使用 `initial_stage=True`，走纯颜色栅格化。
- `color(final)`、`refl_strength`、`base_color`、`refl_color`、`normal` 使用 deferred 渲染输出。

因此 viewer 自身并没有实现完整的渲染算法，它只是对项目已有 renderer 做了“模式化封装”。

### 5.5 椭球调试缓存为什么要提前构建

`build_ellipsoid_debug_cache()` 会提前提取并缓存每个 Gaussian 的：

- 中心坐标
- 旋转
- 缩放
- 3D 协方差
- 透明度
- 调试颜色

原因是这些量在 viewer 中通常不会变化，而椭球调试渲染会反复用到。如果每帧都重新从模型推导，会带来不必要开销。

### 5.6 CPU 椭球调试模式

CPU 模式的关键步骤是：

1. `_project_ellipsoids()`
   - 把 3D Gaussian 协方差投影到 2D。
2. 对 2D 协方差求特征值/特征向量。
3. 得到椭圆长轴、短轴和旋转角。
4. 根据深度排序，从远到近绘制。
5. `_blend_debug_ellipse()` 用 OpenCV 把每个椭圆 alpha 混合到画布。

它的特点是：

- 椭圆轮廓语义明确，适合分析。
- 但逐个绘制在高数量场景下较慢。

### 5.7 GPU 椭球调试模式

GPU 模式并不是在 GPU 上直接画 2D OpenCV 风格椭圆，而是：

1. 先用 `_project_ellipsoids()` 在屏幕空间筛选重要 Gaussian。
2. 再把筛选出的 `means3D/scales/rotations/opacities/colors` 送给 `diff_gaussian_rasterization_c3`。
3. 用较小的内部画布渲染。
4. 再双线性上采样回输出尺寸。

优点：

- 快。
- 能利用原本的高斯栅格化器。

缺点：

- 视觉语义更偏“高斯斑块”而不是“明确椭圆边界”。

所以 `hybrid` 模式的策略是：

- 相机在动时优先快。
- 相机静止后优先可读。

这就是该 viewer 最核心的交互设计思想之一。

## 6. 参数说明

### 6.1 来自项目公共参数系统的关键参数

这些参数不是 viewer 独有，但 viewer 会直接读取：

- `model_path`
  - 模型输出目录，viewer 的主要输入。
- `source_path`
  - 原始数据路径，`Scene` 加载时可能使用。
- `images`
  - 图像目录名。
- `white_background`
  - 决定背景是白色还是黑色。
- `sh_degree`
  - 构造 `GaussianModel` 时使用。
- `data_device`
  - 相机和渲染数据所在设备，通常是 `cuda`。
- `debug`
  - pipeline 调试开关。

### 6.2 viewer 自己新增的参数

#### 基础控制

- `--iteration`
  - 要加载的迭代轮数，`-1` 表示自动选择最新。
- `--camera_set`
  - `auto/test/train/all`。
- `--quiet`
  - 传给 `safe_state()`。
- `--start_camera`
  - 启动后默认使用的相机索引。
- `--start_mode`
  - 启动时默认渲染模式。

#### 交互速度

- `--move_speed`
  - 键盘平移步长。
- `--rot_speed_deg`
  - 键盘旋转步长，单位为度。

#### 分辨率

- `--render_scale`
  - 内部渲染缩放，`<=0` 时自动根据最大尺寸计算。
- `--max_render_width`
  - 自动缩放时允许的最大宽度。
- `--max_render_height`
  - 自动缩放时允许的最大高度。
- `--toolbar_width`
  - 右侧工具栏宽度。

#### 椭球调试

- `--ellipsoid_sigma`
  - 椭球投影半径的 sigma 缩放。
- `--ellipsoid_debug_limit`
  - 最多绘制多少个 Gaussian。
- `--ellipsoid_debug_min_radius`
  - 过滤掉太小的投影椭球。
- `--ellipsoid_debug_min_opacity`
  - 过滤掉透明度太低的 Gaussian。
- `--ellipsoid_debug_scale_mult`
  - 额外尺寸缩放。
- `--ellipsoid_debug_canvas_scale`
  - 椭球模式内部画布缩放。
- `--ellipsoid_debug_backend`
  - `hybrid/gpu/cpu`。

#### 椭球预览

- `--ellipsoid_debug_preview_hold`
  - 交互后预览态持续时间。
- `--ellipsoid_debug_preview_limit_scale`
  - 预览态下 Gaussian 数量缩放比例。
- `--ellipsoid_debug_preview_min_radius`
  - 预览态最小半径阈值。
- `--ellipsoid_debug_preview_min_opacity`
  - 预览态最小透明度阈值。
- `--ellipsoid_debug_preview_canvas_scale`
  - 预览态内部画布缩放。

## 7. 依赖的库与作用

### 7.1 Python 标准库

- `argparse`
  - 命令行解析。
- `json`
  - 读取 `cameras.json`。
- `os` / `sys` / `pathlib`
  - 路径和模块搜索路径处理。
- `time`
  - 实时渲染状态与 FPS 统计。
- `collections.deque`
  - 维护 1 秒窗口的时间戳，计算实时 FPS。
- `math`
  - 旋转和投影中的数学运算。

### 7.2 第三方库

- `numpy`
  - 位姿矩阵、相机内参、CPU 端投影和 UI 图像操作。
- `torch`
  - 核心张量计算、模型数据、GPU 渲染接口。
- `torch.nn.functional`
  - GPU 椭球模式中的上采样。
- `cv2`
  - 窗口显示、鼠标键盘交互、UI 绘制、CPU 椭圆绘制、截图保存。
- `diff_gaussian_rasterization_c3`
  - 3 通道 Gaussian 栅格化。
- `diff_gaussian_rasterization_c7`
  - 7 通道输出，用于 base color、normal、reflection strength 等联合渲染。

## 8. 数据流与渲染链路

可以把 viewer 理解为以下链路：

1. 读取参数
2. 加载 `GaussianModel`
3. 获取或构造相机列表
4. 初始化 `OrbitCamera`
5. 把交互状态转换成当前自由视角的 `ViewerCamera`
6. 根据渲染模式调用：
   - `gaussian_renderer.render()`
   - 或 `render_ellipsoid_debug_cpu/gpu()`
7. 把 `torch.Tensor` 转为 `numpy uint8`
8. 和侧边栏拼接
9. `cv2.imshow()`

其中最关键的接口边界是：

- `OrbitCamera.pose`
  - 自由交互层
- `build_camera_from_c2w(...)`
  - viewer 相机适配层
- `render(...)`
  - 真实渲染层

这是一个比较清晰的三层设计。

## 9. 设计思路总结

这个 viewer 的设计思路可以概括为以下几点。

### 9.1 尽量复用项目现有渲染能力

viewer 没有重写一套新的 DecoupledGS 渲染器，而是把 `gaussian_renderer.render()` 的不同输出包装成多种查看模式。这样可以保证：

- 结果和训练/评测逻辑一致。
- 维护成本低。
- 调试中间结果更方便。

### 9.2 用最轻依赖完成本地交互

viewer 没有引入 Qt、DearPyGui 或浏览器前端，而是直接使用 OpenCV：

- 简单
- 跨平台
- 易部署

代价是 UI 能力有限，但对本项目“本地实时调试”这一目标足够。

### 9.3 把交互控制和渲染相机解耦

`OrbitCamera` 只管理交互状态，不直接参与 rasterization。
真正喂给 renderer 的是 `ViewerCamera`。

这样做有两个好处：

- 相机交互逻辑保持独立和清晰。
- 任何时候都可以把当前自由视角即时转换成标准渲染相机。

### 9.4 把“高质量调试”和“实时交互”分开处理

最典型的例子是 `ellipsoid(debug)`：

- 静止时追求可读性，CPU 椭圆更清楚。
- 运动时追求响应速度，GPU 子集渲染更快。

这种按交互状态切换策略，是整个 viewer 最实用也最工程化的一部分。

### 9.5 支持不完整场景元数据

viewer 既支持完整数据集加载，也支持只有：

- `point_cloud.ply`
- `cameras.json`

的最小运行条件。这让它更像一个“训练结果检查器”，而不是强依赖原始训练环境的工具。

## 10. 适合继续扩展的方向

如果后续要扩展该 viewer，比较自然的方向有：

- 增加更多中间图层显示，例如 mirror weight、visibility、depth。
- 增加轨迹录制和相机路径回放。
- 增加多窗口或多图对比显示。
- 将 UI 状态和参数热更新持久化。
- 为截图增加“带面板”和“纯渲染图”两种导出模式。
- 为椭球调试增加按属性着色，例如按 opacity、scale、refl ratio 着色。

## 11. 一句话总结

`DecoupledGS` 的 viewer 本质上是一个“基于 OpenCV 的本地实时调试壳层”：它复用了项目已有的 Gaussian / deferred renderer，用轻量相机控制和轻量 UI 把最终结果、基础颜色、反射中间量以及 Gaussian 椭球结构统一暴露给用户，并通过按需渲染和 hybrid 椭球预览机制在可读性与交互性之间取得平衡。
