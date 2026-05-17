"""
main.py
"""
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

from point_cloud_io import PointCloudIO
from annotation_tool import MultiGlassAnnotationTool
from bezier_fitting import PiecewiseBezierFitting 
# 引入新的类
from bezier_surface import ArchBezierSurfaceFitting 

def center_point_cloud(pcd):
    center = pcd.get_center()
    pcd.translate(-center)
    return center, center

def restore_point_cloud(pcd, translation):
    pcd.translate(translation)

def calculate_density(pcd, num_samples=1000):
    if len(pcd.points) == 0: return 1000
    sample = pcd.random_down_sample(num_samples / len(pcd.points))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for point in sample.points:
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(point, 2)
        if k > 1: distances.append(np.sqrt(dist[1]))
    if not distances: return 1000
    avg_dist = np.mean(distances)
    return 1 / (avg_dist ** 2) if avg_dist > 0 else 1000

def process_single_glass(glass_info, original_translation, density):
    """处理单块玻璃重建"""
    glass_id = glass_info['id']
    g_type = glass_info['type']
    points = glass_info['points']
    
    print(f"  - 重建玻璃 #{glass_id + 1} [类型: {g_type}]")
    
    glass_mesh = None
    
    try:
        if g_type == 'flat':
            # 平面逻辑
            fitter = PiecewiseBezierFitting(corner_threshold_angle=60)
            fitter.fit(points, is_closed=True)
            glass_mesh = fitter.create_plane_mesh()
            
        elif g_type == 'curved':
            # 曲面逻辑 - 使用贝塞尔曲面算法
            surf_fitter = ArchBezierSurfaceFitting()
            surf_fitter.fit(points)
            # 生成高分辨率网格以保证平滑
            glass_mesh = surf_fitter.generate_mesh(resolution=60)
            
    except Exception as e:
        print(f"    重建失败: {e}")
        return None

    if glass_mesh is None: return None

    # 网格后处理
    glass_mesh.paint_uniform_color([0.0, 0.6, 1.0])
    glass_mesh.compute_vertex_normals()
    
    # 恢复坐标系
    glass_mesh.translate(original_translation)
    
    # 采样点云
    glass_area = glass_mesh.get_surface_area()
    num_points = int(glass_area * density * 1.5)
    num_points = max(100, min(num_points, 500000))
    
    glass_pcd = glass_mesh.sample_points_poisson_disk(number_of_points=num_points)
    
    colors = [[0.6, 0.8, 0.95], [0.5, 0.9, 0.5], [0.9, 0.6, 0.6]]
    glass_pcd.paint_uniform_color(colors[glass_id % 3])
    
    return glass_pcd

def main():
    parser = argparse.ArgumentParser(description="建筑玻璃重建工具")
    parser.add_argument("--input", type=str, required=True, help="输入PLY文件路径")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = input_path.parent
    file_stem = input_path.stem
    
    # 1. 读取
    pcd_io = PointCloudIO()
    if not pcd_io.load_point_cloud(str(input_path)): return
    pcd = pcd_io.get_point_cloud()
    
    # 2. 归一化
    print("坐标归一化...")
    _, trans_vec = center_point_cloud(pcd) 
    
    # 3. 标注
    anno_tool = MultiGlassAnnotationTool(pcd)
    glass_data_list = anno_tool.run_multi_selection()
    
    if not glass_data_list:
        return

    # 4. 密度
    density = calculate_density(pcd)
    
    # 5. 重建
    generated_glass_list = []
    print(f"\n开始重建 {len(glass_data_list)} 块玻璃...")
    
    for info in glass_data_list:
        res = process_single_glass(info, trans_vec, density)
        if res: generated_glass_list.append(res)
        
    # 6. 保存与显示
    restore_point_cloud(pcd, trans_vec)
    
    if generated_glass_list:
        merged_glass = generated_glass_list[0]
        for g in generated_glass_list[1:]:
            merged_glass += g
            
        final_pcd = pcd + merged_glass
        
        out_path = output_dir / f"{file_stem}_merged.ply"
        o3d.io.write_point_cloud(str(out_path), final_pcd)
        print(f"✅ 保存成功: {out_path}")
        
        o3d.visualization.draw_geometries([final_pcd], window_name="最终结果")
    else:
        print("未生成数据。")

if __name__ == "__main__":
    main()
