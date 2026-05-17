"""
annotation_tool.py - 6点标注
"""
import open3d as o3d
import numpy as np
from typing import List, Dict

# ==========================================
# 基础工具类
# ==========================================
class SimpleAnnotationTool:
    def __init__(self, point_cloud: o3d.geometry.PointCloud):
        self.pcd = point_cloud

    def run(self) -> np.ndarray:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("选点工具 (Shift+左键)", width=1024, height=768)
        vis.add_geometry(self.pcd)
        
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0, 0, 0])
        
        vis.run() 
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        
        if picked_indices:
            # 保持点序
            selected_coords = np.asarray(self.pcd.points)[picked_indices]
            return selected_coords
        return np.array([])

# ==========================================
# 主标注逻辑
# ==========================================
class MultiGlassAnnotationTool:
    def __init__(self, point_cloud: o3d.geometry.PointCloud):
        self.pcd = point_cloud
        self.glass_data = [] 
    
    def run_multi_selection(self) -> List[Dict]:
        print("\n=== 多玻璃标注模式 ===")
        print("1. 平面玻璃: 标注3个以上角点")
        print("2. 曲面玻璃(圆拱): 严格标注6个点")
        print("====================\n")
        
        glass_count = 0
        while True:
            glass_count += 1
            print(f"\n--- 标注第 {glass_count} 块玻璃 ---")
            
            glass_type = self.ask_glass_type()
            coords = np.array([])
            
            if glass_type == 'flat':
                print(">> [平面] 请顺时针标注边框角点 (按Q完成)")
                tool = SimpleAnnotationTool(self.pcd)
                coords = tool.run()
                if len(coords) < 3:
                    print("⚠️ 点数不足，跳过。")
                    if not self.ask_continue(): break
                    continue

            elif glass_type == 'curved':
                print(">> [曲面-圆拱] 请按以下顺序标注 6 个点:")
                print("   1. 第一条拱形边: 起点 -> 弧顶(最高点) -> 终点")
                print("   2. 第二条拱形边: 起点 -> 弧顶(最高点) -> 终点")
                print("   (注: 两条边的起点侧应对应，若点反了程序会自动修正)")
                
                tool = SimpleAnnotationTool(self.pcd)
                coords = tool.run()
                
                if len(coords) != 6:
                    print(f"⚠️ 错误: 必须标注 6 个点，检测到 {len(coords)} 个。")
                    if not self.ask_continue(): break
                    continue

            self.glass_data.append({
                'id': glass_count - 1,
                'type': glass_type,
                'points': coords
            })
            
            if not self.ask_continue():
                break
        
        return self.glass_data

    def ask_glass_type(self) -> str:
        while True:
            choice = input("请选择类型 [1:平面, 2:曲面]: ").strip()
            if choice == '2': return 'curved'
            if choice in ['1', '']: return 'flat'
            
    def ask_continue(self) -> bool:
        return input("继续标注下一块? (y/n): ").strip().lower() in ['y', 'yes', '是']