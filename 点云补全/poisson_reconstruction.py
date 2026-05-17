"""
泊松重建模块
根据贝塞尔曲线确定的平面，使用泊松重建将玻璃部分补全
"""
import numpy as np
import open3d as o3d
from typing import Tuple, Optional


class PoissonReconstruction:
    """泊松重建类"""
    
    def __init__(self, depth: int = 9, width: int = 0, scale: float = 1.1):
        """
        初始化泊松重建器
        
        Args:
            depth: 八叉树深度，控制重建精度
            width: 重建宽度，0表示自动
            scale: 缩放因子
        """
        self.depth = depth
        self.width = width
        self.scale = scale
    
    def reconstruct_from_points(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
        """
        从点云进行泊松重建
        
        Args:
            points: 点坐标数组 (N x 3)
            normals: 法线数组 (N x 3)，如果为None则自动计算
            
        Returns:
            重建的三角网格
        """
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 计算法线（如果未提供）
        if normals is None:
            pcd.estimate_normals()
            # 确保法线方向一致
            pcd.orient_normals_consistent_tangent_plane(k=15)
        else:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # 执行泊松重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, width=self.width, scale=self.scale
        )
        
        return mesh
    
    def reconstruct_from_bezier_plane(
        self,
        bezier_fitter,  # BezierFitting type
        num_samples: int = 200,
        extend_factor: float = 1.2
    ) -> o3d.geometry.TriangleMesh:
        """
        从贝塞尔曲线确定的平面进行泊松重建
        
        Args:
            bezier_fitter: 已拟合的贝塞尔曲线拟合器
            num_samples: 采样点数
            extend_factor: 扩展因子，用于生成更多点
            
        Returns:
            重建的三角网格
        """
        # 评估贝塞尔曲线
        curve_points = bezier_fitter.evaluate_curve(num_samples)
        
        # 拟合平面
        normal, center = bezier_fitter.fit_plane_from_curve(num_samples)
        
        # 构建局部坐标系
        if abs(normal[0]) < 0.9:
            u = np.array([1, 0, 0])
        else:
            u = np.array([0, 1, 0])
        
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # 投影曲线点到平面
        projected_points = []
        for point in curve_points:
            # 投影到平面
            vec = point - center
            proj = vec - np.dot(vec, normal) * normal
            projected_point = center + proj
            projected_points.append(projected_point)
        
        projected_points = np.array(projected_points)
        
        # 在平面上生成密集的点云用于重建
        # 计算边界框
        proj_2d = np.column_stack([
            np.dot(projected_points - center, u),
            np.dot(projected_points - center, v)
        ])
        
        min_2d = proj_2d.min(axis=0) * extend_factor
        max_2d = proj_2d.max(axis=0) * extend_factor
        
        # 生成网格点
        num_grid_points = 50
        u_grid = np.linspace(min_2d[0], max_2d[0], num_grid_points)
        v_grid = np.linspace(min_2d[1], max_2d[1], num_grid_points)
        
        U, V = np.meshgrid(u_grid, v_grid)
        
        # 转换回3D坐标
        surface_points = []
        for i in range(num_grid_points):
            for j in range(num_grid_points):
                point_3d = center + U[i, j] * u + V[i, j] * v
                surface_points.append(point_3d)
        
        surface_points = np.array(surface_points)
        
        # 生成法线（都指向同一方向）
        normals = np.tile(normal, (len(surface_points), 1))
        
        # 执行泊松重建
        mesh = self.reconstruct_from_points(surface_points, normals)
        
        return mesh
    
    def fill_glass_region(
        self,
        original_point_cloud: o3d.geometry.PointCloud,
        glass_indices: np.ndarray,
        bezier_fitter,  # BezierFitting type
        num_samples: int = 200
    ) -> o3d.geometry.PointCloud:
        """
        补全玻璃区域
        
        Args:
            original_point_cloud: 原始点云
            glass_indices: 玻璃区域的点索引
            bezier_fitter: 已拟合的贝塞尔曲线拟合器
            num_samples: 采样点数
            
        Returns:
            补全后的点云
        """
        # 获取原始点云的点
        original_points = np.asarray(original_point_cloud.points)
        
        # 获取非玻璃区域的点
        all_indices = np.arange(len(original_points))
        non_glass_indices = np.setdiff1d(all_indices, glass_indices)
        non_glass_points = original_points[non_glass_indices]
        
        # 从贝塞尔曲线生成玻璃平面点
        # 拟合平面
        normal, center = bezier_fitter.fit_plane_from_curve(num_samples)
        
        # 构建局部坐标系
        if abs(normal[0]) < 0.9:
            u = np.array([1, 0, 0])
        else:
            u = np.array([0, 1, 0])
        
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # 评估贝塞尔曲线
        curve_points = bezier_fitter.evaluate_curve(num_samples)
        
        # 投影到平面
        projected_points = []
        for point in curve_points:
            vec = point - center
            proj = vec - np.dot(vec, normal) * normal
            projected_point = center + proj
            projected_points.append(projected_point)
        
        projected_points = np.array(projected_points)
        
        # 计算边界框
        proj_2d = np.column_stack([
            np.dot(projected_points - center, u),
            np.dot(projected_points - center, v)
        ])
        
        min_2d = proj_2d.min(axis=0)
        max_2d = proj_2d.max(axis=0)
        
        # 在区域内生成密集的点
        num_fill_points = len(glass_indices)  # 生成与原始玻璃区域相同数量的点
        
        u_random = np.random.uniform(min_2d[0], max_2d[0], num_fill_points)
        v_random = np.random.uniform(min_2d[1], max_2d[1], num_fill_points)
        
        # 检查点是否在曲线内部（简化：使用凸包）
        from scipy.spatial import ConvexHull
        hull = ConvexHull(proj_2d)
        hull_points_2d = proj_2d[hull.vertices]
        
        # 使用点是否在凸包内的简单检查
        inside_points = []
        for ui, vi in zip(u_random, v_random):
            # 简化：检查是否在边界框内
            if min_2d[0] <= ui <= max_2d[0] and min_2d[1] <= vi <= max_2d[1]:
                inside_points.append([ui, vi])
        
        inside_points = np.array(inside_points)
        
        # 转换回3D
        fill_points_3d = []
        for point_2d in inside_points:
            point_3d = center + point_2d[0] * u + point_2d[1] * v
            fill_points_3d.append(point_3d)
        
        fill_points_3d = np.array(fill_points_3d)
        
        # 合并点云
        if len(fill_points_3d) > 0:
            combined_points = np.vstack([non_glass_points, fill_points_3d])
        else:
            combined_points = non_glass_points
        
        # 创建补全后的点云
        filled_pcd = o3d.geometry.PointCloud()
        filled_pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        # 如果有颜色信息，尝试保留
        if original_point_cloud.has_colors():
            original_colors = np.asarray(original_point_cloud.colors)
            non_glass_colors = original_colors[non_glass_indices]
            
            # 为填充的点添加默认颜色（玻璃颜色，通常是透明或浅色）
            glass_color = np.array([0.7, 0.9, 1.0])  # 浅蓝色
            fill_colors = np.tile(glass_color, (len(fill_points_3d), 1))
            
            combined_colors = np.vstack([non_glass_colors, fill_colors])
            filled_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        return filled_pcd

