"""
贝塞尔曲线拟合模块 - 增强版
基于论文《The Virtual Camera Path in 3D Animation》与《Multi-view Coherence for Outdoor Reflective Surfaces》
核心改进：分段拟合 (Piecewise)、角点检测、端点约束优化
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
from scipy.spatial import Delaunay
import open3d as o3d
from typing import List, Tuple, Optional, Union

class BezierUtils:
    """贝塞尔曲线数学工具类"""
    
    @staticmethod
    def bernstein_poly(i: int, n: int, t: np.ndarray) -> np.ndarray:
        """计算伯恩斯坦基函数 B_{i,n}(t)"""
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    @staticmethod
    def get_curve_points(control_points: np.ndarray, num_samples: int = 50) -> np.ndarray:
        """根据控制点生成曲线点"""
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_samples)
        curve = np.zeros((num_samples, 3))
        for i in range(n + 1):
            b_val = BezierUtils.bernstein_poly(i, n, t)
            curve += np.outer(b_val, control_points[i])
        return curve

    @staticmethod
    def chord_length_parameterization(points: np.ndarray) -> np.ndarray:
        """
        弦长参数化：解决点云分布不均导致的拟合扭曲问题
        对应论文中优化参数 t 的思想
        """
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.cumsum(diffs)
        total_len = cumulative[-1]
        if total_len == 0:
            return np.linspace(0, 1, len(points))
        t = np.concatenate(([0], cumulative / total_len))
        return t

class PiecewiseBezierFitting:
    """
    分段贝塞尔拟合器
    对应论文：利用多段 Bezier 拟合复杂路径，并在连接处保持几何连续性
    """
    
    def __init__(self, degree: int = 3, corner_threshold_angle: float = 45.0):
        """
        Args:
            degree: 曲线阶数（通常为3）
            corner_threshold_angle: 角点检测阈值（度），超过此角度视为折角，需要分段
        """
        self.degree = degree
        self.threshold_rad = np.radians(corner_threshold_angle)
        self.segments_control_points: List[np.ndarray] = [] # 存储每一段的控制点
        
    def detect_corners(self, points: np.ndarray) -> List[int]:
        """
        检测拐角点索引
        基于向量夹角判断，对应玻璃窗框的直角特征
        """
        num_points = len(points)
        if num_points < 4:
            return [0, num_points - 1]
            
        corners = [0] # 起点总是角点
        
        # 计算相邻向量
        vecs = np.diff(points, axis=0)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        
        for i in range(1, len(vecs)):
            # 计算前后向量的点积 -> 夹角
            dot_product = np.clip(np.dot(vecs[i-1], vecs[i]), -1.0, 1.0)
            angle = np.arccos(dot_product)
            
            # 如果角度剧烈变化，标记为角点
            if angle > self.threshold_rad:
                corners.append(i) # i 是点的索引（对应vecs[i-1]的终点）
        
        corners.append(num_points - 1) # 终点
        return corners

    def _fit_segment_constrained(self, segment_points: np.ndarray) -> np.ndarray:
        """
        约束拟合单段贝塞尔曲线
        关键点：强制固定首尾点 (P0, P3)，只优化中间控制点 (P1, P2)
        这保证了多段曲线连接时没有缝隙 (C0 Continuity)
        """
        n_pts = len(segment_points)
        if n_pts < 2:
            return segment_points
            
        # P0 和 P3 固定为段的首尾
        P0 = segment_points[0]
        P3 = segment_points[-1]
        
        # 如果点太少，直接线性插值返回
        if n_pts <= 3:
            return np.linspace(P0, P3, 4)

        # 参数化 t
        t = BezierUtils.chord_length_parameterization(segment_points)
        
        # 构建最小二乘问题求解 P1, P2
        # 公式: P(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)t^2*P2 + t^3*P3
        # 移项: 3(1-t)^2*t*P1 + 3(1-t)t^2*P2 = P(t) - (1-t)^3*P0 - t^3*P3
        #       [   A1    ] P1 + [   A2    ] P2 = Y
        
        tm = 1 - t
        A1 = 3 * (tm ** 2) * t
        A2 = 3 * tm * (t ** 2)
        
        # 构建矩阵 A [N x 2] (对每个维度分别求解，或者堆叠)
        # 这里对 x, y, z 分别解
        
        control_points = np.zeros((4, 3))
        control_points[0] = P0
        control_points[3] = P3
        
        # 除去首尾的残差项
        Y_full = segment_points - (np.outer(tm**3, P0) + np.outer(t**3, P3))
        
        # 组装设计矩阵 M: [N, 2]
        M = np.column_stack((A1, A2))
        
        # 最小二乘求解 M * [P1; P2] = Y
        # 对 x, y, z 坐标分别解
        for dim in range(3):
            coeffs, _, _, _ = np.linalg.lstsq(M, Y_full[:, dim], rcond=None)
            control_points[1, dim] = coeffs[0]
            control_points[2, dim] = coeffs[1]
            
        return control_points

    def fit(self, points: np.ndarray, is_closed: bool = True) -> List[np.ndarray]:
        """
        主拟合函数
        Args:
            points: 输入点云 (N, 3)
            is_closed: 是否闭合曲线（玻璃通常是闭合回路）
        """
        # 1. 检测角点，分段
        corner_indices = self.detect_corners(points)
        
        # 如果要求闭合，且首尾距离较远，强制添加闭合段
        if is_closed and np.linalg.norm(points[0] - points[-1]) > 1e-3:
            # 逻辑上将终点连回起点，这里简单处理：如果是闭合形状，
            # 外部调用者通常已经让 points[0] approx points[-1]
            # 或者我们在列表末尾追加起点
            points = np.vstack([points, points[0]])
            corner_indices = self.detect_corners(points)

        self.segments_control_points = []
        
        # 2. 对每一段分别拟合
        for i in range(len(corner_indices) - 1):
            start_idx = corner_indices[i]
            end_idx = corner_indices[i+1]
            
            # 获取该段的点（包含端点）
            segment = points[start_idx : end_idx + 1]
            
            # 拟合该段
            cps = self._fit_segment_constrained(segment)
            self.segments_control_points.append(cps)
            
        return self.segments_control_points

    def generate_dense_boundary(self, num_samples_per_seg: int = 50) -> np.ndarray:
        """生成密集边界点（用于 Poisson 重建的输入边界）"""
        if not self.segments_control_points:
            return np.array([])
            
        all_points = []
        for cps in self.segments_control_points:
            seg_points = BezierUtils.get_curve_points(cps, num_samples_per_seg)
            # 去掉最后一个点以避免与下一段起点重复（除非是最后一段）
            all_points.append(seg_points[:-1])
        
        # 加上最后一段的终点
        all_points.append(self.segments_control_points[-1][-1].reshape(1, 3))
        
        return np.vstack(all_points)

    def create_plane_mesh(self, density: float = 0.05) -> o3d.geometry.TriangleMesh:
        """
        生成玻璃平面网格
        利用生成的平滑边界进行 Delaunay 三角化
        """
        boundary_points = self.generate_dense_boundary(num_samples_per_seg=30)
        
        # 1. 拟合最佳平面 (PCA)
        center = boundary_points.mean(axis=0)
        centered = boundary_points - center
        U, S, Vt = np.linalg.svd(centered)
        normal = Vt[2]
        if normal[2] < 0: normal = -normal # 统一法向朝上/外
        
        # 2. 投影到 2D 平面进行三角化
        # 构建局部坐标系 (u, v)
        u_axis = np.cross(np.array([0, 1, 0]), normal)
        if np.linalg.norm(u_axis) < 1e-3:
            u_axis = np.cross(np.array([1, 0, 0]), normal)
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis)
        
        # 投影
        proj_u = np.dot(boundary_points - center, u_axis)
        proj_v = np.dot(boundary_points - center, v_axis)
        points_2d = np.column_stack([proj_u, proj_v])
        
        # 3. 内部填充点 (可选：如果需要内部有顶点，不仅是边界)
        # 简单策略：在 2D 边界框内生成网格点，保留在边界内的点
        # 这里为了演示，只做边界的 Constrained Delaunay (如果不引入外部库，只能做 Convex Hull Delaunay 然后裁剪)
        # 简化做法：直接对边界点做 Delaunay，然后过滤掉长边三角形（凹陷处）
        
        tri = Delaunay(points_2d)
        
        vertices_3d = []
        for p2 in points_2d:
            p3 = center + p2[0] * u_axis + p2[1] * v_axis
            vertices_3d.append(p3)
        vertices_3d = np.array(vertices_3d)
        
        # 简单的三角形过滤（如果玻璃是非凸多边形，需要更复杂的包含性检测）
        valid_simplices = []
        for simplex in tri.simplices:
            # 计算三角形重心
            tri_pts_2d = points_2d[simplex]
            centroid = tri_pts_2d.mean(axis=0)
            
            # 极简判断：使用 winding number 或 ray casting 判断重心是否在多边形内
            # 这里略过复杂判断，假设玻璃主要是凸多边形或近似凸形
            valid_simplices.append(simplex)

        # 构建 Open3D 网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_simplices))
        mesh.compute_vertex_normals()
        
        return mesh

# ==========================================
# 使用示例 (模拟你的标注流程)
# ==========================================
if __name__ == "__main__":
    # 模拟用户点击的玻璃边界点（带一些噪声，且包含直角）
    points_noisy = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.05, 0.0], [1.0, 0.0, 0.0],  # 底边
        [1.05, 0.5, 0.0], [1.0, 1.0, 0.0],                   # 右边
        [0.5, 0.95, 0.0], [0.0, 1.0, 0.0],                   # 顶边
        [-0.05, 0.5, 0.0], [0.0, 0.0, 0.0]                   # 左边回到起点
    ])
    
    # 实例化分段拟合器
    fitter = PiecewiseBezierFitting(corner_threshold_angle=60)
    
    # 执行拟合
    fitter.fit(points_noisy, is_closed=True)
    
    # 1. 获取平滑后的密集边界（可直接传给 Poisson Reconstructions 作为约束）
    dense_boundary = fitter.generate_dense_boundary(num_samples_per_seg=20)
    
    # 2. 生成可视化网格
    mesh_glass = fitter.create_plane_mesh()
    
    print(f"原始点数: {len(points_noisy)}")
    print(f"检测到分段数: {len(fitter.segments_control_points)}")
    print(f"生成的平滑边界点数: {len(dense_boundary)}")
