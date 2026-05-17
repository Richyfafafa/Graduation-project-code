"""
bezier_surface.py - 基于张量积的贝塞尔曲面 
"""
import numpy as np
import open3d as o3d
from scipy.special import comb

class BezierMath:
    """贝塞尔数学工具类"""
    
    @staticmethod
    def bernstein(n, i, t):
        """
        伯恩斯坦基函数 B_{i,n}(t)
        公式: C(n,i) * t^i * (1-t)^(n-i)
        """
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    @staticmethod
    def calculate_surface_point(u, v, control_points):
        """
        根据 UV 参数计算曲面上的点
        公式: P(u,v) = sum_i sum_j B_i(u) * B_j(v) * P_ij
        Args:
            u, v: 参数 [0, 1]
            control_points: (n+1) x (m+1) x 3 的点阵
        """
        n = control_points.shape[0] - 1 # U方向阶数
        m = control_points.shape[1] - 1 # V方向阶数
        
        point = np.zeros(3)
        
        for i in range(n + 1):
            b_u = BezierMath.bernstein(n, i, u)
            for j in range(m + 1):
                b_v = BezierMath.bernstein(m, j, v)
                
                # 张量积求和
                point += b_u * b_v * control_points[i, j]
                
        return point

class ArchBezierSurfaceFitting:
    """
    圆拱形曲面构建器
    将 6 个边界点转换为 3x3 控制点矩阵，进而生成贝塞尔曲面
    """
    def __init__(self):
        self.control_net = None # 存储生成的控制点网格

    def fit(self, boundary_points):
        """
        Args:
            boundary_points: 6个点
            顺序: [Arc1_Start, Arc1_Peak, Arc1_End, Arc2_Start, Arc2_Peak, Arc2_End]
        """
        pts = np.array(boundary_points)
        if len(pts) != 6:
            raise ValueError("必须提供 6 个点")

        # 1. 分组：前3个点为拱门A，后3个点为拱门B
        group_a = pts[0:3] # [Start, Peak, End]
        group_b = pts[3:6] # [Start, Peak, End]

        # 2. 自动修正方向：防止曲面扭曲
        dist_direct = np.linalg.norm(group_a[0] - group_b[0])
        dist_cross = np.linalg.norm(group_a[0] - group_b[2])
        
        if dist_cross < dist_direct:
            print("  [提示] 检测到点序相反，已自动翻转第二组点以匹配方向")
            group_b = group_b[::-1] 

        # 3. 计算贝塞尔控制点 (反求中间控制点)
        # 为了让贝塞尔曲线经过 Peak 点 (t=0.5)，中间的控制点 P_ctrl 计算公式为:
        # P_ctrl = 2 * Peak - 0.5 * Start - 0.5 * End
        
        # 拱门 A 的控制点行 (3个点)
        ctrl_a_mid = 2 * group_a[1] - 0.5 * group_a[0] - 0.5 * group_a[2]
        row_a = np.array([group_a[0], ctrl_a_mid, group_a[2]])
        
        # 拱门 B 的控制点行 (3个点)
        ctrl_b_mid = 2 * group_b[1] - 0.5 * group_b[0] - 0.5 * group_b[2]
        row_b = np.array([group_b[0], ctrl_b_mid, group_b[2]])

        # 4. 构建 3x3 控制网格 (Tensor Product Grid)
        # Row 0: 拱门 A
        # Row 2: 拱门 B
        # Row 1: 拱门 A 和 B 的线性插值中点 (这一步保证了纵向是平直的隧道)
        row_mid = (row_a + row_b) / 2
        
        self.control_net = np.array([row_a, row_mid, row_b]) 
        # 形状为 (3, 3, 3) -> 这是一个二次x二次的贝塞尔曲面

    def generate_mesh(self, resolution=50):
        if self.control_net is None:
            raise ValueError("Run fit() first")

        # 生成 UV 网格
        u_vals = np.linspace(0, 1, resolution)
        v_vals = np.linspace(0, 1, resolution)
        
        vertices = []
        
        # 双重循环计算曲面点 
        for u in u_vals:
            for v in v_vals:
                pt = BezierMath.calculate_surface_point(u, v, self.control_net)
                vertices.append(pt)
        
        vertices = np.array(vertices)
        
        # 生成拓扑 (三角形面片)
        triangles = []
        for i in range(resolution - 1):     # u 方向索引
            for j in range(resolution - 1): # v 方向索引
                # 网格索引计算
                # 顶点排列是先v后u (或者先u后v，取决于上面的循环顺序)
                # 上面是 u 外层，v 内层 -> index = i * resolution + j
                
                p0 = i * resolution + j
                p1 = i * resolution + (j + 1)
                p2 = (i + 1) * resolution + j
                p3 = (i + 1) * resolution + (j + 1)
                
                
                triangles.append([p0, p2, p1])
                triangles.append([p1, p2, p3])
        
        # 创建 Open3D Mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        
        return mesh