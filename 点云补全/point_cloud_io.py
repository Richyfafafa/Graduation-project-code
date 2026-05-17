"""
Point cloud IO utilities.
Supports PLY and COLMAP points3D.bin.
"""

import struct
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


class PointCloudIO:
    """Point cloud input/output helper."""

    def __init__(self):
        self.pcd: Optional[o3d.geometry.PointCloud] = None

    def load_ply(self, file_path: str) -> bool:
        """Load a PLY point cloud file."""
        try:
            self.pcd = o3d.io.read_point_cloud(file_path)
            if len(self.pcd.points) == 0:
                print(f"Warning: Point cloud file is empty: {file_path}")
                return False

            if not self.pcd.has_colors():
                self.pcd.paint_uniform_color([0.5, 0.5, 0.5])

            print(f"Loaded point cloud: {len(self.pcd.points)} points")
            return True
        except Exception as e:
            print(f"Failed to load point cloud file: {e}")
            return False

    def _load_colmap_points3d_bin(self, file_path: str) -> bool:
        """Load COLMAP points3D.bin as an Open3D point cloud."""
        try:
            points = []
            colors = []

            with open(file_path, "rb") as f:
                num_points = struct.unpack("<Q", f.read(8))[0]

                for _ in range(num_points):
                    # POINT3D_ID, X, Y, Z, R, G, B, ERROR
                    point_record = f.read(43)
                    if len(point_record) < 43:
                        raise ValueError("Invalid points3D.bin: unexpected EOF in point record")

                    _, x, y, z, r, g, b, _ = struct.unpack("<QdddBBBd", point_record)
                    points.append([x, y, z])
                    colors.append([r / 255.0, g / 255.0, b / 255.0])

                    # TRACK[] with track_length entries, each entry = (IMAGE_ID, POINT2D_IDX) = 2 int32 = 8 bytes
                    track_len_bytes = f.read(8)
                    if len(track_len_bytes) < 8:
                        raise ValueError("Invalid points3D.bin: unexpected EOF in track length")
                    track_len = struct.unpack("<Q", track_len_bytes)[0]
                    f.seek(8 * track_len, 1)

            if len(points) == 0:
                print(f"Warning: COLMAP point cloud file is empty: {file_path}")
                return False

            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

            print(f"Loaded COLMAP point cloud: {len(points)} points")
            return True
        except Exception as e:
            print(f"Failed to load COLMAP points3D.bin: {e}")
            return False

    def load_point_cloud(self, file_path: str) -> bool:
        """Load point cloud based on extension: .ply or .bin."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".ply":
            return self.load_ply(file_path)
        if suffix == ".bin":
            return self._load_colmap_points3d_bin(file_path)

        print(f"Unsupported point cloud format: {file_path}. Only .ply and .bin are supported.")
        return False

    def get_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Get current point cloud object."""
        return self.pcd

    def save_ply(self, file_path: str, point_cloud: Optional[o3d.geometry.PointCloud] = None) -> bool:
        """Save point cloud to PLY."""
        pcd_to_save = point_cloud if point_cloud is not None else self.pcd
        if pcd_to_save is None:
            print("Error: No point cloud available to save")
            return False

        try:
            o3d.io.write_point_cloud(file_path, pcd_to_save)
            print(f"Saved point cloud to: {file_path}")
            return True
        except Exception as e:
            print(f"Failed to save point cloud file: {e}")
            return False

    def get_points_array(self) -> Optional[np.ndarray]:
        """Get point coordinates array (N x 3)."""
        if self.pcd is None:
            return None
        return np.asarray(self.pcd.points)

    def get_colors_array(self) -> Optional[np.ndarray]:
        """Get point colors array (N x 3)."""
        if self.pcd is None or not self.pcd.has_colors():
            return None
        return np.asarray(self.pcd.colors)

    def extract_points_by_indices(self, indices: np.ndarray) -> o3d.geometry.PointCloud:
        """Extract points by indices."""
        if self.pcd is None:
            raise ValueError("Point cloud not loaded")
        return self.pcd.select_by_index(indices)

    def create_point_cloud_from_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> o3d.geometry.PointCloud:
        """Create point cloud from coordinates (+ optional colors)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd

    def get_point_cloud_info(self) -> dict:
        """Get basic point cloud information."""
        if self.pcd is None:
            return {}

        points = np.asarray(self.pcd.points)
        return {
            "num_points": len(points),
            "has_colors": self.pcd.has_colors(),
            "has_normals": self.pcd.has_normals(),
            "bounding_box": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
                "center": points.mean(axis=0).tolist(),
            },
        }
