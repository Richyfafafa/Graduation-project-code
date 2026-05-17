import math
from typing import Optional, Tuple

import numpy as np


def _axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )


class OrbitCamera:
    """Camera controller with orbit + pan + zoom, adapted for local rendering."""

    def __init__(
        self,
        img_wh: Tuple[int, int],
        center: np.ndarray,
        radius: float,
        rot: Optional[np.ndarray] = None,
    ):
        self.W, self.H = img_wh
        self.radius = float(max(radius, 1e-3))
        self.center = np.asarray(center, dtype=np.float32).copy()
        self.rot = np.eye(3, dtype=np.float32) if rot is None else np.asarray(rot, dtype=np.float32).copy()

    @property
    def pose(self) -> np.ndarray:
        """Return camera-to-world (C2W) matrix."""
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.T
        res = rot @ res
        res[:3, 3] -= self.center
        return res

    def set_pose(self, c2w: np.ndarray, keep_radius: bool = True):
        c2w = np.asarray(c2w, dtype=np.float32)
        c2w_rot = c2w[:3, :3]
        c2w_t = c2w[:3, 3]
        self.rot = c2w_rot.T
        if keep_radius:
            self.center = -(c2w_t + self.radius * c2w_rot[:, 2])
        else:
            self.center = -c2w_t
            self.radius = max(np.linalg.norm(c2w_t + self.center), 1e-3)

    def set_orbit_pivot(self, pivot_world: np.ndarray, preserve_view: bool = True):
        """Set orbit pivot in world space for mouse-left orbit interaction."""
        pivot_world = np.asarray(pivot_world, dtype=np.float32).reshape(3)
        if preserve_view:
            current_pose = self.pose
            cam_pos = current_pose[:3, 3]
            forward = current_pose[:3, 2]
            to_pivot = pivot_world - cam_pos
            projected_radius = float(np.dot(to_pivot, forward))
            if projected_radius <= 1e-3:
                projected_radius = float(np.linalg.norm(to_pivot))
            self.radius = float(max(projected_radius, 1e-3))
        self.center = -pivot_world

    def orbit(self, dx: float, dy: float, sensitivity_deg: float = 1.0):
        yaw = sensitivity_deg * dx
        pitch = -sensitivity_deg * dy
        self.yaw(yaw)
        self.pitch(pitch)

    def yaw(self, deg: float):
        mat = _axis_angle_to_matrix(self.rot[:, 1], math.radians(deg))
        self.rot = mat @ self.rot

    def pitch(self, deg: float):
        mat = _axis_angle_to_matrix(self.rot[:, 0], math.radians(deg))
        self.rot = mat @ self.rot

    def roll(self, deg: float):
        mat = _axis_angle_to_matrix(self.rot[:, 2], math.radians(deg))
        self.rot = mat @ self.rot

    def zoom(self, delta: float):
        self.radius *= 1.1 ** (-delta)
        self.radius = float(max(self.radius, 1e-3))

    def pan_pixels(self, dx: float, dy: float, scale: float = 1.5e-4):
        self.center += scale * (self.rot.T @ np.array([dx, dy, 0.0], dtype=np.float32))

    def translate_local(self, dx: float, dy: float, dz: float):
        self.center += self.rot.T @ np.array([dx, dy, dz], dtype=np.float32)
