import numpy as np
import torch

from utils.graphics_utils import getWorld2View2


def build_projection_matrix_correct(znear, zfar, H, W, K, device=None, dtype=torch.float32):
    znear = float(znear)
    zfar = float(zfar)
    H = float(H)
    W = float(W)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    top = cy / fy * znear
    bottom = -(H - cy) / fy * znear
    right = cx / fx * znear
    left = -(W - cx) / fx * znear
    projection = torch.tensor(
        [
            [2.0 * znear / (right - left), 0.0, (right + left) / (right - left), 0.0],
            [0.0, 2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0],
            [0.0, 0.0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return projection.to(device=device, dtype=dtype)


class ViewerCamera:
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image_name,
        uid,
        HWK,
        data_device="cuda",
        trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale=1.0,
    ):
        self.uid = uid
        self.colmap_id = colmap_id
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as exc:
            print(exc)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.HWK = (
            int(HWK[0]),
            int(HWK[1]),
            np.asarray(HWK[2], dtype=np.float32).copy(),
        )
        self.image_height = int(self.HWK[0])
        self.image_width = int(self.HWK[1])
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.asarray(trans, dtype=np.float32)
        self.scale = float(scale)

        R_np = np.asarray(R, dtype=np.float32)
        T_np = np.asarray(T, dtype=np.float32)
        self.world_view_transform = torch.tensor(
            getWorld2View2(R_np, T_np, self.trans, self.scale),
            dtype=torch.float32,
            device=self.data_device,
        ).transpose(0, 1).contiguous()
        self.projection_matrix = build_projection_matrix_correct(
            self.znear,
            self.zfar,
            self.HWK[0],
            self.HWK[1],
            self.HWK[2],
            device=self.data_device,
            dtype=self.world_view_transform.dtype,
        ).transpose(0, 1).contiguous()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]
        self.R = torch.tensor(R_np, dtype=torch.float32, device=self.data_device)
        self.T = torch.tensor(T_np, dtype=torch.float32, device=self.data_device)


def build_camera_from_c2w(c2w, fovx, fovy, hwk, data_device="cuda"):
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    return ViewerCamera(
        colmap_id=0,
        R=w2c[:3, :3].astype(np.float32),
        T=w2c[:3, 3].astype(np.float32),
        FoVx=fovx,
        FoVy=fovy,
        image_name="viewer",
        uid=0,
        HWK=hwk,
        data_device=data_device,
    )
