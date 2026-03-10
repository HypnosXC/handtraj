import torch
import torch.nn.functional as F

def project_3d_to_2d(
    joints_3d: torch.Tensor,       # (B, J, 3)
    cam_intrinsic: torch.Tensor,   # (B, 3, 3)  或 (B, 4) [fx, fy, cx, cy]
    cam_extrinsic: torch.Tensor | None = None,  # (B, 4, 4) 世界坐标->相机坐标
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    将3D关节点通过相机参数投影到2D像素坐标。
    
    Args:
        joints_3d:     (B, J, 3) 世界坐标系或相机坐标系下的3D关节
        cam_intrinsic: (B, 3, 3) 完整内参矩阵，或 (B, 4) 简化形式 [fx, fy, cx, cy]
        cam_extrinsic: (B, 4, 4) 外参矩阵（可选，若joints_3d已在相机坐标系下则不需要）
    
    Returns:
        joints_2d: (B, J, 2) 像素坐标
    """
    B, J, _ = joints_3d.shape

    # Step 1: 如果有外参，先将世界坐标转到相机坐标
    if cam_extrinsic is not None:
        R = cam_extrinsic[:, :3, :3]  # (B, 3, 3)
        t = cam_extrinsic[:, :3, 3:]  # (B, 3, 1)
        # joints_cam = R @ joints_3d^T + t -> (B, 3, J)
        joints_cam = torch.bmm(R, joints_3d.transpose(1, 2)) + t
        joints_cam = joints_cam.transpose(1, 2)  # (B, J, 3)
    else:
        joints_cam = joints_3d  # 已经在相机坐标系下

    # Step 2: 透视除法
    x = joints_cam[..., 0]  # (B, J)
    y = joints_cam[..., 1]  # (B, J)
    z = joints_cam[..., 2].clamp(min=eps)  # (B, J), 防止除零

    # Step 3: 应用内参
    if cam_intrinsic.dim() == 2 and cam_intrinsic.shape[-1] == 4:
        # 简化形式: [fx, fy, cx, cy]
        fx = cam_intrinsic[:, 0:1]  # (B, 1)
        fy = cam_intrinsic[:, 1:2]
        cx = cam_intrinsic[:, 2:3]
        cy = cam_intrinsic[:, 3:4]
        u = fx * (x / z) + cx  # (B, J)
        v = fy * (y / z) + cy
    else:
        # 完整 3x3 内参矩阵
        # K @ [x/z, y/z, 1]^T
        fx = cam_intrinsic[:, 0, 0].unsqueeze(-1)
        fy = cam_intrinsic[:, 1, 1].unsqueeze(-1)
        cx = cam_intrinsic[:, 0, 2].unsqueeze(-1)
        cy = cam_intrinsic[:, 1, 2].unsqueeze(-1)
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

    joints_2d = torch.stack([u, v], dim=-1)  # (B, J, 2)
    return joints_2d


def normalize_2d_keypoints(
    joints_2d: torch.Tensor,  # (B, J, 2) 像素坐标
    img_width: int = 224,
    img_height: int = 224,
) -> torch.Tensor:
    """归一化到 [-1, 1] 范围，方便与网络预测对齐"""
    joints_2d_norm = joints_2d.clone()
    joints_2d_norm[..., 0] = joints_2d[..., 0] / img_width * 2.0 - 1.0
    joints_2d_norm[..., 1] = joints_2d[..., 1] / img_height * 2.0 - 1.0
    return joints_2d_norm