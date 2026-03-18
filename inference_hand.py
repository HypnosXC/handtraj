from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import yaml
import random
import os
import cv2
import pyrender
import trimesh
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm

from egoallo.data.dataclass import collate_dataclass
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.sampling import CosineNoiseScheduleConstants
from egoallo.inference_utils import load_hand_denoiser
from egoallo.data.dataclass import HandTrainingData
from egoallo.data.hand_data import HandHdf5Dataset
from src.egoallo import network, hand_network
from manopth.manolayer import ManoLayer

try:
    from hamer.utils.mesh_renderer import create_raymond_lights
except ImportError:
    create_raymond_lights = None


# ============================================================================
# Timestep schedules
# ============================================================================

def quadratic_ts() -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = 1000
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]


def linear_ts(num_steps: int = 50) -> np.ndarray:
    """Uniform timestep schedule for flow matching (from 1.0 to 0.0)."""
    return np.linspace(1.0, 0.0, num_steps + 1)


# ============================================================================
# Procrustes alignment & PA metrics
# ============================================================================

def compute_procrustes_alignment(
    pred: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    """Procrustes alignment: find optimal rotation, translation, and scale
    to align pred to gt.

    Args:
        pred: (N, 3) predicted points.
        gt:   (N, 3) ground truth points.

    Returns:
        aligned: (N, 3) aligned prediction.
    """
    mu_pred = pred.mean(axis=0, keepdims=True)
    mu_gt = gt.mean(axis=0, keepdims=True)
    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    # Optimal rotation via SVD.
    H = pred_c.T @ gt_c  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    # Correct for reflection.
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_mat @ U.T

    # Optimal scale.
    scale = np.trace(R @ H) / np.trace(pred_c.T @ pred_c)

    # Apply alignment.
    aligned = scale * pred_c @ R.T + mu_gt
    return aligned


def compute_pa_mpjpe(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Procrustes-Aligned MPJPE (mm).

    Args:
        pred_joints: (B, T, J, 3)
        gt_joints:   (B, T, J, 3)
        mask:        (B, T) bool, valid frames.

    Returns:
        Scalar PA-MPJPE in mm.
    """
    pred_np = pred_joints.detach().cpu().numpy()
    gt_np = gt_joints.detach().cpu().numpy()
    B, T, J, _ = pred_np.shape

    errors = []
    for b in range(B):
        for t in range(T):
            if mask is not None and not mask[b, t]:
                continue
            aligned = compute_procrustes_alignment(pred_np[b, t], gt_np[b, t])
            err = np.sqrt(((aligned - gt_np[b, t]) ** 2).sum(axis=-1)).mean()
            errors.append(err)
    if len(errors) == 0:
        return 0.0
    return float(np.mean(errors)) * 1000  # m -> mm


def compute_pa_mpvpe(
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Procrustes-Aligned MPVPE (mm).

    Args:
        pred_verts: (B, T, V, 3)
        gt_verts:   (B, T, V, 3)
        mask:       (B, T) bool, valid frames.

    Returns:
        Scalar PA-MPVPE in mm.
    """
    pred_np = pred_verts.detach().cpu().numpy()
    gt_np = gt_verts.detach().cpu().numpy()
    B, T, V, _ = pred_np.shape

    errors = []
    for b in range(B):
        for t in range(T):
            if mask is not None and not mask[b, t]:
                continue
            aligned = compute_procrustes_alignment(pred_np[b, t], gt_np[b, t])
            err = np.sqrt(((aligned - gt_np[b, t]) ** 2).sum(axis=-1)).mean()
            errors.append(err)
    if len(errors) == 0:
        return 0.0
    return float(np.mean(errors)) * 1000  # m -> mm


# ============================================================================
# Rendering helpers
# ============================================================================

def render_joint(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a hand mesh into an image using pyrender."""
    render_res = (h, w)
    renderer = pyrender.OffscreenRenderer(
        viewport_width=render_res[1], viewport_height=render_res[0], point_size=1.0
    )
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=(0.3, 0.3, 0.3)
    )
    scene.add(mesh, "mesh")

    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics[0], fy=intrinsics[1],
        cx=intrinsics[2], cy=intrinsics[3],
        zfar=1e12, znear=0.001,
    )
    if create_raymond_lights is not None:
        for node in create_raymond_lights():
            scene.add_node(node)

    camera_pose = np.eye(4)
    camera_pose[1:3, :] *= -1
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = color[..., -1] > 0
    return color[..., :3], rend_depth, mask


# ============================================================================
# 2D pose visualization
# ============================================================================

# MANO joint connectivity for visualization (parent -> child).
MANO_JOINT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]

FINGER_COLORS = [
    (255, 0, 0),     # thumb - red
    (0, 255, 0),     # index - green
    (0, 0, 255),     # middle - blue
    (255, 255, 0),   # ring - yellow
    (255, 0, 255),   # pinky - magenta
]


def draw_2d_joints_on_image(
    image: np.ndarray,
    joints_2d: np.ndarray,
    confidence: np.ndarray | None = None,
    radius: int = 3,
    thickness: int = 2,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Draw 2D hand joints and skeleton on an image.

    Args:
        image: (H, W, 3) BGR image.
        joints_2d: (21, 2) 2D joint positions (x, y).
        confidence: (21,) optional confidence per joint.
        radius: joint circle radius.
        thickness: bone line thickness.
        conf_threshold: min confidence to draw.

    Returns:
        Image with drawn joints.
    """
    img = image.copy()
    J = joints_2d.shape[0]

    for finger_idx, (i, j) in enumerate(MANO_JOINT_CONNECTIONS):
        if i >= J or j >= J:
            continue
        if confidence is not None:
            if confidence[i] < conf_threshold or confidence[j] < conf_threshold:
                continue
        color = FINGER_COLORS[finger_idx // 4]
        pt1 = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        pt2 = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        cv2.line(img, pt1, pt2, color, thickness)

    for j_idx in range(J):
        if confidence is not None and confidence[j_idx] < conf_threshold:
            continue
        pt = (int(joints_2d[j_idx, 0]), int(joints_2d[j_idx, 1]))
        cv2.circle(img, pt, radius, (255, 255, 255), -1)
        cv2.circle(img, pt, radius, (0, 0, 0), 1)

    return img


def visualize_2d_pose(
    rgb_frames: torch.Tensor,
    pred_joints_2d: torch.Tensor,
    gt_joints_2d: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    out_dir: str = "tmp/visualize_2d",
    fps: int = 10,
) -> None:
    """Save a video with predicted (and optionally GT) 2D joints overlaid on RGB.

    Args:
        rgb_frames: (T, H, W, 3) uint8 RGB frames.
        pred_joints_2d: (T, 21, 2) predicted 2D joints.
        gt_joints_2d: (T, 21, 2) optional GT 2D joints.
        confidence: (T, 21, 1) optional confidence.
        mask: (T,) bool mask for valid frames.
        out_dir: output directory.
        fps: video frame rate.
    """
    os.makedirs(out_dir, exist_ok=True)
    T, H, W, _ = rgb_frames.shape
    pred_np = pred_joints_2d.detach().cpu().numpy()
    gt_np = gt_joints_2d.detach().cpu().numpy() if gt_joints_2d is not None else None
    conf_np = confidence.detach().cpu().numpy().squeeze(-1) if confidence is not None else None
    mask_np = mask.cpu().numpy() if mask is not None else np.ones(T, dtype=bool)

    has_gt = gt_np is not None
    out_w = W * 2 if has_gt else W
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(out_dir, "pose_2d.mp4"), fourcc, fps, (out_w, H))

    for t in range(T):
        if not mask_np[t]:
            continue
        frame_bgr = cv2.cvtColor(rgb_frames[t].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        conf_t = conf_np[t] if conf_np is not None else None

        pred_frame = draw_2d_joints_on_image(frame_bgr, pred_np[t], conf_t)
        if has_gt:
            gt_frame = draw_2d_joints_on_image(frame_bgr.copy(), gt_np[t])
            combined = np.concatenate([gt_frame, pred_frame], axis=1)
        else:
            combined = pred_frame
        writer.write(combined)

    writer.release()

    # Save a single snapshot.
    mid = T // 2
    frame_bgr = cv2.cvtColor(rgb_frames[mid].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    conf_mid = conf_np[mid] if conf_np is not None else None
    snapshot = draw_2d_joints_on_image(frame_bgr, pred_np[mid], conf_mid)
    if has_gt:
        gt_snap = draw_2d_joints_on_image(frame_bgr.copy(), gt_np[mid])
        snapshot = np.concatenate([gt_snap, snapshot], axis=1)
    cv2.imwrite(os.path.join(out_dir, "pose_2d_snapshot.png"), snapshot)
    print(f"2D pose visualization saved at: {out_dir}")


# ============================================================================
# 3D mesh / pose visualization (video)
# ============================================================================

def visualize_mesh_in_rgb(
    trajs: List,
    intrinsics: torch.Tensor,
    rgb_frames: torch.Tensor,
    subseq_len: int,
    out_dir: str = "tmp/visualize_hand",
    fps: int = 10,
    start_frame: int = 0,
    resize: tuple | None = None,
) -> None:
    """Render predicted (and GT) hand meshes overlaid on RGB video.

    Args:
        trajs: list of HandDenoiseTraj (first is GT, rest are predictions).
        intrinsics: (T, 4) or (4,) camera intrinsics [fx, fy, cx, cy].
        rgb_frames: (N, H, W, 3) uint8 RGB frames.
        subseq_len: number of frames to render.
        out_dir: output directory.
        fps: video frame rate.
        start_frame: offset into rgb_frames.
        resize: optional (W, H) resize.
    """
    os.makedirs(out_dir, exist_ok=True)

    vertices_list = []
    faces_list = []
    for traj in trajs:
        vertices, faces, _ = traj.apply_to_hand()
        vertices_list.append(vertices.squeeze(0).cpu().numpy())
        faces_list.append(faces.cpu().numpy())

    intr = intrinsics.cpu().numpy()
    border_color = [255, 0, 0]
    h, w = rgb_frames.shape[1], rgb_frames.shape[2]

    if resize is not None:
        out_w, out_h = resize
    else:
        out_w = w * (len(trajs) + 1)
        out_h = h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(out_dir, "infer.mp4"), fourcc, fps, (out_w, out_h))

    for i in range(subseq_len):
        image = cv2.cvtColor(
            rgb_frames[i + start_frame].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        composited = image.copy()
        panels = [image.copy()]  # first panel: raw image

        for j in range(len(trajs)):
            verts = vertices_list[j]
            faces = faces_list[j]
            frame_intr = intr[i] if len(intr.shape) == 2 else intr
            render_rgb, _, render_mask = render_joint(
                verts[i], faces, frame_intr, h=h, w=w
            )

            overlay = panels[0].copy() if j > 0 else image.copy()
            border_width = 5
            dilated = binary_dilation(
                render_mask, np.ones((border_width, border_width), dtype=bool)
            )[:, :, None]
            overlay = np.where(
                dilated,
                np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
                overlay,
            )
            overlay = np.where(render_mask[:, :, None], render_rgb, overlay)
            panels.append(overlay)

        composited = np.concatenate(panels, axis=1)
        if resize is not None:
            composited = cv2.resize(composited, resize)
        writer.write(composited)

    writer.release()
    print(f"Mesh visualization saved at: {out_dir}")


# ============================================================================
# MANO helper
# ============================================================================

def mano_poses2joints_3d(
    mano_pose: torch.Tensor, mano_betas: torch.Tensor, mano_side: str
) -> torch.Tensor:
    """Convert MANO pose to joint 3D positions."""
    mano_betas = mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1)
    mano_layer = ManoLayer(
        ncomps=45, side=mano_side,
        mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        use_pca=False, flat_hand_mean=True,
    )
    verts, joints = mano_layer(
        mano_pose[:, :48], mano_betas, mano_pose[:, 48:51],
    )
    return joints / 1000


# ============================================================================
# Args
# ============================================================================

@dataclasses.dataclass
class Args:
    checkpoint_dir: Path = Path(
        "/public/home/xuchen/handtraj/experiments/image_1616_sublen_128/v3/checkpoints_300000"
    )
    training_mode: str = "diffusion"
    """Training paradigm: 'diffusion' or 'flow_matching'."""
    num_flow_steps: int = 50
    """Number of Euler steps for flow matching sampling."""
    visualize: bool = False
    Test_hamer: bool = False
    glasses_x_angle_offset: float = 0.0
    start_index: int = 0
    traj_length: int = 1080
    num_samples: int = 1
    guidance_mode: GuidanceMode = "aria_hamer"
    guidance_inner: bool = True
    guidance_post: bool = True
    save_traj: bool = True
    visualize_traj: bool = False
    out_dir: str = "tmp/visualize_hand"
    """Output directory for visualizations."""


# ============================================================================
# Inference (sampling)
# ============================================================================

def inference_and_visualize(
    denoiser_network,
    train_batch,
    device,
    visualized: bool = True,
    start_frame: int = 0,
    training_mode: str = "diffusion",
    num_flow_steps: int = 50,
):
    """Run denoising sampling and optionally visualize.

    Returns:
        (traj, pose_2d_final): HandDenoiseTraj and optional (B, T, 21, 2) 2D joints.
    """
    if training_mode == "diffusion":
        noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
            device=device,
        )
        alpha_bar_t = noise_constants.alpha_bar_t
        alpha_t = noise_constants.alpha_t
    else:
        alpha_bar_t = None
        alpha_t = None

    batch, seq_len, _ = train_batch.mano_pose.shape
    seq_len = seq_len - start_frame

    train_batch.to(device)
    using_mat = denoiser_network.config.using_mat
    using_img_feat = denoiser_network.config.using_img_feat

    x_0_packed = hand_network.HandDenoiseTraj(
        mano_betas=train_batch.mano_betas.unsqueeze(1).expand((batch, seq_len, 10)),
        mano_poses=train_batch.mano_pose[:, start_frame:, 3:48].reshape(batch, seq_len, 15, 3),
        global_orientation=train_batch.mano_pose[:, start_frame:, 0:3],
        global_translation=train_batch.mano_pose[:, start_frame:, 48:],
        mano_side=train_batch.mano_side.unsqueeze(1).expand(batch, seq_len, -1),
    )
    rel_palm_pose = train_batch.mano_pose[:, start_frame:, 48:].to(device)
    cond_feat = train_batch.img_feature.to(device) if using_img_feat else None

    x_t_packed = torch.randn((batch, seq_len, denoiser_network.get_d_state()), device=device)
    x_t_list = []
    start_time = None

    # --- Helper ---
    def _predict_x0_window(x_t_window, t_int, rel_palm_window, conds_window,
                           noisy_joint_2d_window=None):
        x0_pred, pose_2d_pred, _ = denoiser_network.forward(
            x_t_window,
            torch.tensor([t_int], device=device).expand((batch,)),
            rel_palm_pose=rel_palm_window,
            project_output_rotmats=False,
            conds=conds_window,
            img_feat=cond_feat,
            mask=None,
            cond_dropout_keep_mask=torch.ones((batch,), device=device),
            noisy_joint_2d=noisy_joint_2d_window,
        )
        return x0_pred, pose_2d_pred

    def _make_conds(start_t, end_t):
        return hand_network.HandDenoiseTraj(
            mano_betas=x_0_packed.mano_betas[:, start_t:end_t, :],
            mano_poses=x_0_packed.mano_poses[:, start_t:end_t, :, :],
            global_orientation=x_0_packed.global_orientation[:, start_t:end_t, :],
            global_translation=x_0_packed.global_translation[:, start_t:end_t, :],
            mano_side=x_0_packed.mano_side[:, start_t:end_t, :],
        ).to(device)

    def _make_overlap_weights():
        overlap_size = 32
        return (
            torch.from_numpy(
                np.minimum(
                    overlap_size,
                    np.minimum(
                        np.arange(1, seq_len + 1),
                        np.arange(1, seq_len + 1)[::-1],
                    ),
                ) / overlap_size,
            ).to(device).to(torch.float32)
        )

    # ---- Flow matching ----
    if training_mode == "flow_matching":
        max_t = denoiser_network.config.max_t
        ts_frac = linear_ts(num_flow_steps)
        num_joints = 21
        joint_2d_t = torch.randn((batch, seq_len, num_joints, 2), device=device)

        if seq_len > 128:
            window_size, overlap_size = 64, 32
            canonical_ow = _make_overlap_weights()

            for i in tqdm(range(len(ts_frac) - 1)):
                t_curr, t_next = ts_frac[i], ts_frac[i + 1]
                t_int = max(1, int(round(t_curr * max_t)))

                with torch.inference_mode():
                    x_0_packed_pred = torch.zeros_like(x_t_packed)
                    j2d_accum = torch.zeros_like(joint_2d_t)
                    ow = torch.zeros((batch, seq_len, 1), device=device)

                    for s in range(0, seq_len, window_size - overlap_size):
                        e = min(s + window_size, seq_len)
                        ow_slice = canonical_ow[None, :e - s, None]
                        ow[:, s:e, :] += ow_slice

                        temp_x0, temp_j2d = _predict_x0_window(
                            x_t_packed[:, s:e, :], t_int,
                            rel_palm_pose[:, s:e, :], _make_conds(s, e),
                            noisy_joint_2d_window=joint_2d_t[:, s:e, :, :],
                        )
                        x_0_packed_pred[:, s:e, :] += temp_x0 * ow_slice
                        if temp_j2d is not None:
                            j2d_accum[:, s:e, :, :] += temp_j2d * ow_slice[..., None]

                    x_0_packed_pred /= ow
                    j2d_accum /= ow[..., None]
                    x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(
                        x_0_packed_pred, using_mat=using_mat, mano_side=x_0_packed.mano_side
                    ).pack(using_mat=using_mat)

                if start_time is None:
                    start_time = time.time()

                if t_curr > 1e-6 and t_next > 1e-6:
                    x_t_packed = x_0_packed_pred + (t_next / t_curr) * (x_t_packed - x_0_packed_pred)
                    joint_2d_t = j2d_accum + (t_next / t_curr) * (joint_2d_t - j2d_accum)
                else:
                    x_t_packed = x_0_packed_pred
                    joint_2d_t = j2d_accum

                x_t_list.append(
                    hand_network.HandDenoiseTraj.unpack(x_t_packed, using_mat=using_mat, mano_side=x_0_packed.mano_side)
                )
        else:
            for i in tqdm(range(len(ts_frac) - 1)):
                t_curr, t_next = ts_frac[i], ts_frac[i + 1]
                t_int = max(1, int(round(t_curr * max_t)))

                with torch.inference_mode():
                    x_0_packed_pred, pose_2d_pred = _predict_x0_window(
                        x_t_packed, t_int, rel_palm_pose, _make_conds(0, seq_len),
                        noisy_joint_2d_window=joint_2d_t,
                    )
                x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(
                    x_0_packed_pred, using_mat=using_mat, mano_side=x_0_packed.mano_side
                ).pack(using_mat=using_mat)

                if start_time is None:
                    start_time = time.time()

                if t_curr > 1e-6 and t_next > 1e-6:
                    x_t_packed = x_0_packed_pred + (t_next / t_curr) * (x_t_packed - x_0_packed_pred)
                    if pose_2d_pred is not None:
                        joint_2d_t = pose_2d_pred + (t_next / t_curr) * (joint_2d_t - pose_2d_pred)
                else:
                    x_t_packed = x_0_packed_pred
                    if pose_2d_pred is not None:
                        joint_2d_t = pose_2d_pred

                x_t_list.append(
                    hand_network.HandDenoiseTraj.unpack(x_t_packed, using_mat=using_mat, mano_side=x_0_packed.mano_side)
                )

        pose_2d_final = joint_2d_t

    # ---- Diffusion (DDIM) ----
    else:
        pose_2d_final = None

        if seq_len > 128:
            window_size, overlap_size = 64, 32
            canonical_ow = _make_overlap_weights()
            ts = quadratic_ts()

            for i in tqdm(range(len(ts) - 1)):
                t, t_next = ts[i], ts[i + 1]

                with torch.inference_mode():
                    x_0_packed_pred = torch.zeros_like(x_t_packed)
                    p2d_accum = None
                    ow = torch.zeros((batch, seq_len, 1), device=x_t_packed.device)

                    for s in range(0, seq_len, window_size - overlap_size):
                        e = min(s + window_size, seq_len)
                        ow_slice = canonical_ow[None, :e - s, None]
                        ow[:, s:e, :] += ow_slice

                        temp_x0, temp_p2d = _predict_x0_window(
                            x_t_packed[:, s:e, :], t,
                            rel_palm_pose[:, s:e, :], _make_conds(s, e),
                        )
                        x_0_packed_pred[:, s:e, :] += temp_x0 * ow_slice
                        if temp_p2d is not None:
                            if p2d_accum is None:
                                p2d_accum = torch.zeros((batch, seq_len, temp_p2d.shape[2], 2), device=device)
                            p2d_accum[:, s:e, :, :] += temp_p2d * ow_slice[..., None]

                    x_0_packed_pred /= ow
                    if p2d_accum is not None:
                        p2d_accum /= ow[..., None]
                        pose_2d_final = p2d_accum
                    x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(
                        x_0_packed_pred, using_mat=using_mat, mano_side=x_0_packed.mano_side
                    ).pack(using_mat=using_mat)

                sigma_t = torch.cat([
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    ) * 0.8,
                ])
                if start_time is None:
                    start_time = time.time()

                x_t_packed = (
                    torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                    + (
                        torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                        * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                        / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                    )
                    + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
                )
                x_t_list.append(
                    hand_network.HandDenoiseTraj.unpack(x_t_packed, using_mat=using_mat, mano_side=x_0_packed.mano_side)
                )
        else:
            ts = quadratic_ts()
            for i in tqdm(range(len(ts) - 1)):
                t, t_next = ts[i], ts[i + 1]

                with torch.inference_mode():
                    conds = _make_conds(0, seq_len)
                    x_0_packed_pred, pose_2d_step, _ = denoiser_network.forward(
                        x_t_packed,
                        torch.tensor([t], device=device).expand((batch,)),
                        rel_palm_pose=rel_palm_pose,
                        project_output_rotmats=False,
                        conds=conds,
                        img_feat=cond_feat,
                        cond_dropout_keep_mask=torch.ones((batch,), device=device),
                        mask=None,
                    )
                    if pose_2d_step is not None:
                        pose_2d_final = pose_2d_step

                x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(
                    x_0_packed_pred, using_mat=using_mat, mano_side=x_0_packed.mano_side
                ).pack(using_mat=using_mat)

                sigma_t = torch.cat([
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    ) * 0.8,
                ])
                if start_time is None:
                    start_time = time.time()

                x_t_packed = (
                    torch.sqrt(alpha_bar_t[t_next]) * x_0_packed_pred
                    + (
                        torch.sqrt(1 - alpha_bar_t[t_next] - sigma_t[t] ** 2)
                        * (x_t_packed - torch.sqrt(alpha_bar_t[t]) * x_0_packed_pred)
                        / torch.sqrt(1 - alpha_bar_t[t] + 1e-1)
                    )
                    + sigma_t[t] * torch.randn(x_0_packed_pred.shape, device=device)
                )
                x_t_list.append(
                    hand_network.HandDenoiseTraj.unpack(x_t_packed, using_mat=using_mat, mano_side=x_0_packed.mano_side)
                )

    traj = x_t_list[-1]

    # Use GT global trans/orientation.
    traj.global_translation = x_0_packed.global_translation.to(device)
    traj.global_orientation = x_0_packed.global_orientation.to(device)

    if visualized:
        visualize_mesh_in_rgb(
            [x_0_packed, traj],
            intrinsics=train_batch.intrinsics.squeeze(0),
            rgb_frames=train_batch.rgb_frames.squeeze(0),
            subseq_len=seq_len,
            start_frame=start_frame,
            out_dir="tmp/visualize_hand",
        )
        # 2D pose visualization.
        if pose_2d_final is not None and hasattr(train_batch, 'rgb_frames'):
            gt_j2d = train_batch.joint_2d.squeeze(0) if hasattr(train_batch, 'joint_2d') and train_batch.joint_2d is not None else None
            visualize_2d_pose(
                rgb_frames=train_batch.rgb_frames.squeeze(0),
                pred_joints_2d=pose_2d_final.squeeze(0),
                gt_joints_2d=gt_j2d,
                mask=train_batch.mask.squeeze(0) if hasattr(train_batch, 'mask') else None,
                out_dir="tmp/visualize_2d",
            )

    return traj, pose_2d_final


# ============================================================================
# Metric computation
# ============================================================================

def compute_all_metrics(
    pred_traj,
    gt_traj,
    pred_pose_2d: torch.Tensor | None,
    gt_joints_3d: torch.Tensor,
    gt_joints_2d: torch.Tensor | None,
    mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Compute all evaluation metrics for a batch.

    Returns dict with keys: MPJPE, PA-MPJPE, PA-MPVPE, MPVPE, accel,
                            2D_joint_err, and per-param MSE.
    """
    metrics: dict[str, float] = {}
    B, T = mask.shape

    # --- 3D joints ---
    _, _, joints_pred = pred_traj.apply_to_hand()
    _, _, joints_gt_from_traj = gt_traj.apply_to_hand()

    # MPJPE (mm)
    err_3d = torch.sqrt(((gt_joints_3d.to(device) - joints_pred) ** 2).sum(dim=-1)).mean(dim=-1).cpu()
    mpjpe = (torch.where(mask.cpu(), err_3d, torch.zeros_like(err_3d)).sum(dim=-1) / mask.sum(dim=-1)).sum().item() * 1000
    metrics["MPJPE"] = mpjpe / B

    # PA-MPJPE (mm)
    metrics["PA-MPJPE"] = compute_pa_mpjpe(joints_pred, gt_joints_3d.to(device), mask)

    # --- Vertices ---
    verts_pred, faces_pred, _ = pred_traj.apply_to_hand()
    verts_gt, _, _ = gt_traj.apply_to_hand()

    # MPVPE (mm)
    err_v = torch.sqrt(((verts_gt.to(device) - verts_pred) ** 2).sum(dim=-1)).mean(dim=-1).cpu()
    mpvpe = (torch.where(mask.cpu(), err_v, torch.zeros_like(err_v)).sum(dim=-1) / mask.sum(dim=-1)).sum().item() * 1000
    metrics["MPVPE"] = mpvpe / B

    # PA-MPVPE (mm)
    metrics["PA-MPVPE"] = compute_pa_mpvpe(verts_pred, verts_gt.to(device), mask)

    # --- Acceleration error ---
    accel_gt = gt_joints_3d.to(device)[:, 2:, :] - 2 * gt_joints_3d.to(device)[:, 1:-1, :] + gt_joints_3d.to(device)[:, :-2, :]
    accel_pred = joints_pred[:, 2:, :] - 2 * joints_pred[:, 1:-1, :] + joints_pred[:, :-2, :]
    accel_err = torch.sqrt(((accel_gt - accel_pred) ** 2).sum(dim=-1)).mean(dim=-1).cpu()
    mask_accel = mask[:, 1:-1]
    accel_val = (torch.where(mask_accel.cpu(), accel_err, torch.zeros_like(accel_err)).sum(dim=-1) / mask_accel.sum(dim=-1)).sum().item() * 1000
    metrics["accel_score"] = accel_val / B

    # --- 2D joint error (px) ---
    if pred_pose_2d is not None and gt_joints_2d is not None:
        gt_2d = gt_joints_2d.to(pred_pose_2d.device)
        err_2d = torch.sqrt(((gt_2d - pred_pose_2d) ** 2).sum(dim=-1)).mean(dim=-1).cpu()
        val_2d = (torch.where(mask.cpu(), err_2d, torch.zeros_like(err_2d)).sum(dim=-1) / mask.sum(dim=-1)).sum().item()
        metrics["2D_joint_err"] = val_2d / B
    else:
        metrics["2D_joint_err"] = 0.0

    # --- Per-parameter MSE ---
    for var in ["mano_betas", "mano_poses", "global_orientation", "global_translation"]:
        err = ((getattr(gt_traj, var) - getattr(pred_traj, var).cpu()) ** 2).sum(dim=-1)
        if len(err.shape) > 2:
            err = err.sum(dim=-1)
        val = (torch.where(mask.cpu(), err, torch.zeros_like(err)).sum(dim=-1) / mask.sum(dim=-1)).sum().item()
        metrics[f"MSE_{var}"] = val / B

    return metrics


# ============================================================================
# Main
# ============================================================================

def main(args: Args) -> None:
    device = torch.device("cuda")
    dataset = HandHdf5Dataset(
        split="test", dataset_name='ho3d', vis=True,
        use_feature="cls_token", subseq_len=128,
    )
    print("Dataset size:", len(dataset))
    visualized = args.visualize
    test_hamer = args.Test_hamer

    if test_hamer:
        # --- HaMeR baseline evaluation (unchanged logic) ---
        N = len(dataset)
        errors = {"mano_poses": 0, "mano_betas": 0, "global_orientation": 0, "global_translation": 0}
        joint_errors = 0
        from hamer_helper import HamerHelper
        hamer_helper = HamerHelper()

        for i in range(N):
            print(f"processed at index {i}")
            sample = [dataset.__getitem__(i, resize=None).to(device)]
            keys = vars(sample[0]).keys()
            gt = type(sample[0])(**{k: torch.stack([getattr(b, k) for b in sample]) for k in keys})
            skip_cam = -1
            subseq_len = gt.mano_pose.shape[1]
            hamer_out = []
            mask = gt.mask
            subseq_len = min(mask.sum().cpu().numpy(), subseq_len)

            for j in range(subseq_len):
                hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
                    sample[0].rgb_frames[j].cpu().numpy().astype(np.uint8),
                    focal_length=sample[0].intrinsics[0].cpu().item(),
                )
                is_right = gt.mano_side[0].cpu().numpy() != 0
                detection = hamer_out_right if is_right else hamer_out_left
                if detection is None:
                    skip_cam = j
                    break
                hamer_out.append({
                    "verts": detection["verts"],
                    "keypoints_3d": detection["keypoints_3d"],
                    "mano_poses": detection["mano_hand_pose"],
                    "mano_betas": detection["mano_hand_betas"],
                    "global_orientation": detection["mano_hand_global_orient"],
                    "global_translation": detection["global_translation"],
                })

            if skip_cam >= 0:
                print(f"Encounter None hamer output at pos {j} and index {i}")
                continue

            hamer_keys = hamer_out[0].keys()
            hamer_out = {k: np.stack([b[k] for b in hamer_out], axis=0) for k in hamer_keys}
            hamer_out["mano_poses"] = hand_network.SO3.from_matrix(
                torch.from_numpy(hamer_out["mano_poses"])
            ).log().reshape(-1, 45).numpy()
            hamer_out["global_orientation"] = hand_network.SO3.from_matrix(
                torch.from_numpy(hamer_out["global_orientation"])
            ).log().reshape(-1, 3).numpy()
            hamer_kp3d = torch.from_numpy(hamer_out["keypoints_3d"]).to(device).permute(1, 0, 2, 3)

            joint_errors += torch.sqrt(
                ((gt.mano_joint_3d - hamer_kp3d) ** 2).sum(dim=-1)
            ).mean(dim=-1).sum().cpu().numpy()
            for var in errors.keys():
                gt_val = getattr(hand_network.HandDenoiseTraj(
                    mano_betas=gt.mano_betas.unsqueeze(1).expand((1, subseq_len, 10)),
                    mano_poses=gt.mano_pose[:, :, 3:48].reshape(1, subseq_len, 15, 3),
                    global_orientation=gt.mano_pose[:, :, 0:3],
                    global_translation=gt.mano_pose[:, :, 48:],
                    mano_side=gt.mano_side.unsqueeze(1).expand((1, subseq_len, -1)),
                ), var)
                errors[var] += ((gt_val.cpu()) ** 2).sum().numpy()

            for var in errors.keys():
                print(f"  {var} MSE: {errors[var] / (i + 1):.6f}")
            print(f"  3D Joint error: {joint_errors / (i + 1):.4f}")

        print("\n=== Final HaMeR Results ===")
        for var in errors.keys():
            print(f"  {var} MSE: {errors[var] / N:.6f}")
        print(f"  3D Joint error: {joint_errors / N:.4f}")

    else:
        # --- Our model evaluation ---
        denoiser_network = load_hand_denoiser(args.checkpoint_dir).to(device)
        print(f"Loaded checkpoint from: {args.checkpoint_dir}")

        if visualized:
            # --- Visualization mode ---
            data_id = random.randint(1, len(dataset))
            sample = dataset.__getitem__(data_id, resize=None)
            keys = vars(sample).keys()
            train_batch = type(sample)(**{k: torch.stack([getattr(sample, k)]) for k in keys})
            subseq_len = train_batch.mano_pose.shape[1]

            pred, pred_pose_2d = inference_and_visualize(
                denoiser_network, train_batch, device, visualized=True,
                training_mode=args.training_mode, num_flow_steps=args.num_flow_steps,
            )

            # Build GT traj for metrics.
            gt_traj = hand_network.HandDenoiseTraj(
                mano_betas=train_batch.mano_betas.unsqueeze(1).expand((1, subseq_len, 10)),
                mano_poses=train_batch.mano_pose[:, :, 3:48].reshape(1, subseq_len, 15, 3),
                global_orientation=train_batch.mano_pose[:, :, 0:3],
                global_translation=train_batch.mano_pose[:, :, 48:],
                mano_side=train_batch.mano_side.unsqueeze(1).expand(1, subseq_len, -1),
            )
            gt_j2d = train_batch.joint_2d if hasattr(train_batch, 'joint_2d') else None
            metrics = compute_all_metrics(
                pred, gt_traj, pred_pose_2d,
                train_batch.mano_joint_3d, gt_j2d,
                train_batch.mask, device,
            )
            print(f"\n=== Metrics (sample {data_id}) ===")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # Multi-sample visualization.
            pred_list = [gt_traj]
            for _ in range(5):
                p, _ = inference_and_visualize(
                    denoiser_network, train_batch, device, visualized=False,
                    training_mode=args.training_mode, num_flow_steps=args.num_flow_steps,
                )
                pred_list.append(p)
            visualize_mesh_in_rgb(
                pred_list,
                intrinsics=train_batch.intrinsics.squeeze(0),
                rgb_frames=train_batch.rgb_frames.squeeze(0),
                subseq_len=train_batch.mask.sum().cpu().numpy(),
                out_dir=args.out_dir,
            )

        else:
            # --- Quantitative evaluation ---
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=256, shuffle=False, num_workers=2,
                pin_memory=True, collate_fn=collate_dataclass, drop_last=False,
            )
            agg_metrics: dict[str, float] = {}
            total_size = 0

            for train_batch in tqdm(dataloader, desc="Evaluating"):
                B = train_batch.mano_betas.shape[0]
                total_size += B
                seq_len = train_batch.mano_pose.shape[1]

                pred, pred_pose_2d = inference_and_visualize(
                    denoiser_network, train_batch, device, visualized=False,
                    training_mode=args.training_mode, num_flow_steps=args.num_flow_steps,
                )
                gt_traj = hand_network.HandDenoiseTraj(
                    mano_betas=train_batch.mano_betas.unsqueeze(1).expand((B, seq_len, 10)),
                    mano_poses=train_batch.mano_pose[:, :, 3:48].reshape(B, seq_len, 15, 3),
                    global_orientation=train_batch.mano_pose[:, :, 0:3],
                    global_translation=train_batch.mano_pose[:, :, 48:],
                    mano_side=train_batch.mano_side.unsqueeze(1).expand((B, seq_len, -1)),
                )
                gt_j2d = train_batch.joint_2d if hasattr(train_batch, 'joint_2d') else None
                batch_metrics = compute_all_metrics(
                    pred, gt_traj, pred_pose_2d,
                    train_batch.mano_joint_3d, gt_j2d,
                    train_batch.mask, device,
                )
                for k, v in batch_metrics.items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v * B

            print("\n=== Evaluation Results ===")
            for k, v in agg_metrics.items():
                print(f"  {k}: {v / total_size:.4f}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
