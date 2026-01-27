from __future__ import annotations

import dataclasses
import time
from pathlib import Path
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
import viser
import yaml
import random
import os
#from egoallo.transforms import SE3, SO3
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.data.dataclass import collate_dataclass
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.sampling import CosineNoiseScheduleConstants
from egoallo.inference_utils import load_hand_denoiser
from egoallo.data.dataclass import HandTrainingData
from hamer.utils.mesh_renderer import create_raymond_lights
from egoallo.data.hand_data import HandHdf5Dataset
from src.egoallo import network, hand_network
# from PIL import Image
import time    
from manopth.manolayer import ManoLayer
import pyrender
import trimesh
import cv2
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm

def quadratic_ts() -> np.ndarray:
    """DDIM sampling schedule."""
    end_step = 0
    start_step = 1000
    x = np.arange(end_step, int(np.sqrt(start_step))) ** 2
    x[-1] = start_step
    return x[::-1]

def render_joint(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    # camera_center = [render_res[1] / 2.0, render_res[0] / 2.0]
    camera = pyrender.IntrinsicsCamera(
        fx= intrinsics[0],
        fy= intrinsics[1],
        cx= intrinsics[2],
        cy= intrinsics[3],
        zfar=1e12,
        znear=0.001,
    )

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    # Create camera node and add it to pyRender scene
    camera_pose = np.eye(4)
    camera_pose[1:3, :] *= -1  # flip the y and z axes to match opengl
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)  # type: ignore
    mask = color[..., -1] > 0
    return color[..., :3], rend_depth, mask
# Function to create a natural skin-like texture
import numpy as np
import open3d as o3d
import cv2

def create_skin_texture(size=(512, 512)):
    base_color = np.array([200, 150, 130], dtype=np.uint8)  # Skin base (BGR)
    texture = np.full((size[0], size[1], 3), base_color, dtype=np.uint8)
    noise = np.random.normal(0, 20, size=(size[0], size[1], 3)).astype(np.int16)
    texture = np.clip(texture + noise, 0, 255).astype(np.uint8)
    grad_x = np.linspace(0, 1, size[1])
    grad_y = np.linspace(0, 1, size[0])[:, None]
    gradient = (grad_x + grad_y) / 2
    texture = np.clip(texture * (1 - 0.2 * gradient[..., None]), 0, 255).astype(np.uint8)
    texture = cv2.GaussianBlur(texture, (15, 15), 0)
    return texture

# Safe check for CUDA availability
def is_cuda_available_safe():
    try:
        return o3d.core.cuda.device_count() > 0
    except AttributeError:
        return False

# Improved offscreen render: Use dictionary access for attributes
def render_joint_with_texture(vertices, faces, intrinsics, h, w, texture_image=None):
    cuda_available = is_cuda_available_safe()
    device = o3d.core.Device("CUDA:0") if cuda_available else o3d.core.Device("CPU:0")
    print(f"Using device for render: {device}")  # Debug print
    
    # Create tensor-based TriangleMesh on device
    mesh = o3d.t.geometry.TriangleMesh(device=device)
    
    # Set positions and indices using dictionary
    mesh.vertex["positions"] = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32, device=device)
    mesh.triangle["indices"] = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int64, device=device)
    
    # Create natural texture if none
    if texture_image is None:
        texture_image = create_skin_texture()
    
    # Generate UVs (spherical projection for demo)
    verts_np = np.asarray(mesh.vertex["positions"].cpu().numpy())  # Temp to CPU for np ops
    norms = np.linalg.norm(verts_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    uv = np.zeros((len(verts_np), 2))
    uv[:, 0] = np.arctan2(verts_np[:, 0], verts_np[:, 2]) / (2 * np.pi) + 0.5
    uv[:, 1] = np.arcsin(verts_np[:, 1] / norms[:, 0]) / np.pi + 0.5
    faces_flat = faces.reshape(-1)
    mesh.triangle["uvs"] = o3d.core.Tensor(uv[faces_flat], dtype=o3d.core.Dtype.Float32, device=device)
    
    # Texture as tensor image on device
    texture = o3d.t.geometry.Image(texture_image).to(device)
    #o3d_texture = o3d.cuda.pybind.geometry.Image(texture_image)
    # o3d_texture = o3d_texture.to(device=device)
    # print(f"Texture device after transfer: {o3d_texture.device}")  # Debug print
    
    # Material setup
    material = o3d.visualization.rendering.MaterialRecord()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.shader = "defaultLit"
    if isinstance(texture, o3d.t.geometry.Image):
        legacy_texture = texture.to_legacy()
        material.albedo_img = legacy_texture
    else:
        material.albedo_img = texture
    
    # Compute normals on device
    mesh.compute_vertex_normals()
    
    # Offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width=w, height=h)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black
    renderer.scene.add_geometry("mesh", mesh, material)
    
    # Camera
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    renderer.setup_camera(fx, fy, cx, cy, [0, 0, 0], [0, 0, 1], [0, -1, 0])  # Adjust extrinsic if needed
    
    # Render
    color = renderer.render_to_image()
    depth = renderer.render_to_depth_image()
    
    render_rgb = np.asarray(color)[:, :, :3].astype(np.uint8)  # RGB to BGR if needed: cv2.cvtColor(..., cv2.COLOR_RGB2BGR)
    rend_depth = np.asarray(depth)
    render_mask = (rend_depth > 0).astype(bool)
    
    return render_rgb, rend_depth, render_mask

def visualize_joints_in_rgb(Trajs:List[HandDenoiseTraj],
                            intrinsics:torch.FloatTensor,
                            rgb_frames:torch.FloatTensor,
                            subseq_len: int,
                            out_dir:str = "tmp",
                            fps=10, 
                            start_frame=0,
                            resize=None,
                            ) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vertices_list=[]
    faces_list=[]
    for traj in Trajs:
        vertices, faces,_ = traj.apply_to_hand()
        vertices = vertices.squeeze(0).cpu().numpy()
        faces = faces.cpu().numpy()
        faces_list.append(faces)
        vertices_list.append(vertices)
    intrinsics = intrinsics.cpu().numpy()
    border_color = [255, 0, 0]
    video_len, img_height, img_width, _ = rgb_frames.shape
    if resize is not None:
        width, height = resize
    else:
        width = img_width * 2
        height = img_height
    # 确定视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    
    # 创建视频写入对象
    out = cv2.VideoWriter(out_dir+"/infer.mp4", fourcc, fps, (width, height))
    print("start making the video")
    # for i in range(subseq_len):
    #     image = cv2.cvtColor(rgb_frames[i+start_frame].numpy().astype(np.uint8),cv2.COLOR_RGB2BGR)
    #     composited =  image
    #     for j in range(len(Trajs)):
    #         vertices=vertices_list[j]
    #         faces = faces_list[j]
    #         if len(intrinsics.shape)==2:
    #             render_rgb, rend_depth, render_mask = render_joint(vertices[i], faces,
    #                                                                 intrinsics[i], h=rgb_frames.shape[1], w=rgb_frames.shape[2])
    #         else:
    #             render_rgb, rend_depth, render_mask = render_joint(vertices[i], faces,
    #                                                                 intrinsics, h=rgb_frames.shape[1], w=rgb_frames.shape[2])
    #         # breakpoint()
    #         border_width = 10
    #         composited_tmp = np.where(
    #             binary_dilation(
    #                 render_mask, np.ones((border_width, border_width), dtype=bool)
    #             )[:, :, None],
    #             np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
    #             image,
    #         )
    #         composited_tmp = np.where(render_mask[:, :, None], render_rgb, image)
    #         composited = np.concatenate([composited,composited_tmp], axis=1)
    #     if resize is not None:
    #         composited = cv2.resize(composited, resize)
    #     out.write(composited)
    # out.release()import trimesh


    # Main code with improvements: Use textured render, handle empty cases better
    height, width = rgb_frames.shape[1], rgb_frames.shape[2]  # From rgb_frames (N, H, W, C)

    image = cv2.cvtColor(rgb_frames[0].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    composited = np.zeros_like(image)  # Black background (or np.ones_like(image)*255 for white)
    out_image = None
    loop_executed = False

    # Collect traj_points for fixed vertex
    fixed_vertex_index = 0  # e.g., wrist
    traj_points = []
    traj_color = (0, 255, 0)  # Green
    traj_thickness = 2
    draw_points = True
    point_radius = 3
    point_color = (255, 0, 0)  # Red
    for i in range(0, subseq_len, 24):
        if len(Trajs) > 1:
            print("len(Trajs) is ", len(Trajs), " only process the first one for traj visualization")
            break
        loop_executed = True
        print("process wrist trajs")
        j = 0
        vertices = vertices_list[j]
        faces = faces_list[j]
        if len(intrinsics.shape) == 2:
            render_rgb, rend_depth, render_mask = render_joint_with_texture(
                vertices[i], faces, intrinsics[i], h=height, w=width
            )
            # Project fixed vertex to 2D
            vertex_3d = vertices[i][fixed_vertex_index]
            fx, fy, cx, cy = intrinsics[i][0], intrinsics[i][1], intrinsics[i][2], intrinsics[i][3]
            if vertex_3d[2] != 0:  # Avoid div by zero
                u = int(fx * vertex_3d[0] / vertex_3d[2] + cx)
                v = int(fy * vertex_3d[1] / vertex_3d[2] + cy)
                traj_points.append((u, v))
        else:
            render_rgb, rend_depth, render_mask = render_joint_with_texture(
                vertices[i], faces, intrinsics, h=height, w=width
            )
            # Project fixed vertex
            vertex_3d = vertices[i][fixed_vertex_index]
            fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
            if vertex_3d[2] != 0:
                u = int(fx * vertex_3d[0] / vertex_3d[2] + cx)
                v = int(fy * vertex_3d[1] / vertex_3d[2] + cy)
                traj_points.append((u, v))
        
        if render_mask.size == 0 or render_rgb.size == 0:
            print(f"Warning: Empty render at frame {i}. Skipping.")
            continue
        
        border_width = 5
        dilated_mask = binary_dilation(render_mask, np.ones((border_width, border_width), dtype=bool))[:, :, None]
        composited = np.where(dilated_mask, np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8), composited)
        composited = np.where(render_mask[:, :, None], render_rgb, composited)
        
        if resize is not None:
            composited = cv2.resize(composited, resize)
        
        out_image = composited
    # Draw trajectory on final composited image
    if out_image is not None and len(traj_points) > 1:
        # Simple lines
        for p in range(1, len(traj_points)):
            cv2.line(out_image, traj_points[p-1], traj_points[p], traj_color, traj_thickness)
        
        # Optional: Smooth spline
        # if len(traj_points) >= 4:
        #     x, y = zip(*traj_points)
        #     tck, u = splprep([x, y], s=0)
        #     new_points = splev(np.linspace(0, 1, 100), tck)
        #     for p in range(1, len(new_points[0])):
        #         pt1 = (int(new_points[0][p-1]), int(new_points[1][p-1]))
        #         pt2 = (int(new_points[0][p]), int(new_points[1][p]))
        #         cv2.line(out_image, pt1, pt2, traj_color, traj_thickness)
        
        if draw_points:
            for pt in traj_points:
                cv2.circle(out_image, pt, point_radius, point_color, -1)

    # Save with checks
    if out_image is not None and out_image.size > 0:
        output_path = os.path.join(out_dir, f'hand_motion_traj.png')
        cv2.imwrite(output_path, out_image)

    for i in range(subseq_len):
        image = cv2.cvtColor(rgb_frames[i+start_frame].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        composited = image.copy()  # 初始化为原图像的拷贝，避免修改原图像
        for j in range(len(Trajs)):
            vertices = vertices_list[j]
            faces = faces_list[j]
            if len(intrinsics.shape) == 2:
                render_rgb, rend_depth, render_mask = render_joint(vertices[i], faces,
                                                                intrinsics[i], h=rgb_frames.shape[1], w=rgb_frames.shape[2])
            else:
                render_rgb, rend_depth, render_mask = render_joint(vertices[i], faces,
                                                                intrinsics, h=rgb_frames.shape[1], w=rgb_frames.shape[2])
            # breakpoint()
            border_width = 5
            # 先添加边框（如果多个traj，可能需要不同颜色或调整逻辑；这里假设共享边框颜色）
            composited = np.where(
                binary_dilation(
                    render_mask, np.ones((border_width, border_width), dtype=bool)
                )[:, :, None],
                np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
                composited,
            )
            # 叠加当前traj的渲染到composited上
            composited = np.where(render_mask[:, :, None], render_rgb, composited)
        composited = np.concatenate([image,composited], axis=1)
        if resize is not None:
            composited = cv2.resize(composited, resize) # data~[N,T,J,3],[J,N*T*3],[J,3,N*T]=mean(var(N*T))
        out.write(composited)
    out.release()
    print(f"visualization saved at : {out_dir}")

def mano_poses2joints_3d(mano_pose: torch.FloatTensor, mano_betas: torch.FloatTensor, mano_side: str) -> torch.FloatTensor:
    """Convert MANO pose to joint 3D positions."""
    mano_betas = mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1)
    assert mano_pose.shape == (64, 51), f"Expected mano_pose shape (1, 51), got {mano_pose.shape}"
    assert mano_betas.shape == (64, 10), f"Expected mano_betas shape (1, 10), got {mano_betas.shape}"
    mano_layer = ManoLayer(
        ncomps=45,
        side=mano_side,
        mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
         use_pca=False,
            flat_hand_mean=True,)
        
    verts, joints = mano_layer(
        mano_pose[:,:48],
        mano_betas,  # (64, 10)
        mano_pose[:,48:51],  # (64, 3)
    )
    joints = joints/1000
    return joints
    
@dataclasses.dataclass
class Args:
    checkpoint_dir: Path = Path("/data-share/L202500064/handtraj/experiments/cfg_train/v1/checkpoints_400000")#Path("/data-share/L202500064/handtraj/experiments/all_data/v1/checkpoints_500000/")# 
    visualize: bool = False
    Test_hamer: bool = False
    glasses_x_angle_offset: float = 0.0
    """Rotate the CPF poses by some X angle."""
    start_index: int = 0
    """Index within the downsampled trajectory to start inference at."""
    traj_length: int = 1080
    """How many timesteps to estimate body motion for."""
    num_samples: int = 1
    """Number of samples to take."""
    guidance_mode: GuidanceMode = "aria_hamer"
    """Which guidance mode to use."""
    guidance_inner: bool = True
    """Whether to apply guidance optimizer between denoising steps. This is
    important if we're doing anything with hands. It can be turned off to speed
    up debugging/experiments, or if we only care about foot skating losses."""
    guidance_post: bool = True
    """Whether to apply guidance optimizer after diffusion sampling."""
    save_traj: bool = True
    """Whether to save the output trajectory, which will be placed under `traj_dir/egoallo_outputs/some_name.npz`."""
    visualize_traj: bool = False
    """Whether to visualize the trajectory after sampling."""

def inference_and_visualize(
        denoiser_network,
        train_batch,
        device,
        visualized=True,
        start_frame=0):
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device,
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t
    batch,seq_len,_ = train_batch.mano_pose.shape
    seq_len = seq_len - start_frame

    train_batch.to(device)
    print(denoiser_network)
    using_mat = denoiser_network.config.using_mat
    using_img_feat = denoiser_network.config.using_img_feat
    print("Using image feature: ", using_img_feat, ", using rot mat:", using_mat)
    x_0_packed = hand_network.HandDenoiseTraj(
        mano_betas=train_batch.mano_betas.unsqueeze(1).expand((batch, seq_len, 10)),
        mano_poses=train_batch.mano_pose[:,start_frame:,3:48].reshape(batch,seq_len,15,3),
        global_orientation=train_batch.mano_pose[:,start_frame:,0:3],
        global_translation=train_batch.mano_pose[:,start_frame:,48:],
        mano_side=train_batch.mano_side.unsqueeze(1).expand(batch, seq_len, -1),
    )
    rel_palm_pose = train_batch.mano_pose[:,start_frame:,48:].to(device)
    if using_img_feat:
        cond_feat = train_batch.img_feature.to(device)
    else:
        cond_feat = None
    x_t_packed = torch.randn((batch, seq_len, denoiser_network.get_d_state()), device=device)
    x_t_list = []
    start_time = None
    if seq_len > 64:
        window_size = 64
        overlap_size = 32
        canonical_overlap_weights = (
            torch.from_numpy(
                np.minimum(
                    # Make this shape /```\
                    overlap_size,
                    np.minimum(
                        # Make this shape: /
                        np.arange(1, seq_len + 1),
                        # Make this shape: \
                        np.arange(1, seq_len + 1)[::-1],
                    ),
                )
                / overlap_size,
            )
            .to(device)
            .to(torch.float32)
        )
        ts = quadratic_ts()

        for i in tqdm(range(len(ts) - 1)):
            print(f"Sampling {i}/{len(ts) - 1}")
            t = ts[i]
            t_next = ts[i + 1]

            with torch.inference_mode():
                # Chop everything into windows.
                x_0_packed_pred = torch.zeros_like(x_t_packed)
                overlap_weights = torch.zeros((batch, seq_len, 1), device=x_t_packed.device)

                # Denoise each window.
                for start_t in range(0, seq_len, window_size - overlap_size):
                    end_t = min(start_t + window_size, seq_len)
                    assert end_t - start_t > 0
                    overlap_weights_slice = canonical_overlap_weights[
                        None, : end_t - start_t, None
                    ]
                    overlap_weights[:, start_t:end_t, :] += overlap_weights_slice
                    conds = hand_network.HandDenoiseTraj(
                                                        mano_betas=x_0_packed.mano_betas[:, start_t:end_t, :],
                                                        mano_poses=x_0_packed.mano_poses[:, start_t:end_t, :,:],
                                                        global_orientation=x_0_packed.global_orientation[:, start_t:end_t, :],
                                                        global_translation=x_0_packed.global_translation[:, start_t:end_t, :],
                                                        mano_side=x_0_packed.mano_side[:, start_t:end_t, :],
                                                        ).to(device)
                    x_0_packed_pred[:, start_t:end_t, :] += (
                        denoiser_network.forward(
                            x_t_packed[:, start_t:end_t, :],
                            torch.tensor([t], device=device).expand((batch,)),
                            rel_palm_pose=rel_palm_pose[:,start_t:end_t, :],
                            project_output_rotmats=False,
                            conds=conds,
                            img_feat=cond_feat,
                            mask=None,
                            cond_dropout_keep_mask = torch.zeros((batch,), device=device),
                        )* overlap_weights_slice
                    )

                # Take the mean for overlapping regions.
                x_0_packed_pred /= overlap_weights

                x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(x_0_packed_pred,using_mat=using_mat,mano_side=x_0_packed.mano_side).pack(using_mat=using_mat)

            if torch.any(torch.isnan(x_0_packed_pred)):
                print("found nan", i)
            sigma_t = torch.cat(
                [
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    )
                    * 0.8,
                ]
            )

            """ if guidance_mode != "off" and guidance_inner:
                x_0_pred, _ = do_guidance_optimization(
                    It's important that we _don't_ use the shifted transforms here.
                    Ts_world_cpf=Ts_world_cpf[1:, :],
                    traj=network.EgoDenoiseTraj.unpack(
                        x_0_packed_pred, include_hands=denoiser_network.config.include_hands
                    ),
                    body_model=body_model,
                    guidance_mode=guidance_mode,
                    phase="inner",
                    
                    hamer_detections=hamer_detections,
                    aria_detections=aria_detections,
                    verbose=guidance_verbose,
                )
                x_0_packed_pred = x_0_pred.pack()
                del x_0_pred
            """
            if start_time is None:
                start_time = time.time()

            # print(sigma_t)
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
                hand_network.HandDenoiseTraj.unpack(x_t_packed,using_mat=using_mat,mano_side=x_0_packed.mano_side)
            )

            assert start_time is not None
            print("RUNTIME (exclude first optimization)", time.time() - start_time)
    else:
        ts = quadratic_ts()
        for i in tqdm(range(len(ts) - 1)):
            print(f"Sampling {i}/{len(ts) - 1}")
            t = ts[i]
            t_next = ts[i + 1]

            with torch.inference_mode():
                # Chop everything into windows.
                x_0_packed_pred = torch.zeros_like(x_t_packed)
                conds = hand_network.HandDenoiseTraj(
                                                    mano_betas=x_0_packed.mano_betas,
                                                    mano_poses=x_0_packed.mano_poses,
                                                    global_orientation=x_0_packed.global_orientation,
                                                    global_translation=x_0_packed.global_translation,
                                                    mano_side=x_0_packed.mano_side,
                                                    ).to(device)
                x_0_packed_pred = denoiser_network.forward(
                                                            x_t_packed,
                                                            torch.tensor([t], device=device).expand((batch,)),
                                                            rel_palm_pose=rel_palm_pose,
                                                            project_output_rotmats=False,
                                                            conds=conds,
                                                            img_feat=cond_feat,
                                                            cond_dropout_keep_mask = torch.zeros((batch,), device=device),
                                                            mask=None,
                                                        )

            x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(x_0_packed_pred,using_mat=using_mat,mano_side=x_0_packed.mano_side).pack(using_mat=using_mat)

            if torch.any(torch.isnan(x_0_packed_pred)):
                print("found nan", i)
            sigma_t = torch.cat(
                [
                    torch.zeros((1,), device=device),
                    torch.sqrt(
                        (1.0 - alpha_bar_t[:-1]) / (1 - alpha_bar_t[1:]) * (1 - alpha_t)
                    )
                    * 0.8,
                ]
            )

            """ if guidance_mode != "off" and guidance_inner:
                x_0_pred, _ = do_guidance_optimization(
                    It's important that we _don't_ use the shifted transforms here.
                    Ts_world_cpf=Ts_world_cpf[1:, :],
                    traj=network.EgoDenoiseTraj.unpack(
                        x_0_packed_pred, include_hands=denoiser_network.config.include_hands
                    ),
                    body_model=body_model,
                    guidance_mode=guidance_mode,
                    phase="inner",
                    
                    hamer_detections=hamer_detections,
                    aria_detections=aria_detections,
                    verbose=guidance_verbose,
                )
                x_0_packed_pred = x_0_pred.pack()
                del x_0_pred
            """
            if start_time is None:
                start_time = time.time()

            # print(sigma_t)
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
                hand_network.HandDenoiseTraj.unpack(x_t_packed,using_mat=using_mat,mano_side=x_0_packed.mano_side)
            )

            assert start_time is not None
            print("RUNTIME (exclude first optimization)", time.time() - start_time)
    traj = x_t_list[-1]
    
    ## Assume we have gt global trans/orientation
    traj.global_translation = x_0_packed.global_translation.to(device)
    traj.global_orientation = x_0_packed.global_orientation.to(device)
    if visualized == True:
        visualize_joints_in_rgb([x_0_packed,traj],
                                intrinsics=train_batch.intrinsics.squeeze(0),
                                rgb_frames=train_batch.rgb_frames.squeeze(0),
                                subseq_len = seq_len,
                                start_frame=start_frame,
                                out_dir = "tmp/visualize_hand")
        return traj
    else:
        return traj



def main(args: Args) -> None:
    device = torch.device("cuda")
    dataset = HandHdf5Dataset(split="test",dataset_name = 'ho3d',vis=False,subseq_len=64)#(dataset_name = 'ho3d', vis=True)# 
    print("Dataset size:", len(dataset))
    visualized = args.visualize
    test_hamer = args.Test_hamer
    if test_hamer == True:
        N = len(dataset)
        errors ={"mano_poses": 0, "mano_betas": 0, "global_orientation":0,"global_translation":0}
        joint_errors = 0
        from hamer_helper import HamerHelper
        hamer_helper = HamerHelper()   
        for i in range(N):

            print("processed at index ", i)
            sample = [dataset.__getitem__(i,resize=None).to(device)] 
            keys = vars(sample[0]).keys()
            gt = type(sample[0])(**{k: torch.stack([getattr(b, k) for b in sample]) for k in keys})
            skip_cam = -1
            subseq_len = gt.mano_pose.shape[1]
            hamer_out = []
            mask = gt.mask
            subseq_len = min(mask.sum().cpu().numpy(),subseq_len)
            print("Meet index ",i," video len ", subseq_len)
            for j in range(subseq_len):
                feat = hamer_helper.get_img_feats(
                sample[0].rgb_frames[j].cpu().numpy().astype(np.uint8),
                mano_side = (gt.mano_side.cpu().numpy() == 1)
                )
                hamer_out_frame = {}            
                hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
                    sample[0].rgb_frames[j].cpu().numpy().astype(np.uint8),
                    focal_length=sample[0].intrinsics[0].cpu().item(),
                )
                if gt.mano_side[0].cpu().numpy() != 0:
                    if hamer_out_right is None:
                        skip_cam = j
                        break
                    else:
                        out_dict = {
                                    "verts": hamer_out_right["verts"],
                                    "keypoints_3d": hamer_out_right["keypoints_3d"],
                                    "mano_poses": hamer_out_right["mano_hand_pose"],
                                    "mano_betas": hamer_out_right["mano_hand_betas"],
                                    "global_orientation": hamer_out_right["mano_hand_global_orient"],
                                    "global_translation": hamer_out_right["global_translation"],
                        }
                        hamer_out.append(out_dict)
                else:
                    if hamer_out_left is None:
                        skip_cam = j
                        break
                    else:
                        out_dict = {
                                    "verts": hamer_out_left["verts"],
                                    "keypoints_3d": hamer_out_left["keypoints_3d"],
                                    "mano_poses": hamer_out_left["mano_hand_pose"],
                                    "mano_betas": hamer_out_left["mano_hand_betas"],
                                    "global_orientation": hamer_out_left["mano_hand_global_orient"],
                                    "global_translation": hamer_out_left["global_translation"],
                        }
                        hamer_out.append(out_dict)          
            if skip_cam >=0:
                print("Encounter None hamer output at pos ",j," and index : ",i)
                continue
            keys = hamer_out[0].keys()
            hamer_out = type(hamer_out[0])(**{k: np.stack([b[k] for b in hamer_out],axis=0) for k in keys})
            gt_values={}
            gt_values["joint_3d"] = gt.mano_joint_3d
            hamer_out["mano_poses"] = hand_network.SO3.from_matrix(torch.from_numpy(hamer_out["mano_poses"])).log().reshape(-1,45).numpy()
            hamer_out["global_orientation"] = hand_network.SO3.from_matrix(torch.from_numpy(hamer_out["global_orientation"])).log().reshape(-1,3).numpy()
            hamer_out["keypoints_3d"] = torch.from_numpy(hamer_out["keypoints_3d"]).to(device).permute(1,0,2,3)
            print(hamer_out["global_translation"].shape)
            x_0_packed = hand_network.HandDenoiseTraj(
                                                    mano_betas=gt.mano_betas.unsqueeze(1).expand((1, subseq_len, 10)),
                                                    mano_poses=gt.mano_pose[:,:,3:48].reshape(1,subseq_len,15,3),
                                                    global_orientation=gt.mano_pose[:,:,0:3],
                                                    global_translation=gt.mano_pose[:,:,48:],
                                                    mano_side=gt.mano_side.unsqueeze(1).expand((1, subseq_len, -1)),
                                                )
            traj = hand_network.HandDenoiseTraj(
                                                    mano_betas=torch.from_numpy(hamer_out["mano_betas"]).permute((1, 0, 2)),
                                                    mano_poses=torch.from_numpy(hamer_out["mano_poses"]).reshape(1,subseq_len,15,3),
                                                    global_orientation=torch.from_numpy(hamer_out['global_orientation']).unsqueeze(0),
                                                    global_translation=torch.from_numpy(hamer_out["global_translation"]).squeeze(1).permute(1,0,2),
                                                    mano_side=gt.mano_side.unsqueeze(1).expand((1, subseq_len, -1)),
                                                )
            joint_errors+= torch.sqrt(((gt_values["joint_3d"]-hamer_out["keypoints_3d"]) ** 2).sum(dim=-1)).mean(dim=-1).sum().cpu().numpy()
            for var in errors.keys():
                print(var,"gt shape",getattr(x_0_packed,var).shape," and pred shape:", getattr(traj,var).shape)
                errors[var]+=((getattr(x_0_packed,var).cpu()-getattr(traj,var))**2).sum().numpy()
            for var in errors.keys():
                print("var: ",var," MSE error is:",errors[var]/(i+1))
            print("3D Joint mean error per video: ",joint_errors/(i+1))
        for var in errors.keys():
                print("var: ",var," MSE error is:",errors[var]/N)
        print("3D Joint mean error per video: ",joint_errors/N)
    else: 
        denoiser_network = load_hand_denoiser(args.checkpoint_dir).to(device)
        print("Success when load ckpt from:", args.checkpoint_dir)
        train_batch = []
        rand_id = 0 
        if visualized == True:
            num_samples = 1
            for i in range(num_samples):
                data_id = random.randint(1,len(dataset)) # 2023 for arctic
                rand_id = data_id
                sample = dataset.__getitem__(data_id,resize=None)
                train_batch.append(sample)
            keys = vars(train_batch[0]).keys()
            train_batch = type(train_batch[0])(**{k: torch.stack([getattr(b, k) for b in train_batch]) for k in keys})
            subseq_len = train_batch.mano_pose.shape[1]
            pred = inference_and_visualize(denoiser_network,train_batch,device,visualized)
            _,_,joints_pred = pred.apply_to_hand()
            x_0_packed = hand_network.HandDenoiseTraj(
                                                            mano_betas=train_batch.mano_betas.unsqueeze(1).expand((1, subseq_len, 10)),
                                                            mano_poses=train_batch.mano_pose[:,:,3:48].reshape(1,subseq_len,15,3),
                                                            global_orientation=train_batch.mano_pose[:,:,0:3],
                                                            global_translation=train_batch.mano_pose[:,:,48:],
                                                            mano_side=train_batch.mano_side.unsqueeze(1).expand(1,subseq_len, -1),
                                                    )
            _,_,joints_gt = x_0_packed.apply_to_hand()
            loss_mask = train_batch.mask
            #train_batch.mano_joint_3d.to(device)
            print(train_batch.mano_joint_3d.to(device))
            error_joints =  torch.sqrt(((joints_gt.cpu() - joints_pred.cpu())**2).sum(dim=-1)).mean(dim=-1).cpu()
            error_joints = (torch.where(loss_mask,error_joints,torch.zeros_like(error_joints)).sum()/loss_mask.sum()).numpy()*1000
            print("Joint error is ", error_joints)
            print("take the id",rand_id,"as the test sample")
            pred_list = []#[x_0_packed]
            print("reach visualization here!")
            for i in range(5):
                pred = inference_and_visualize(denoiser_network,train_batch,device,False)
                pred_list.append(pred)
            visualize_joints_in_rgb(pred_list,
                                    intrinsics=train_batch.intrinsics.squeeze(0),
                                    rgb_frames=train_batch.rgb_frames.squeeze(0),
                                    subseq_len = train_batch.mask.sum().cpu().numpy(),
                                    out_dir = "tmp/visualize_hand")
        else:
            errors={"mano_betas":0,"mano_poses":0,"global_orientation":0,"global_translation":0,"3D_joints":0,"accel_score":0}
            total_size = 0
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=256,
                                                    shuffle=False,
                                                    num_workers=2,
                                                    pin_memory=True,
                                                    collate_fn=collate_dataclass,
                                                    drop_last=False)
            var_batch = []
            mean_batch=[]
            gt_joints = []
            for train_batch in dataloader:
                total_size += train_batch.mano_betas.shape[0]
                joints_list = []
                for i in range(1):
                    pred = inference_and_visualize(denoiser_network,train_batch,device,visualized)
                    _,_,joints_pred = pred.apply_to_hand()
                    joints_list.append(joints_pred) # BxTxJx3
                    for j in range(train_batch.mano_joint_3d.shape[0]):
                        gt_joints.append(joints_pred.to(device)[j].unsqueeze(0))
                variance,mean = cal_variance(joints_list) 
                var_batch.append(variance.cpu().numpy())
                mean_batch.append(mean.cpu().numpy())
                batch,seq_len,_ = train_batch.mano_pose.shape
                loss_mask = train_batch.mask
                x_0_packed = hand_network.HandDenoiseTraj(
                                                            mano_betas=train_batch.mano_betas.unsqueeze(1).expand((batch, seq_len, 10)),
                                                            mano_poses=train_batch.mano_pose[:,:,3:48].reshape(batch,seq_len,15,3),
                                                            global_orientation=train_batch.mano_pose[:,:,0:3],
                                                            global_translation=train_batch.mano_pose[:,:,48:],
                                                            mano_side=train_batch.mano_side.unsqueeze(1).expand((batch, seq_len, -1)),
                                                        )
                _,_,joints_pred = pred.apply_to_hand()
                for var in errors.keys():
                    if var  != "3D_joints" and var != "accel_score":
                        err = ((getattr(x_0_packed,var)-getattr(pred,var).cpu())**2).sum(dim=-1)
                        if len(err.shape)>2:
                            err = err.sum(dim=-1)
                        errors[var]+=(torch.where(loss_mask.cpu(),err,torch.zeros_like(err)).sum(dim=-1)/loss_mask.sum(dim=-1)).sum().numpy()
                    else:#MPJPE & ACCEL score
                        err = torch.sqrt(((train_batch.mano_joint_3d.to(device) - joints_pred)**2).sum(dim=-1)).mean(dim=-1).cpu()
                        accel_gt = train_batch.mano_joint_3d.to(device)[:,2:,:]-2*train_batch.mano_joint_3d.to(device)[:,1:-1,:]+train_batch.mano_joint_3d.to(device)[:,:-2,:]
                        accel_pred = joints_pred[:,2:,:]-2*joints_pred[:,1:-1,:]+joints_pred[:,:-2,:]
                        accel_err = torch.sqrt(((accel_gt - accel_pred)**2).sum(dim=-1)).mean(dim=-1).cpu()
                        errors["accel_score"]+=(torch.where(loss_mask[:,1:-1].cpu(),accel_err,torch.zeros_like(accel_err)).sum(dim=-1)/loss_mask[:,1:-1].sum(dim=-1)).sum().numpy() * 1000
                        errors[var]+=(torch.where(loss_mask.cpu(),err,torch.zeros_like(err)).sum(dim=-1)/loss_mask.sum(dim=-1)).sum().numpy() * 1000
            for var in errors.keys():
                print("var: ",var," MSE error is:",errors[var]/total_size)
            var_batch = np.stack(var_batch,axis=0) # N x J
            mean_batch = np.stack(mean_batch,axis=0) # N x J
            print(var_batch.shape)
            mean_variance = np.mean(var_batch,axis=0) # J
            final_mean = np.mean(mean_batch,axis=0) # J
            print(mean_variance)
            mean_variance,final_mean= cal_variance(gt_joints)
            mean_variance = mean_variance.cpu().numpy()
            final_mean = final_mean.cpu().numpy()
            plot_joint_variance_curve( mean_variance)
            plot_joint_variance_curve(final_mean, title="Joint Mean Curve", line_color='green',name='mean')


def cal_variance(joints_list):
    """joints_list: list of BxTxJx3"""
    batch,seq_len,num_joints,_ = joints_list[0].shape
    sample_num = len(joints_list)
    stacked_joints = torch.stack(joints_list, dim=0).permute(3,4,1,2,0)# [J,3,B,T,S]
    stacked_joints_var = torch.var(stacked_joints, dim=-1).reshape(num_joints,-1)  # [J,3*B*T]
    stacked_joint_mean = torch.mean(stacked_joints.reshape(num_joints,-1), dim=-1)  # [J,3*B*T]
    variance = stacked_joints_var.mean(dim=1)  # [J]
    return variance,stacked_joint_mean
    data #[N,T,J,3]
    data_t = data[:,t,:,:] #[N,J,3]
    mask = torch.any(data_t != 0, dim=(0, 2))  # shape: (J,), bool tensor
    filtered_data = data_t[:, mask, :]

# 过滤 data
filtered_data = data[:, mask, :]
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_joint_variance_curve( vars,
                               #means,
                               title="Joint Variance Curve",
                               figsize=(12, 6),
                               line_color='blue',
                               joint_names=None,
                               fill_alpha=0.2,
                               show_std=True,
                               name="var"):
    """
    绘制关节方差曲线图
    
    Args:
        joints: shape=(batch, time, joint_num, 3)
        joint_names: 关节名称列表，可选
        title: 图表标题
        figsize: 图表大小
        line_color: 曲线颜色
        fill_alpha: 填充区域的透明度
        show_std: 是否显示标准差区域
    """
    joint_num = vars.shape[0]
    # 如果没有提供关节名称，使用数字索引
    if joint_names is None:
        joint_names = [f"J{i}" for i in range(joint_num)]
    
    # 计算每个关节的方差和标准差
    joint_variances = vars
    # joint_std = means
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # x轴位置
    x_positions = np.arange(joint_num)
    
    # 绘制方差曲线
    ax.plot(x_positions, joint_variances, 
            marker='o', markersize=8, linewidth=3, 
            color=line_color, label=name)
    
    #如果需要，添加标准差区域
    # if show_std:
    #     ax.fill_between(x_positions, 
    #                     joint_variances - joint_std,
    #                     joint_variances + joint_std,
    #                     alpha=fill_alpha, color=line_color,
    #                     label='± Std Dev')
    
    # 设置坐标轴
    ax.set_xlabel('Joint Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 添加图例
    ax.legend(loc='best')
    
    # 在每个点上添加方差值
    for i, v in enumerate(joint_variances):
        if v<1e-6:
            ax.text(i, v * 1.05, f'{v:.8f}', 
                    ha='center', fontsize=9, fontweight='bold')
        else:
            ax.text(i, v * 1.001, f'{v:.8f}', 
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('./tmp/'+name+'.png', dpi=300, bbox_inches='tight')
    return 0
if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))

