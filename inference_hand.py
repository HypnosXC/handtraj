from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import viser
import yaml

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
    load_denoiser,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.data.dataclass import HandTrainingData
from hamer.utils.mesh_renderer import create_raymond_lights
# from PIL import Image
import time    
from manopth.manolayer import ManoLayer
import pyrender
import trimesh
import cv2
from scipy.ndimage import binary_dilation
mano_side_str=['left','right']

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

def get_vertices_faces(mano_side:torch.FloatTensor,
                       mano_betas:torch.FloatTensor,
                       mano_poses:torch.FloatTensor,
                       global_orientation:torch.FloatTensor,
                       camera_pose:torch.FloatTensor) -> tuple[torch.Tensor, torch.Tensor]:
        mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side=mano_side_str[(int)(mano_side.numpy())],
            mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        )
        vertices, joints = mano_layer(
            torch.cat((global_orientation,mano_pose),dim=-1),
            mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
            camera_pose
        )
        vertices = vertices/ 1000  # Convert to meters
        faces_m = mano_layer.th_faces
        return vertices, faces_m

def visualize_joints_in_rgb(mano_side:torch.FloatTensor,
                            mano_betas:torch.FloatTensor,
                            mano_poses:torch.FloatTensor,
                            global_orientation:torch.FloatTensor,
                            camera_pose:torch.FloatTensor,
                            intrinsics:torch.FloatTensor,
                            rgb_frames:torch.FloatTensor,
                            out_dir:str = "tmp") -> None:
    os.makedirs(out_dir, exist_ok=True)
    vertices, faces = get_vertices_faces(mano_side,
                                         mano_betas,
                                         mano_poses,
                                         global_orientation,
                                         camera_pose)
    intrinsics = intrinsics.numpy()
    border_color = [255, 0, 0]
    for i in range(self._subseq_len):
        image = sample.rgb_frames[i].numpy().astype(np.uint8)
        render_rgb, rend_depth, render_mask = render_joint(vertices[i].numpy(), faces.numpy(),
                                                            intrinsics, h=rgb_frames.shape[1], w=rgb_frames.shape[2])
        # breakpoint()
        border_width = 10
        composited = np.where(
            binary_dilation(
                render_mask, np.ones((border_width, border_width), dtype=bool)
            )[:, :, None],
            np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
            image,
        )
        composited = np.where(render_mask[:, :, None], render_rgb, image)
        composited = np.concatenate([image, composited], axis=1)
        iio.imwrite(os.path.join(out_dir, f"{i:03d}.jpg"), composited)

def mano_poses2joints_3d(mano_pose: torch.FloatTensor, mano_betas: torch.FloatTensor, mano_side: str) -> torch.FloatTensor:
    """Convert MANO pose to joint 3D positions."""
    mano_betas = mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1)
    assert mano_pose.shape == (64, 51), f"Expected mano_pose shape (1, 51), got {mano_pose.shape}"
    assert mano_betas.shape == (64, 10), f"Expected mano_betas shape (1, 10), got {mano_betas.shape}"
    mano_layer = ManoLayer(flat_hand_mean=False,
        ncomps=45,
        side=mano_side,
        mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        use_pca=True)
    verts, joints = mano_layer(
        mano_pose[:,:48],
        mano_betas,  # (64, 10)
        mano_pose[:,48:51],  # (64, 3)
    )
    joints = joints/1000
    return joints
    
@dataclasses.dataclass
class Args:
    traj_root: Path
    """Search directory for hand trajectories. This should generally be laid out as something like:

    traj_dir/
        video.vrs
        egoallo_outputs/
            {date}_{start_index}-{end_index}.npz
            ...
        ...
    """
    checkpoint_dir: Path = Path("./experiments/first_try/v2/checkpoints_300000/")#Path("./egoallo_checkpoint_april13/checkpoints_3000000/")

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


def main(args: Args) -> None:
    device = torch.device("cuda")

    traj_paths = InferenceTrajectoryPaths.find(args.traj_root)
    if traj_paths.splat_path is not None:
        print("Found splat at", traj_paths.splat_path)
    else:
        print("No scene splat found.")
    # Get point cloud + floor.
    points_data, floor_z = load_point_cloud_and_find_ground(traj_paths.points_path)

    # Read transforms from VRS / MPS, downsampled.
    transforms = InferenceInputTransforms.load(
        traj_paths.vrs_file, traj_paths.slam_root_dir, fps=30
    ).to(device=device)

    # Note the off-by-one for Ts_world_cpf, which we need for relative transform computation.
    Ts_world_cpf = (
        SE3(
            transforms.Ts_world_cpf[
                args.start_index : args.start_index + args.traj_length + 1
            ]
        )
        @ SE3.from_rotation(
            SO3.from_x_radians(
                transforms.Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
            )
        )
    ).parameters()
    pose_timestamps_sec = transforms.pose_timesteps[
        args.start_index + 1 : args.start_index + args.traj_length + 1
    ]
    Ts_world_device = transforms.Ts_world_device[
        args.start_index + 1 : args.start_index + args.traj_length + 1
    ]
    del transforms

    # Get temporally corresponded HaMeR detections.
    if traj_paths.hamer_outputs is not None:
        hamer_detections = CorrespondedHamerDetections.load(
            traj_paths.hamer_outputs,
            pose_timestamps_sec,
        ).to(device)
    else:
        print("No hand detections found.")
        hamer_detections = None

    # Get temporally corresponded Aria wrist and palm estimates.
    if traj_paths.wrist_and_palm_poses_csv is not None:
        aria_detections = CorrespondedAriaHandWristPoseDetections.load(
            traj_paths.wrist_and_palm_poses_csv,
            pose_timestamps_sec,
            Ts_world_device=Ts_world_device.numpy(force=True),
        ).to(device)
    else:
        print("No Aria hand detections found.")
        aria_detections = None

    print(f"{Ts_world_cpf.shape=}")

    server = None
    if args.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)

    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=hamer_detections,
        aria_detections=aria_detections,
        num_samples=args.num_samples,
        device=device,
        floor_z=floor_z,
    )

    # Save outputs in case we want to visualize later.
    if args.save_traj:
        save_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"_{args.start_index}-{args.start_index + args.traj_length}"
        )
        out_path = args.traj_root / "egoallo_outputs" / (save_name + ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert not out_path.exists()
        (args.traj_root / "egoallo_outputs" / (save_name + "_args.yaml")).write_text(
            yaml.dump(dataclasses.asdict(args))
        )

        posed = traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, Ts_world_cpf[..., 1:, :]
        )
        print(f"Saving to {out_path}...", end="")
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
            Ts_world_root=Ts_world_root.numpy(force=True),
            body_quats=posed.local_quats[..., :21, :].numpy(force=True),
            left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
            right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
            contacts=traj.contacts.numpy(force=True),  # Sometimes we forgot this...
            betas=traj.betas.numpy(force=True),
            frame_nums=np.arange(args.start_index, args.start_index + args.traj_length),
            timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
        )
        print("saved!")

    # Visualize.
    if args.visualize_traj:
        assert server is not None
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf[1:],
            traj,
            body_model,
            hamer_detections,
            aria_detections,
            points_data=points_data,
            splat_path=traj_paths.splat_path,
            floor_z=floor_z,
        )
        while True:
            loop_cb()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
