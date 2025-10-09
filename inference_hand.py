from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
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
from egoallo import network, hand_network
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

def visualize_joints_in_rgb(Trajs:List[HandDenoiseTraj],
                            intrinsics:torch.FloatTensor,
                            rgb_frames:torch.FloatTensor,
                            out_dir:str = "tmp",
                            fps=10, 
                            resize=None) -> None:
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
    subseq_len, height, width, _ = rgb_frames.shape
    if resize is not None:
        width, height = resize
    else:
        width = width * (len(Trajs)+1)
    # 确定视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    
    # 创建视频写入对象
    out = cv2.VideoWriter(out_dir+"/infer.mp4", fourcc, fps, (width, height))
    print("start making the video")
    
    for i in range(subseq_len-2):
        image = rgb_frames[i].numpy().astype(np.uint8)
        composited = image
        for j in range(len(Trajs)):
            vertices=vertices_list[j]
            faces = faces_list[j]
            render_rgb, rend_depth, render_mask = render_joint(vertices[i], faces,
                                                                intrinsics, h=rgb_frames.shape[1], w=rgb_frames.shape[2])
            # breakpoint()
            border_width = 10
            composited_tmp = np.where(
                binary_dilation(
                    render_mask, np.ones((border_width, border_width), dtype=bool)
                )[:, :, None],
                np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
                image,
            )
            composited_tmp = np.where(render_mask[:, :, None], render_rgb, image)
            composited = np.concatenate([composited,composited_tmp], axis=1)
        if resize is not None:
            composited = cv2.resize(composited, resize)
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
    checkpoint_dir: Path = Path("./experiments/hand_train_cond_wrist_motion_more_data/v0/checkpoints_300000/")#Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
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
        visualized=True):
    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(
        device=device,
    )
    alpha_bar_t = noise_constants.alpha_bar_t
    alpha_t = noise_constants.alpha_t
    batch,seq_len,_ = train_batch.mano_pose.shape
    train_batch.to(device)
    print(denoiser_network)
    x_0_packed = hand_network.HandDenoiseTraj(
        mano_betas=train_batch.mano_betas.unsqueeze(1).expand((batch, seq_len, 10)),
        mano_poses=train_batch.mano_pose[:,:,3:48].reshape(batch,seq_len,15,3),
        global_orientation=train_batch.mano_pose[:,:,0:3],
        global_translation=train_batch.mano_pose[:,:,48:],
        mano_side=train_batch.mano_side.unsqueeze(1).expand((batch, seq_len, -1)),
    )
    rel_palm_pose = train_batch.mano_pose[:,:,48:].to(device)

    x_t_packed = torch.randn((batch, seq_len, denoiser_network.get_d_state()), device=device)
    x_t_list = []
    start_time = None

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
                                                    mano_side=x_0_packed.mano_side,
                                                    ).to(device)
                x_0_packed_pred[:, start_t:end_t, :] = (
                    denoiser_network.forward(
                        x_t_packed[:, start_t:end_t, :],
                        torch.tensor([t], device=device).expand((batch,)),
                        rel_palm_pose=rel_palm_pose[:,start_t:end_t, :],
                        project_output_rotmats=False,
                        conds=conds,
                        mask=None,
                    )
                )

            # Take the mean for overlapping regions.
            x_0_packed_pred /= overlap_weights

            x_0_packed_pred = hand_network.HandDenoiseTraj.unpack(x_0_packed_pred).pack()

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
            hand_network.HandDenoiseTraj.unpack(x_t_packed)
        )

        assert start_time is not None
        print("RUNTIME (exclude first optimization)", time.time() - start_time)
    traj = x_t_list[-1]
    ## Assume we have gt global trans/orientation
    traj.global_translation = x_0_packed.global_translation.to(device)
    traj.global_orientation = x_0_packed.global_orientation.to(device)
    if visualized == True:
        visualize_joints_in_rgb([traj,x_0_packed],
                                intrinsics=train_batch.intrinsics.squeeze(0),
                                rgb_frames=train_batch.rgb_frames.squeeze(0),
                                out_dir = "tmp/visualize_hand")
        return traj
    else:
        return traj



def main(args: Args) -> None:
    device = torch.device("cuda")
    dataset = HandHdf5Dataset(split="test",dataset_name = 'dexycb')
    print("Dataset size:", len(dataset))
    visualized = args.visualize
    test_hamer = args.Test_hamer
    if test_hamer == True:
        N = len(dataset)
        errors ={"mano_poses": 20.05293 * 648, "mano_betas": 55.318596*648, "global_orientation":81.87283*648,"global_translation":6.5493684*648}
        joint_errors = 9.660945416403516 * 648
        from hamer_helper import HamerHelper
        hamer_helper = HamerHelper()        
        for i in range(N):
            if i<648:
                continue
            print("processed at index ", i)
            sample = [dataset.__getitem__(i).to(device)] 
            keys = vars(sample[0]).keys()
            gt = type(sample[0])(**{k: torch.stack([getattr(b, k) for b in sample]) for k in keys})
            skip_cam = -1
            subseq_len = gt.mano_pose.shape[1]
            hamer_out = []
            for j in range(subseq_len):
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
        train_batch = []
        rand_id = 0 
        if visualized == True:
            num_samples = 1
            for i in range(num_samples):
                data_id = 324#random.randint(1,1000)
                rand_id = data_id
                sample = dataset.__getitem__(data_id)
                train_batch.append(sample)
            keys = vars(train_batch[0]).keys()
            train_batch = type(train_batch[0])(**{k: torch.stack([getattr(b, k) for b in train_batch]) for k in keys})
            pred = inference_and_visualize(denoiser_network,train_batch,device,visualized)
            _,_,joints_pred = pred.apply_to_hand()
            error_joints =  torch.sqrt(((train_batch.mano_joint_3d.to(device) - joints_pred)**2).sum(dim=-1)).mean(dim=-1).sum().cpu().numpy()
            print("Joint error is ", error_joints)
            print("take the id",rand_id,"as the test sample")
            # pred_list = []
            # for i in range(4):
            #     pred = inference_and_visualize(denoiser_network,train_batch,device,False)
            #     pred_list.append(pred)
            # visualize_joints_in_rgb(pred_list,
            #                         intrinsics=train_batch.intrinsics.squeeze(0),
            #                         rgb_frames=train_batch.rgb_frames.squeeze(0),
            #                         out_dir = "tmp/visualize_hand")
        else:
            errors={"mano_betas":0,"mano_poses":0,"global_orientation":0,"global_translation":0,"3D_joints":0}
            total_size = 0
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=64,
                                                    shuffle=False,
                                                    num_workers=2,
                                                    pin_memory=True,
                                                    collate_fn=collate_dataclass,
                                                    drop_last=False)
            for train_batch in dataloader:
                total_size += train_batch.mano_betas.shape[0]
                pred = inference_and_visualize(denoiser_network,train_batch,device,visualized)
                batch,seq_len,_ = train_batch.mano_pose.shape
                x_0_packed = hand_network.HandDenoiseTraj(
                                                            mano_betas=train_batch.mano_betas.unsqueeze(1).expand((batch, seq_len, 10)),
                                                            mano_poses=train_batch.mano_pose[:,:,3:48].reshape(batch,seq_len,15,3),
                                                            global_orientation=train_batch.mano_pose[:,:,0:3],
                                                            global_translation=train_batch.mano_pose[:,:,48:],
                                                            mano_side=train_batch.mano_side.unsqueeze(1).expand((batch, seq_len, -1)),
                                                        )
                _,_,joints_pred = pred.apply_to_hand()
                for var in errors.keys():
                    if var  != "3D_joints" and var  !="mano_poses":
                        errors[var]+=((getattr(x_0_packed,var)-getattr(pred,var).cpu())**2).sum().numpy()
                    else:#MJPE
                        if var == "3D_joints":
                            errors[var]+= torch.sqrt(((train_batch.mano_joint_3d.to(device) - joints_pred)**2).sum(dim=-1)).mean(dim=-1).sum().cpu().numpy()
                        else:
                            errors[var]+=((getattr(x_0_packed,var)-getattr(pred,var).cpu())**2).sum().numpy()
            for var in errors.keys():
                print("var: ",var," MSE error is:",errors[var]/total_size)
            


    
    
    
if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
