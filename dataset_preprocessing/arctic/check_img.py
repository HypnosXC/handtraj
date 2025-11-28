
from common.body_models import construct_layers
from src.arctic.preprocess_dataset import construct_loader
from glob import glob
from tqdm import tqdm
import os

from manopth.manolayer import ManoLayer
import os
from tqdm import tqdm
import torch
import imageio.v3 as iio


import numpy as np
# from scipy.spatial.transform import Rotation as R

# from hamer.utils.mesh_renderer import create_raymond_lights
import cv2
from scipy.ndimage import binary_dilation
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh
import pyrender
import matplotlib.pyplot as plt
def create_raymond_lights():
    # import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

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

# def get_vertices_faces(mano_pose,mano_betas,mano_side="right") -> tuple[torch.Tensor, torch.Tensor]:
#     mano_layer = ManoLayer(
#         flat_hand_mean=True,
#         ncomps=45,
#         use_pca=False,
#         side=mano_side,
#         mano_root="/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano",
#     )
#     vertices, joints = mano_layer(
#         mano_pose[:,:48],
#         mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
#         mano_pose[:,48:51]
#     )
#     vertices = vertices/ 1000  # Convert to meters
#     faces_m = mano_layer.th_faces
#     return vertices, faces_m

def cal_root_j(th_betas, layer: ManoLayer) -> torch.Tensor:
    th_v_shaped = torch.matmul(layer.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + layer.th_v_template
    th_j = torch.matmul(layer.th_J_regressor, th_v_shaped)
    return th_j[:, 0, :].contiguous().view(3, 1)

def get_vertices_faces(mano_pose,mano_betas,mano_side="right",ext=None) -> tuple[torch.Tensor, torch.Tensor]:
    manopth_mano_layer = ManoLayer(
        flat_hand_mean=True,
        ncomps=45,
        use_pca=False,
        side=mano_side,
        mano_root="/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano",
    )
    # mano_layer = smplx.create(
    #     "/home/yunqili/code/hamer/_DATA/data/mano/MANO_LEFT.pkl",
    #     "mano",
    #     use_pca=False,
    #     is_rhand= mano_side=="right",
    #     flat_hand_mean=True,
    # )
    if ext is None:
        delta = torch.zeros((1,3),dtype=mano_betas.dtype)
    else:
        root_j = cal_root_j(mano_betas.unsqueeze(0), manopth_mano_layer)
        delta = -root_j + torch.tensor(ext[:3,:3],dtype=root_j.dtype) @ root_j
    # breakpoint()
    breakpoint()
    vertices, joints = manopth_mano_layer(
        mano_pose[:,:48],
        mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
        mano_pose[:,48:51]+delta.reshape(1,3)
    )
    vertices = vertices/ 1000  # Convert to meters
    faces_m = manopth_mano_layer.th_faces
    # faces_m = mano_layer.faces_tensor
    return vertices, faces_m,joints

def visualize_joints_in_rgb(images,intrinsics,mano_pose, mano_betas, mano_side, out_dir:str = "tmp",ind=0,ext=None,joints=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if joints is None:
        vertices, faces,_ = get_vertices_faces(mano_pose, mano_betas,mano_side,ext)
        breakpoint()
        vertices = vertices[0].numpy()

        border_color = [255, 0, 0]
        render_rgb, rend_depth, render_mask = render_joint(vertices, faces.numpy(),
                                                            intrinsics, h=images.shape[0],w = images.shape[1])

        border_width = 10
        composited = np.where(
            binary_dilation(
                render_mask, np.ones((border_width, border_width), dtype=bool)
            )[:, :, None],
            np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
            images,
        )
        # breakpoint()
        composited = np.where(render_mask[:, :, None], render_rgb, images)
        composited = np.concatenate([images, composited], axis=1)
        iio.imwrite(os.path.join(out_dir, f"{ind}_{mano_side}.jpg"), composited)
    else:
        _,_,joi_ = get_vertices_faces(mano_pose, mano_betas,mano_side,ext)
        ext_rot = ext[:3,:3]
        ext_trans = ext[:3,3]
        joints_cam = ext_rot @ joints.T + ext_trans.reshape(3,1)
        joints_cam = joints_cam.T
        proj_joints = np.zeros((joints_cam.shape[0],2))
        for j in range(joints_cam.shape[0]):
            proj_joints[j,:] = np.array([
                intrinsics[0]*joints_cam[j,0]/joints_cam[j,2]+intrinsics[2],
                intrinsics[1]*joints_cam[j,1]/joints_cam[j,2]+intrinsics[3]
            ])
        border_color = [255, 0, 0]
        composited = images.copy()
        for j in range(proj_joints.shape[0]):
            cv2.circle(composited, (int(proj_joints[j,0]), int(proj_joints[j,1])), 3, border_color, -1)
        composited = np.concatenate([images, composited], axis=1)
        # breakpoint()
        iio.imwrite(os.path.join(out_dir, f"{ind}_{mano_side}_joints.jpg"), composited)




import torch

@torch.no_grad()
def pca15_to_axis45(coeffs15: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    """
    coeffs15: (B, 15)  的 PCA 系数
    返回:     (B, 45)  的 axis-angle（仅手指15关节；不含全局3维）
    """
    # manopth 内部变量名（常见为以下；不同版本可能略有差异）
    # th_selected_comps: PCA基
    # th_hands_mean:     均值 (通常 shape 为 (1, 45))
    basis = layer.th_selected_comps  # 可能是 (15,45) 或 (45,15)
    mean  = layer.th_hands_mean      # 形如 (1,45) 或 (45,)
    print("mean : ", mean)
    # 统一 mean 形状为 (1,45)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)  # (1,45)
    # 根据 basis 形状做乘法（确保得到 (B,45)）
    if basis.shape == (coeffs15.shape[-1], 45):     # (15,45)
        pose45 = coeffs15 @ basis                   # (B,45)
    elif basis.shape == (45, coeffs15.shape[-1]):   # (45,15)
        pose45 = coeffs15 @ basis.t()               # (B,45)
    else:
        raise ValueError(f"Unexpected basis shape: {basis.shape}")

    pose45 = pose45 + mean                          # 广播加均值
    return pose45

def add_mean_pose45(coeffs45: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    """
    coeffs45: (B, 45)  的 axis-angle（仅手指15关节；不含全局3维）
    返回:     (B, 45)  的 axis-angle（仅手指15关节；不含全局3维）
    """
    mean  = layer.th_hands_mean      # 形如 (1,45) 或 (45,)
    print("mean : ", mean)
    # 统一 mean 形状为 (1,45)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)  # (1,45)
    coeffs45 = coeffs45 + mean                          # 广播加均值
    return coeffs45

import json

# def world2cam(world_coord, R, T):
#     cam_coord = np.dot(R, world_coord.transpose(1,0) - T)
#     return cam_coord


# mano_ps = glob(f"/public/datasets/handdata/arctic/unpack/arctic_data/data/raw_seqs/*/*.mano.npy")
raw_seq_path = "/public/datasets/handdata/arctic/unpack/arctic_data/data/raw_seqs"
cropped_img_path = "/public/datasets/handdata/arctic/unpack/arctic_data/data/cropped_images"
# pbar = tqdm(mano_ps)
misc_p = "/public/datasets/handdata/arctic/unpack/arctic_data/data/meta/misc.json"
import json
with open(misc_p, "r") as f:
    misc = json.load(f)
    # breakpoint()
# misc.keys()
# dict_keys(['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10'])
# cam 0 : ego 
# cam 1-8: intr 0-7

# misc['s01'].keys()
# dict_keys(['gender', 'image_size', 'intris_mat', 'ioi_offset', 'world2cam'])
# misc['s01']['intris_mat']
# 8*3*3 list

# misc['s01']['intris_mat'][0]
# [[7270.5244140625, 0.0, 891.011962890625], [0.0, 7270.5244140625, 1358.8863525390625], [0.0, 0.0, 1.0]]

# len(misc['s01']['world2cam'])
# 8
# (Pdb) len(misc['s01']['world2cam'][0])
# 4
# (Pdb) len(misc['s01']['world2cam'][0][0])
# 4

data_split = {
    "train": ["s01", "s02", "s04", "s05", "s06", "s07", "s08"],
    "val": ["s09"],
    "test": ["s10"],
}

import h5py
str_dt = h5py.string_dtype('utf-8', None)
dtypes = {
    'intrinsics':'f4',
    'mano_side':str_dt,
    'mano_betas':'f4',
    'extrinsics':'f4',
    'mano_poses':'f4',
    'mano_joint_3d':'f4',
    'video_name': str_dt,
}
shapes = {
    'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
    'intrinsics': (4,),  # fx,fy,ppx,ppy
    'mano_side': (1,),  # single side (left or right)
    'mano_betas': (10,),  # 10 betas for MANO model 
    'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
    'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
    'video_name': (1,),  # single video name
}
hdf5_path="/public/datasets/handdata/arctic.hdf5"

mano_model_folder = "/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano"
mano_layer_right = ManoLayer(
    flat_hand_mean=False,
    ncomps=45,
    use_pca=False,
    side="right",
    mano_root=mano_model_folder,
)
mano_layer_left = ManoLayer(
    flat_hand_mean=False,
    ncomps=45,
    use_pca=False,
    side="left",
    mano_root=mano_model_folder,
)
mano_layer_joint_left = ManoLayer(
    flat_hand_mean=True,
    ncomps=45,
    use_pca=False,
    side="left",
    mano_root=mano_model_folder,
)
mano_layer_joint_right = ManoLayer(
    flat_hand_mean=True,
    ncomps=45,
    use_pca=False,
    side="right",
    mano_root=mano_model_folder,
)

# mappings={
#     'train': [],
#     'val': [],
#     'test': [],
# }

import imageio.v3 as iio
import cv2

# with h5py.File(hdf5_path, 'w') as f:
save_mp4_root = "/public/datasets/handdata/arctic/picked_videos"

resplit_p = "/public/datasets/handdata/arctic/record_arctic_valid_seqs_32_64_resplit.json"
img_root = "/public/datasets/handdata/arctic/unpack/arctic_data/data/cropped_images"

task_list = []
resplit_dict = {}

split = 'test'
# for split in ['train', 'val', 'test']:
    # resplit_dict[split] = []
    # split_mp4_dir = os.path.join(save_mp4_root, split)
    # os.makedirs(split_mp4_dir, exist_ok=True)
s_idx = 's10'
video_path = "/public/datasets/handdata/arctic/picked_videos/test/s10_capsulemachine_use_01_66_64_5.mp4"
cam_id = 5
    # for s_idx in data_split[split]:

ioi_offset = misc[s_idx]['ioi_offset']
        # print(f"Processing subject {s_idx}")
s_path = os.path.join(raw_seq_path, s_idx)
        # manos = os.listdir(s_path)
        # manos = [m for m in manos if m.endswith(".mano.npy")]
        # mano_names = [m.replace(".mano.npy", "") for m in manos]
        # for mano_name in tqdm(mano_names):
mano_name = "capsulemachine_use_01"

handtype = 'right'
start_idx = 66
saved_length = 64
mano_p = os.path.join(s_path, f"{mano_name}.mano.npy")
# egocam_p = os.path.join(s_path, f"{mano_name}.egocam.dist.npy")
# breakpoint()
this_img_dir = os.path.join(img_root, s_idx, mano_name)
# print(f"Processing {mano_p}")
loader = construct_loader(mano_p)
# for idx,batch in enumerate(loader):
#     assert idx == 0

batch = next(iter(loader))

world2cam = np.array(misc[s_idx]['world2cam'][cam_id-1])

            # world2cam_rot = world2cam[:3,:3]
            # world2cam_T = world2cam[:3,3]
            # cam2world_rot = world2cam_rot.T
            # cam2world_T = -cam2world_rot @ world2cam_T
cam2world_rot = world2cam[:3,:3]
cam2world_T = world2cam[:3,3]

extrinsics = np.concatenate([cam2world_rot, cam2world_T[:,None]], axis=1)
# for handtype in ['right', 'left']:
if handtype == 'right':
    mano_poses_45 = batch['pose_r'][start_idx:start_idx+saved_length]

    mano_betas = torch.mean(batch['shape_r'], dim=0)  # (10)
    mano_poses_45 = add_mean_pose45(mano_poses_45, mano_layer_right)
    root_j = cal_root_j(mano_betas[None,:], mano_layer_right)  # (3,1)
    world2mano_rot = batch['rot_r'][start_idx:start_idx+saved_length]
    world2mano_trans = batch['trans_r'][start_idx:start_idx+saved_length]
    mano_for_joint = mano_layer_joint_right
else:
    mano_poses_45 = batch['pose_l'][start_idx:start_idx+saved_length]
    mano_betas = torch.mean(batch['shape_l'], dim=0)  # (10)
    mano_poses_45 = add_mean_pose45(mano_poses_45, mano_layer_left)
    root_j = cal_root_j(mano_betas[None,:], mano_layer_left)  # (3,1)
    world2mano_rot = batch['rot_l'][start_idx:start_idx+saved_length]
    world2mano_trans = batch['trans_l'][start_idx:start_idx+saved_length]
    mano_for_joint = mano_layer_joint_left
# breakpoint()
if saved_length < 64:
    pad_len = 64 - saved_length
    # breakpoint()
    mano_poses_45 = torch.cat([mano_poses_45, torch.zeros((pad_len, 45), dtype=mano_poses_45.dtype)], dim=0)
    world2mano_rot = torch.cat([world2mano_rot, torch.zeros((pad_len, 3), dtype=world2mano_rot.dtype)], dim=0)
    world2mano_trans = torch.cat([world2mano_trans, torch.zeros((pad_len, 3), dtype=world2mano_trans.dtype)], dim=0)

intr_mat = misc[s_idx]['intris_mat'][cam_id-1] # (3,3)list
fx = intr_mat[0][0]
fy = intr_mat[1][1]
ppx = intr_mat[0][2]/2
ppy = intr_mat[1][2]/2.8
# ppx = 500
# ppy = 500  # because the image is resized to half
# breakpoint()
intrinsics = [fx, fy, ppx, ppy]
mano_beta_np = mano_betas.numpy()
world2cam_rot_matrix = []
for w2m_rot in world2mano_rot:
    w2m_rot_matrix, _ = cv2.Rodrigues(w2m_rot.numpy())
    w2m_rot_matrix = torch.tensor(w2m_rot_matrix, dtype=torch.float32)
    world2cam_rot_matrix.append(w2m_rot_matrix)
world2mano_rot_matrix = torch.stack(world2cam_rot_matrix, dim=0)
cam2mano_rot = torch.tensor(cam2world_rot, dtype=torch.float32) @ world2mano_rot_matrix  # (T, 3, 3)
cam2mano_rot_angle = []
for c2m_rot in cam2mano_rot:
    c2m_rot_angle, _ = cv2.Rodrigues(c2m_rot.numpy())
    cam2mano_rot_angle.append(torch.tensor(c2m_rot_angle, dtype=torch.float32))
cam2mano_rot_angle = torch.stack(cam2mano_rot_angle, dim=0)
# (3,3)@(T,3,1) + (T,3,1) = (T,3,1)
cam2mano_trans = torch.tensor(cam2world_rot, dtype=torch.float32) @ torch.tensor(world2mano_trans.numpy(), dtype=torch.float32).reshape(cam2mano_rot_angle.shape[0],3,1) + torch.tensor(cam2world_T, dtype=torch.float32).reshape(3,1)
delta = -root_j + torch.tensor(cam2world_rot,dtype=root_j.dtype) @ root_j
mano_pose_51 = torch.cat([cam2mano_rot_angle.squeeze(2), mano_poses_45, (cam2mano_trans+delta).squeeze(2)], dim=1)  # (T, 51)

mano_pose_51_np = mano_pose_51.numpy().reshape(64, 1, 51)
_, joints = mano_for_joint(
    mano_pose_51[:,:48],
    mano_betas.repeat(mano_pose_51.shape[0], 1),
    mano_pose_51[:,48:51]+delta.reshape(1,3)
)
joints = joints.numpy().reshape(64, 1, 21, 3)/1000
mask = np.zeros((64,), dtype=np.uint8)
mask[:saved_length] = 1

# read video
video = iio.imread(video_path)
want_id=50
for frame_id in range(want_id,want_id+3):
    frame = video[frame_id]
    this_mano_pose_51 = mano_pose_51[frame_id:frame_id+1]
    visualize_joints_in_rgb(frame, intrinsics, this_mano_pose_51, mano_betas, handtype, out_dir="tmp",ind=frame_id)