from manopth.manolayer import ManoLayer
import os
# from typing import Any, Dict, Optional
# from Hot3DVisualizer import Hot3DVisualizer
# from dataset_api import Hot3dDataProvider
from tqdm import tqdm
import torch
import imageio.v3 as iio
# from data_loaders.HeadsetPose3dProvider import (
#     HeadsetPose3dProvider,
#     load_headset_pose_provider_from_csv,
# )

# from data_loaders.headsets import Headset
# from data_loaders.io_utils import load_json
# from data_loaders.loader_object_library import ObjectLibrary
# from data_loaders.mano_layer import MANOHandModel
# from data_loaders.ManoHandDataProvider import MANOHandDataProvider

# from data_loaders.ObjectPose3dProvider import (
#     load_pose_provider_from_csv,
#     ObjectPose3dProvider,
# )
# from data_loaders.PathProvider import Hot3dDataPathProvider

# from data_loaders.QuestDataProvider import QuestDataProvider
# from data_loaders.UmeTrackHandDataProvider import UmeTrackHandDataProvider


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
    global_translation = np.eye(4)
    global_translation[1:3, :] *= -1  # flip the y and z axes to match opengl
    camera_node = pyrender.Node(camera=camera, matrix=global_translation)
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
    vertices, joints = manopth_mano_layer(
        mano_pose[:,:48],
        mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
        mano_pose[:,48:51]+delta.reshape(1,3)
    )
    vertices = vertices/ 1000  # Convert to meters
    
    # left_mano_output = mano_layer(
    #     betas=mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
    #     global_orient=mano_pose[:, :3],
    #     hand_pose=mano_pose[:, 3:48],
    #     transl=mano_pose[:, -3:],
    #     return_verts=True,
    # )
    # vertices = left_mano_output.vertices
    faces_m = manopth_mano_layer.th_faces
    # faces_m = mano_layer.faces_tensor
    return vertices, faces_m,joints

def visualize_joints_in_rgb(images,intrinsics,mano_pose, mano_betas, mano_side, out_dir:str = "tmp",ind=0,ext=None,joints=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if joints is None:
        vertices, faces,joint = get_vertices_faces(mano_pose, mano_betas,mano_side,ext)
        # breakpoint()
        # hand_vertices_in_camera = []
        # # reset plt
        # plt.clf()

        # plt.imshow(images, interpolation="nearest")
        vertices = vertices[0].numpy()
        # # vertices @ [1 0 0, 0 -1 0, 0 0 -1]
        # vertices[:,1] = -vertices[:,1]
        # vertices[:,2] = -vertices[:,2]
        # if ext!=None:
        #     vertices = (ext @ vertices.T).T
        # for vert in vertices:
        #     vertice_2d_camera_coordinates = intrinsics.project(
        #         vert
        #     )
        #     if vertice_2d_camera_coordinates is not None:
        #         hand_vertices_in_camera.append(vertice_2d_camera_coordinates)

        # plt.scatter(
        #     x=[x[0] for x in hand_vertices_in_camera],
        #     y=[x[1] for x in hand_vertices_in_camera],
        #     s=1,
        #     c="r" if hand_pose_data.handedness == Handedness.Right else "b",
        # )
        # plt.savefig(os.path.join(out_dir, f"{ind}_{mano_side}.jpg"))
        # return

        border_color = [255, 0, 0]
        # for i in range(vertices.shape[0]):
        # image = sample.rgb_frames[i].numpy().astype(np.uint8)
        # render_rgb, render_depth, render_mask = render_joint(
        #     vertices[i].numpy(),
        #     faces.numpy(),
        #     intrinsics,
        #     h=image.shape[0],
        #     w=image.shape[1],
        # )

        # # Convert to uint8
        # render_rgb = (render_rgb * 255).astype(np.uint8)
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
import pickle

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord.transpose(1,0) - T)
    return cam_coord

if __name__ == '__main__':
    ho3d_root = "/public/datasets/handdata/HO3D_v3"
    split = "train" # or 'evaluation'
    
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
    cam_id = "0"
    seq = "ABF1"
    seq_name = "ABF1"+cam_id
    rgb_dir = os.path.join(ho3d_root, split, seq_name, "rgb")
    meta_dir = os.path.join(ho3d_root, split, seq_name, "meta")
    img_id='0115'
    img_path = os.path.join(rgb_dir, f"{img_id}.jpg")
    meta_path = os.path.join(meta_dir, f"{img_id}.pkl")


    camera_trans_path = os.path.join(ho3d_root, "calibration", seq, "calibration", f"trans_{cam_id}.txt")
    camera_intr_path = os.path.join(ho3d_root, "calibration", seq, "calibration", f"cam_{cam_id}_intrinsics.txt")
    img = iio.imread(img_path)
    with open(meta_path, 'rb') as f:
        meta_data =  pickle.load(f)
    with open(camera_trans_path, 'r') as f:
        camera_trans_lines = f.readlines() # to 4*4 matrix
        for i in range(4):
            camera_trans_lines[i] = camera_trans_lines[i].strip().split(" ")
            camera_trans_lines[i] = list(map(float, camera_trans_lines[i]))
        camera_trans = np.array(camera_trans_lines)

    # with open(camera_intr_path, 'r') as f:
    #     camera_intr = f.readlines()
    # breakpoint()

    intrinsics_matrix = meta_data['camMat']
    intrinsics = np.array([intrinsics_matrix[0,0], intrinsics_matrix[1,1], intrinsics_matrix[0,2], intrinsics_matrix[1,2]])
    mano_pose_right = torch.FloatTensor(meta_data['handPose'])
    mano_betas_right = torch.FloatTensor(meta_data['handBeta'])
    mano_rot_aa = mano_pose_right[:3].reshape(1,3)
    mano_rot_mat, _ = cv2.Rodrigues(mano_rot_aa.numpy())
    # [1 0 0, 0 -1 0, 0 0 -1] @ mano_rot_mat
    mano_rot_mat[1,:] = -mano_rot_mat[1,:]
    mano_rot_mat[2,:] = -mano_rot_mat[2,:]
    mano_fix_aa,_ = cv2.Rodrigues(mano_rot_mat)
    mano_pose_right[:3] = torch.FloatTensor(mano_fix_aa.reshape(3))
    cam2world_rot = camera_trans[:3,:3]
    cam2world_trans = camera_trans[:3,3]
    mano_trans = torch.FloatTensor(meta_data['handTrans'])
    # mano_trans @ [1 0 0, 0 -1 0, 0 0 -1]
    mano_trans = [mano_trans[0], -mano_trans[1], -mano_trans[2]]
    # breakpoint()
    root_j = cal_root_j(mano_betas_right.unsqueeze(0), mano_layer_right)
    delta = -root_j + torch.tensor([1,0,0,0,-1,0,0,0,-1],dtype=root_j.dtype).reshape(3,3) @ root_j
    mano_pose_right = torch.cat([mano_pose_right.reshape(1,48), torch.FloatTensor(mano_trans).reshape(1,3)+delta.reshape(1,3)],dim=1)

    # cam2mano_rot = cam2world_rot @ mano_rot_mat
    # # cam2mano_trans = cam2world_trans + cam2world_rot @ mano_trans.numpy()
    # cam2mano_trans = cam2world_trans
    # root_j = cal_root_j(mano_betas_right.unsqueeze(0), mano_layer_right)
    # delta = -root_j + torch.tensor(cam2world_rot,dtype=root_j.dtype) @ root_j
    # cam2mano_trans = cam2mano_trans + delta.reshape(3).numpy()
    # cam2mano_rot_aa,_ = cv2.Rodrigues(cam2mano_rot)
    # mano_pose_right[:3] = torch.FloatTensor(cam2mano_rot_aa.reshape(3))
    # mano_pose_right = torch.cat([mano_pose_right.reshape(1,48), torch.FloatTensor(cam2mano_trans).reshape(1,3)],dim=1)
# dict_keys(['handBeta', 'handTrans', 'handPose', 'handJoints3D', 'camMat', 'objRot', 'objTrans', 'camIDList', 'objCorners3D', 'objCorners3DRest', 'objName', 'objLabel', 'handVertContact', 'handVertDist', 'handVertIntersec', 'handVertObjSurfProj'])


        # cam2mano_rot = cam2world_rot @ mano_rot
        # cam2mano_trans = cam2world_trans + cam2world_rot @ mano_trans

        # mano_rot_angle,_ = cv2.Rodrigues(cam2mano_rot.numpy())
        # mano_rot_angle = torch.FloatTensor(mano_rot_angle.reshape(3))
        # mano_pose_right[0,:3] = torch.FloatTensor(mano_rot_angle.reshape(3))
        # mano_pose_right[:,3:48] = add_mean_pose45(mano_pose_right[:,3:48], mano_layer_right)
        # mano_pose_right = torch.cat([mano_pose_right,cam2mano_trans.reshape(1,3)],dim=1)

        # joint_array = np.array(this_joint_data['world_coord']).reshape(-1,3)
        # valid = np.array(this_joint_data['joint_valid']).reshape(-1)
        # valid_joint_array = joint_array[valid==1]
    # breakpoint()
    visualize_joints_in_rgb(img, intrinsics, mano_pose_right, mano_betas_right, "right", out_dir="tmp", ind=int(img_id))