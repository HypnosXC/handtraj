from manopth.manolayer import ManoLayer
import os
# from typing import Any, Dict, Optional
# from Hot3DVisualizer import Hot3DVisualizer
# from dataset_api import Hot3dDataProvider
from tqdm import tqdm
import torch
# from data_loaders.loader_hand_poses import HandType
# import data_loaders.HandBox2dDataProvider as HandBox2dDataProvider
# from data_loaders.mano_layer import loadManoHandModel

# import data_loaders.ObjectBox2dDataProvider as ObjectBox2dDataProvider
# from projectaria_tools.core.stream_id import StreamId  # @manual
# from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
# from data_loaders.AriaDataProvider import AriaDataProvider
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
        breakpoint()
        # hand_vertices_in_camera = []
        # # reset plt
        # plt.clf()

        # plt.imshow(images, interpolation="nearest")
        vertices = vertices[0].numpy()
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
        breakpoint()
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

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord.transpose(1,0) - T)
    return cam_coord

if __name__ == '__main__':

    # val_data_path = "/public/datasets/handdata/interhand26m/anno/annotation/val/InterHand2.6M_val_data.json"
    # with open(val_data_path) as f:
    #     val_data = json.load(f)
    # breakpoint()


    # interhand_img_dir= "/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images"
    # img_list_file = "/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/sort_all.json"
    interhand_anno_dir = "/public/datasets/handdata/interhand26m/anno/annotation"
    mano_model_folder = "/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano"
    capture_cnt = 0
    img_root="/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images"
    camera_keys = ['campos', 'camrot', 'focal', 'princpt']
    # camrot 3*3 list[list] camera_data['0']['camrot']['400006']
    # campos 3 list camera_data['0']['campos']['400006']
    # focal 2 list camera_data['0']['focal']['400006']
    # princpt 2 list camera_data['0']['princpt']['400006']

    # mano_data['0']['66662']['left'].keys()
    mano_keys = ['pose', 'shape', 'trans']
    # shape 10 list mano_data['0']['66662']['left']['shape']
    # pose 48 list mano_data['0']['66662']['left']['pose']
    # trans 3 list mano_data['0']['66662']['left']['trans']

    joint_data_keys = ['left', 'right']
    # joint_data['0']['66662'].keys()
    joint_data_keys = ['world_coord', 'joint_valid', 'hand_type', 'hand_type_valid', 'seq']
    # world_coord 42*3 list[list] joint_data['0']['66662']['world_coord']
    # joint_valid 42*1 list joint_data['0']['66662']['joint_valid'] 1 or 0
    # hand_type str joint_data['0']['66662']['hand_type']
    # hand_type_valid True or False joint_data['0']['66662']['hand_type_valid']
    # seq str joint_data['0']['66662']['seq']

    picked_camera = 'cam400265'
    picked_camera_id = 400265
    picked_capture = 'Capture0'
    picked_split = 'test'
    picked_capture_id = 0
    picked_split_root = os.path.join(img_root, picked_split)
    # picked_seq = '0030_middletip'

    all_json="/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/sort_all.json"
    # with open(all_json) as f:
    #     all_data = json.load(f) 
    # this_seq = all_data[picked_split][picked_capture][picked_camera]
    # print(f"picked {picked_split} {picked_capture} {picked_camera}, total {len(this_seq)} images")
    # breakpoint()
    with open(os.path.join(interhand_anno_dir, picked_split, f'InterHand2.6M_{picked_split}_MANO_NeuralAnnot.json')) as f:
        mano_data = json.load(f)
    with open(os.path.join(interhand_anno_dir, picked_split, f'InterHand2.6M_{picked_split}_camera.json')) as f:
        camera_data = json.load(f)
    with open(os.path.join(interhand_anno_dir, picked_split, f'InterHand2.6M_{picked_split}_joint_3d.json')) as f:
        joint_data = json.load(f)
    # with open(img_list_file) as f:
    #     img_list_data = json.load(f)
    # img_list_data = img_list_data[picked_split][picked_capture][picked_camera]
    # self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
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
    save_path = os.path.join("tmp_interhand", f"{picked_capture}_{picked_camera}")
    os.makedirs(save_path, exist_ok=True)
    # img_path_list = os.path.join(interhand_img_dir, picked_split, 'Capture' + picked_capture, picked_seq, 'cam' + picked_camera)
    # img_path_list = sorted(os.listdir(img_path_list))
    # print(img_path_list)
    # print(len(img_list_data))
    # breakpoint()

    this_img_dir = os.path.join(picked_split_root, picked_capture,'ROM07_RT_Finger_Occlusions',picked_camera)
    img_list = sorted(os.listdir(this_img_dir))
    for img_path in tqdm(img_list[1009:]):
        frame_idx_ori = img_path.split('image')[-1].split('.')[0]
        frame_idx = str(int(frame_idx_ori))

        this_joint_data = joint_data[str(picked_capture_id)][frame_idx]
        this_mano_data = mano_data[str(picked_capture_id)][frame_idx]
        camrot_list = camera_data[str(picked_capture_id)]['camrot'][str(picked_camera_id)]
        campos_list = camera_data[str(picked_capture_id)]['campos'][str(picked_camera_id)]
        focal_list = camera_data[str(picked_capture_id)]['focal'][str(picked_camera_id)]
        princpt_list = camera_data[str(picked_capture_id)]['princpt'][str(picked_camera_id)]
        camrot = np.array(camrot_list).reshape(3,3)
        campos = np.array(campos_list).reshape(3)/1000

        cam2world_rot = torch.FloatTensor(camrot)
        cam2world_trans = -cam2world_rot @ torch.FloatTensor(campos).reshape(3,1)
        # cam2world_rot = torch.FloatTensor(camrot)
        # cam2world_trans = torch.FloatTensor(campos).reshape(3,1)

        focal = np.array(focal_list).reshape(2)
        princpt = np.array(princpt_list).reshape(2)
        intrinsics = np.array([focal[0], focal[1], princpt[0], princpt[1]])
        img = iio.imread(os.path.join(this_img_dir, img_path))
        img_height, img_width, _ = img.shape
        joint_world = np.array(this_joint_data['world_coord']).reshape(-1,3)
        # joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
        # right 0-21
        # this_mano_data['left']=None
        # mano_right = this_mano_data['right']
        mano_right = this_mano_data['right']
        # breakpoint()
        mano_betas_right = torch.FloatTensor(mano_right['shape']).view(-1)
        mano_pose_right = torch.FloatTensor(mano_right['pose']).view(1,-1)
        # mano_trans_right = torch.FloatTensor(mano_right['trans']).view(1,-1)
        # breakpoint()
        mano_rot,_ = cv2.Rodrigues(mano_pose_right[0,:3].numpy())
        mano_rot = torch.FloatTensor(mano_rot)
        mano_trans = torch.FloatTensor(mano_right['trans']).reshape(3,1)
        extrinsics_matrix = torch.zeros((4,4))
        extrinsics_matrix[:3,:3] = cam2world_rot
        extrinsics_matrix[:3,3] = cam2world_trans.reshape(3)

        cam2mano_rot = cam2world_rot @ mano_rot
        cam2mano_trans = cam2world_trans + cam2world_rot @ mano_trans

        # cam2mano_rot = mano_rot
        # cam2mano_trans = mano_trans

        mano_rot_angle,_ = cv2.Rodrigues(cam2mano_rot.numpy())
        mano_rot_angle = torch.FloatTensor(mano_rot_angle.reshape(3))
        mano_pose_right[0,:3] = torch.FloatTensor(mano_rot_angle.reshape(3))
        mano_pose_right[:,3:48] = add_mean_pose45(mano_pose_right[:,3:48], mano_layer_right)
        mano_pose_right = torch.cat([mano_pose_right,cam2mano_trans.reshape(1,3)],dim=1)

        joint_array = np.array(this_joint_data['world_coord']).reshape(-1,3)
        valid = np.array(this_joint_data['joint_valid']).reshape(-1)
        valid_joint_array = joint_array[valid==1]
        breakpoint()
        visualize_joints_in_rgb(img, intrinsics, mano_pose_right, mano_betas_right, "right", out_dir=save_path, ind=frame_idx_ori, ext=extrinsics_matrix.numpy())


