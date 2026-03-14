import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh
import imageio.v3 as iio
import cv2
import torch
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
import pyrender
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer
import pandas as pd

os.environ["PYOPENGL_PLATFORM"] = "egl"

def create_raymond_lights():
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
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


def render_joint(vertices: np.ndarray, faces: np.ndarray, intrinsics: np.ndarray, h: int, w: int):
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h, point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(1.0, 1.0, 0.9, 1.0),
    )
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, "mesh")
    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics[0],
        fy=intrinsics[1],
        cx=intrinsics[2],
        cy=intrinsics[3],
        zfar=1e12,
        znear=0.001,
    )
    for node in create_raymond_lights():
        scene.add_node(node)
    cam_pose = np.eye(4)
    cam_pose[1:3, :] *= -1
    camera_node = pyrender.Node(camera=camera, matrix=cam_pose)
    scene.add_node(camera_node)
    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)  
    mask = color[..., -1] > 0
    return color[..., :3], rend_depth, mask

def cal_root_j(th_betas: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    th_v_shaped = torch.matmul(layer.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1) + layer.th_v_template
    th_j = torch.matmul(layer.th_J_regressor, th_v_shaped)
    return th_j[:, 0, :].contiguous().view(3, 1)

def get_vertices_faces(mano_pose, mano_betas, mano_side="right", ext=None):
    manopth_mano_layer = mano_layer_right if mano_side == "right" else mano_layer_left
    if ext is None:
        delta = torch.zeros((1,3),dtype=mano_betas.dtype)
    else:
        root_j = cal_root_j(mano_betas.unsqueeze(0), manopth_mano_layer)
        # 这里的 ext[:3,:3] 必须是 World -> Camera 的旋转矩阵 R，配合平移才能完美抵消 J0 偏移
        delta = -root_j + torch.tensor(ext[:3,:3],dtype=root_j.dtype) @ root_j

    vertices, joints = manopth_mano_layer(
        mano_pose[:,:48],
        mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
        mano_pose[:,48:51]+delta.reshape(1,3)
    )
    vertices = vertices / 1000  
    faces_m = manopth_mano_layer.th_faces
    return vertices, faces_m, joints

def visualize_joints_in_rgb(images, intrinsics, mano_pose, mano_betas, mano_side, out_dir:str = "tmp", ind=0, ext=None, joints=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if joints is None:
        vertices, faces, joint = get_vertices_faces(mano_pose, mano_betas, mano_side, ext)
        vertices = vertices[0].numpy()
        breakpoint()
        render_rgb, rend_depth, render_mask = render_joint(vertices, faces.numpy(), intrinsics, h=images.shape[0], w=images.shape[1])
        composited = np.where(render_mask[:, :, None], render_rgb, images)
        composited = np.concatenate([images, composited], axis=1)
        iio.imwrite(os.path.join(out_dir, f"{ind}_{mano_side}.jpg"), composited)
    else:
        _,_,joi_ = get_vertices_faces(mano_pose, mano_betas, mano_side, ext)
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
        iio.imwrite(os.path.join(out_dir, f"{ind}_{mano_side}_joints.jpg"), composited)

@torch.no_grad()
def pca15_to_axis45(coeffs15: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    basis = layer.th_selected_comps  
    mean  = layer.th_hands_mean      
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)  
    if basis.shape == (coeffs15.shape[-1], 45):     
        pose45 = coeffs15 @ basis                   
    elif basis.shape == (45, coeffs15.shape[-1]):   
        pose45 = coeffs15 @ basis.t()               
    else:
        raise ValueError(f"Unexpected basis shape: {basis.shape}")
    pose45 = pose45 + mean                          
    return pose45

def add_mean_pose45(coeffs45: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    mean = layer.th_hands_mean
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    return coeffs45 + mean

def read_gigahand_cameras(params_path: Path, target_cam_name: str):
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ],
    )
    for param in params:
        if param["cam_name"] == target_cam_name:
            R = Rotation.from_quat([param['qvecx'], param['qvecy'], param['qvecz'], param['qvecw']]).as_matrix()
            T = np.array([param['tvecx'], param['tvecy'], param['tvecz']])
            intrinsics = np.array([param['fx'], param['fy'], param['cx'], param['cy']])
            orig_w = param['width']
            orig_h = param['height']
            return R, T, intrinsics, orig_w, orig_h
    raise ValueError(f"Camera {target_cam_name} not found.")

def construct_mano_pose(param_dict, mano_layer, world2cam_rot, world2cam_trans):
    Rh = np.array(param_dict['Rh']).reshape(3)
    Th = np.array(param_dict['Th']).reshape(3, 1)
    poses = np.array(param_dict['poses']).reshape(-1)
    shapes = torch.FloatTensor(param_dict['shapes']).view(-1)

    mano_rot, _ = cv2.Rodrigues(Rh)
    mano_rot = torch.FloatTensor(mano_rot)
    mano_trans = torch.FloatTensor(Th)

    # 完美对齐你 test_interhand 的矩阵推导：R_c2m = R_w2c @ R_m2w, T_c2m = T_w2c + R_w2c @ T_m2w
    cam2mano_rot = world2cam_rot @ mano_rot
    cam2mano_trans = world2cam_trans + world2cam_rot @ mano_trans

    mano_rot_angle, _ = cv2.Rodrigues(cam2mano_rot.numpy())
    mano_rot_angle = torch.FloatTensor(mano_rot_angle.reshape(3))

    mano_pose = torch.zeros((1, 51))
    mano_pose[0, :3] = mano_rot_angle
    finger_pose = torch.FloatTensor(poses[3:48]).view(1, -1)
    # mano_pose[0, 3:48] = add_mean_pose45(finger_pose, mano_layer)
    mano_pose[0, 3:48] = finger_pose
    mano_pose[0, 48:51] = cam2mano_trans.reshape(3)
    root_j = cal_root_j(shapes.unsqueeze(0), mano_layer)
    delta = -root_j + cam2mano_rot @ root_j
    mano_pose[0, 48:51] += delta.reshape(3)

    return mano_pose, shapes

if __name__ == '__main__':
    mano_model_folder = "/public/home/annie/code/GigaHands/body_models/smplh"
    dataset_root = Path('/data/annie/gigahand/')  
    session_name = "p003-instrument"
    seq_id = 33
    render_camera = "brics-odroid-011_cam0"
    
    mano_layer_right = ManoLayer(flat_hand_mean=False, ncomps=45, use_pca=False, side="right", mano_root=mano_model_folder)
    mano_layer_left = ManoLayer(flat_hand_mean=False, ncomps=45, use_pca=False, side="left", mano_root=mano_model_folder)
    
    camera_video_map_csv = dataset_root / "multiview_camera_video_map.csv"
    hand_pose_path = dataset_root / "hand_poses" / session_name
    keypoints3d_path = hand_pose_path / "keypoints_3d" / f"{int(seq_id):03d}"
    mano_main_path = hand_pose_path / "params" / f"{int(seq_id):03d}.json"
    camera_params_path = hand_pose_path / "optim_params.txt"
    img_dir = dataset_root / "multiview_rgb_vids6" / session_name / render_camera / f"{int(seq_id):03d}"
    
    save_path = Path("tmp_gigahand") / f"{session_name}_{seq_id}_{render_camera}"
    save_path.mkdir(parents=True, exist_ok=True)

    R, T, intrinsics, orig_w, orig_h = read_gigahand_cameras(camera_params_path, render_camera)
    
    world2cam_rot = torch.FloatTensor(R)
    world2cam_trans = torch.FloatTensor(T.reshape(3, 1))
    
    extrinsics_matrix = torch.zeros((4,4))
    extrinsics_matrix[:3,:3] = world2cam_rot
    extrinsics_matrix[:3,3] = world2cam_trans.reshape(3)

    df = pd.read_csv(camera_video_map_csv)
    rgb_path=df.loc[(df['scene'] == 'p003-instrument') & (df['sequence'] == 33), 'brics-odroid-011_cam0'].item()
    find_path = []
    for view_path_id in range(10):
        video_file = dataset_root / f"multiview_rgb_vids{view_path_id}" / session_name / rgb_path
        if video_file.exists():
            find_path.append(video_file)
    assert len(find_path) == 1, f"Expected to find exactly one video file for camera {render_camera}, but found {len(find_path)}: {find_path}"
    find_path = find_path[0]
    print(f"Camera {render_camera} corresponds to video file: {video_file}")

    with open(keypoints3d_path / "chosen_frames_left.json", "r") as f:
        chosen_frames_left = set(json.load(f))
    with open(keypoints3d_path / "chosen_frames_right.json", "r") as f:
        chosen_frames_right = set(json.load(f))
    
    chosen_hand_union_frames = list(chosen_frames_right | chosen_frames_left)
    chosen_hand_intersect_frames = list(chosen_frames_right & chosen_frames_left)
    valid_frames = sorted(chosen_hand_intersect_frames)

    with open(mano_main_path, 'r') as f:
        manos_params = json.load(f)

    print(f"Total valid frames to render: {len(valid_frames)}")

    for nf in tqdm(valid_frames, desc="Rendering GigaHand Frames"):
        try:
            img = iio.imread(find_path, index=nf)
        except Exception as e:
            print(f"[Warning] 读取视频帧失败，跳过帧 {nf}: {e}")
            continue
        img_h, img_w = img.shape[:2]  
        try:
            rel_idx = chosen_hand_union_frames.index(nf)
            param_right = {k: v[rel_idx] for k, v in manos_params['right'].items() if k in ['Rh', 'Th', 'poses']}
            param_right['shapes'] = manos_params['right']['shapes']
            param_left = {k: v[rel_idx] for k, v in manos_params['left'].items() if k in ['Rh', 'Th', 'poses']}
            param_left['shapes'] = manos_params['left']['shapes']
        except Exception as e:
            breakpoint()
            continue

        mano_pose_r, mano_betas_r = construct_mano_pose(param_right, mano_layer_right, world2cam_rot, world2cam_trans)
        mano_pose_l, mano_betas_l = construct_mano_pose(param_left, mano_layer_left, world2cam_rot, world2cam_trans)
        # breakpoint()
        mano_betas_r_test = torch.zeros_like(mano_betas_r)
        visualize_joints_in_rgb(img, intrinsics, mano_pose_r, mano_betas_r_test, "right", out_dir=str(save_path), ind=f"{nf}_right")
        visualize_joints_in_rgb(img, intrinsics, mano_pose_l, mano_betas_l, "left", out_dir=str(save_path), ind=f"{nf}_left")