import sys

from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import json

import imageio.v3 as iio
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
# from hamer_helper import HamerHelper
# from .dataclass import HandTrainingData
from egoallo.data.dataclass import HandTrainingData
# from hamer.utils.mesh_renderer import create_raymond_lights
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
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

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
    # breakpoint()
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




# from hamer_helper import HamerHelper, HandOutputsWrtCamera
# helper = HamerHelper()
# def composite_detections(
#     image: np.ndarray,
#     detections: HandOutputsWrtCamera | None,
#     h,w,
#     border_color: tuple[int, int, int],
# ) -> np.ndarray:
#     if detections is None:
#         return image

#     for index in range(detections["verts"].shape[0]):
#         print(index)
#         render_rgb, _, render_mask = helper.render_detection(
#             detections, hand_index=0, h=h, w=w, focal_length=None
#         )
#         border_width = 15
#         image = np.where(
#             binary_dilation(
#                 render_mask, np.ones((border_width, border_width), dtype=bool)
#             )[:, :, None],
#             np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
#             image,
#         )
#         image = np.where(render_mask[:, :, None], render_rgb, image)

#     return image


# def put_text(
#     image: np.ndarray,
#     text: str,
#     line_number: int,
#     color: tuple[int, int, int],
#     font_scale: float = 10.0,
# ) -> np.ndarray:
#     image = image.copy()
#     font = cv2.FONT_HERSHEY_PLAIN  # type: ignore
#     cv2.putText(  # type: ignore
#         image,
#         text=text,
#         org=(2, 1 + int(15 * font_scale * (line_number + 1))),
#         fontFace=font,
#         fontScale=font_scale,
#         color=(0, 0, 0),
#         thickness=int(font_scale),
#         lineType=cv2.LINE_AA,  # type: ignore
#     )
#     cv2.putText(  # type: ignore
#         image,
#         text=text,
#         org=(2, 1 + int(15 * font_scale * (line_number + 1))),
#         fontFace=font,
#         fontScale=font_scale,
#         color=color,
#         thickness=int(font_scale),
#         lineType=cv2.LINE_AA,  # type: ignore
#     )
#     return image


class HandHdf5EachDataset(torch.utils.data.Dataset[HandTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file.

    Args:
        dataset_name: Name of the dataset to load. Options are "dexycb", "interhand26m", "arctic" and "ho3d".
        splits: "train", "val" or "test" splits to include in the dataset. For ho3d,
            use "train" and "evaluation".
        vis: Whether to load RGB frames for visualization.
        subseq_len: Length of subsequences to sample from the dataset.
        clip_stride: Stride between subsequences.
        min_len: Minimum length of subsequences to include.
    """

    def __init__(
        self,
        dataset_name: str = "dexycb", # dexycb, interhand26m, arctic, ho3d
        split: Literal["train", "val", "test", "evaluation"] = "train",
        vis: bool = False,
        subseq_len: int = 64,
        clip_stride: int = 8,
        min_len: int = 32,
    ) -> None:
        if dataset_name=="dexycb":
            self._hdf5_path = "/public/datasets/handdata/dexycb_v6.hdf5"
            self.video_root = "/public/datasets/handdata/dexycb/videos_v4"
        elif dataset_name=="interhand26m":
            self._hdf5_path = "/public/datasets/handdata/interhand26m_v4.hdf5"
            self.video_root = "/public/datasets/handdata/interhand26m/data/picked_videos"
        elif dataset_name=="arctic":
            self._hdf5_path = "/public/datasets/handdata/arctic_v8.hdf5"
            self.video_root = "/public/datasets/handdata/arctic/picked_videos"
        elif dataset_name == "ho3d":
            self.video_root = "/public/datasets/handdata/HO3D_v3_new/picked_videos_v2"
            if split == "train":
                self._hdf5_path = "/public/datasets/handdata/ho3d_v3.hdf5"
            elif split == "evaluation":
                self._hdf5_path = "/public/datasets/handdata/ho3d_evaluation_v7.hdf5"
            else:
                raise ValueError("For ho3d, split should be train or evaluation")
        else:
            raise ValueError("dataset_name should be dexycb, interhand26m, arctic or ho3d")
        self.split = split
        self.dataset_name = dataset_name
        self.vis = vis
        self._clip_stride = clip_stride
        self._subseq_len = subseq_len
        self._min_len = min_len
        json_dir = self._hdf5_path.replace('.hdf5','_json')
        json_file_path = os.path.join(json_dir, f"{split}.json")
        self._mapping = []
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        for group_name in json_data.keys():
            group_length = json_data[group_name]
            if self._subseq_len == -1:
                self._mapping.append((group_name, 0, group_length))
            else:
                if group_length < self._min_len:
                    continue
                if group_length <= self._subseq_len:
                    self._mapping.append((group_name, 0, group_length)) # group_name, start_idx, len
                else:
                    start_idx = 0
                    while start_idx + self._subseq_len < group_length:
                        self._mapping.append((group_name, start_idx, self._subseq_len))
                        start_idx += self._clip_stride
                    # last clip
                    self._mapping.append((group_name, group_length - self._subseq_len, self._subseq_len))

        self.N = len(self._mapping)
        self.left_mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='left',
            mano_root="/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano",
        )
        self.right_mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='right',
            mano_root="/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano",
        )
        
    def __getitem__(self, index: int,resize=(512,512)) -> HandTrainingData:
        kwargs: dict[str, Any] = {}
        group_name, start_idx, clip_len = self._mapping[index]
        if self._subseq_len == -1:
            pad_len = 0
        else:
            pad_len = self._subseq_len - clip_len
        with h5py.File(self._hdf5_path, "r") as f:
            dataset = f[group_name]
            kwargs["mano_betas"] = torch.from_numpy(dataset['mano_betas'][:])
            kwargs["mano_pose"]=torch.from_numpy(dataset['mano_poses'][start_idx:start_idx+clip_len,0])
            kwargs["extrinsics"] =torch.from_numpy(dataset['extrinsics'][:])
                
            kwargs["mano_joint_3d"] = torch.from_numpy(dataset['mano_joint_3d'][start_idx:start_idx+clip_len,0])
            # if self.dataset_name=="arctic":
            #     kwargs["mano_joint_3d"] = kwargs["mano_joint_3d"] / 1000  # Convert to meters
            kwargs["intrinsics"]= torch.from_numpy(dataset['intrinsics'][:])
            if len(kwargs["intrinsics"].shape) == 1:
                if self._subseq_len !=-1: 
                    kwargs["intrinsics"] = kwargs["intrinsics"].unsqueeze(0).expand(self._subseq_len, -1)
                else:
                    kwargs["intrinsics"] = kwargs["intrinsics"].unsqueeze(0).expand(clip_len, -1)
            else:
                kwargs["intrinsics"] = kwargs["intrinsics"][start_idx:start_idx+clip_len]
                if self._subseq_len != -1: 
                    kwargs["intrinsics"] = torch.cat([kwargs["intrinsics"], torch.zeros((pad_len, kwargs["intrinsics"].shape[1]))], dim=0)
            if dataset['mano_side'][()].decode('utf-8') == 'left':
                kwargs["mano_side"] = torch.zeros(1)
            else:
                kwargs["mano_side"] = torch.ones(1)
            # kwargs["mask"] = torch.from_numpy(dataset['mask'][index]).type(torch.bool)
            kwargs["mask"] = torch.ones((clip_len), dtype=torch.bool)
            # pad if needed
            if pad_len > 0:
                kwargs["mano_pose"] = torch.cat([kwargs["mano_pose"], torch.zeros((pad_len, *kwargs["mano_pose"].shape[1:]))], dim=0)
                kwargs["mano_joint_3d"] = torch.cat([kwargs["mano_joint_3d"], torch.zeros((pad_len, *kwargs["mano_joint_3d"].shape[1:]))], dim=0)
                kwargs['mask'] = torch.cat([kwargs['mask'], torch.zeros((pad_len), dtype=torch.bool)], dim=0)
            video_name = dataset['video_name'][()].decode('utf-8')
            if not os.path.exists(os.path.join(self.video_root, self.split, video_name)):
                print(f"Video not found: {os.path.join(self.video_root, self.split, video_name)}")
            if self.vis:
                video_path = os.path.join(self.video_root, self.split, video_name)
                reader = iio.imread(video_path)
                rgb_frames = []
                for i in range(start_idx, start_idx + clip_len):
                    frame = reader[i]
                    rgb_frames.append(torch.from_numpy(frame))
                if pad_len > 0:
                    rgb_frames.extend([torch.zeros_like(rgb_frames[0])] * pad_len)
                kwargs["rgb_frames"] = torch.stack(rgb_frames, dim=0)
                # resize
                if resize is not None:
                    resized_frames = []
                    intrinsics = kwargs["intrinsics"].clone()
                    for i in range(kwargs["rgb_frames"].shape[0]):
                        frame = kwargs["rgb_frames"][i].numpy().astype(np.uint8)
                        frame_resized = cv2.resize(frame, resize)
                        resized_frames.append(torch.from_numpy(frame_resized))
                    intrinsics[:, 0] = intrinsics[:, 0] * resize[0] / kwargs["rgb_frames"].shape[2]
                    intrinsics[:, 1] = intrinsics[:, 1] * resize[1] / kwargs["rgb_frames"].shape[1]
                    intrinsics[:, 2] = intrinsics[:, 2] * resize[0] / kwargs["rgb_frames"].shape[2]
                    intrinsics[:, 3] = intrinsics[:, 3] * resize[1] / kwargs["rgb_frames"].shape[1]
                    kwargs["intrinsics"] = intrinsics
                    kwargs["rgb_frames"] = torch.stack(resized_frames, dim=0)
            else:
                if self._subseq_len == -1:
                    kwargs["rgb_frames"] = torch.ones((clip_len), dtype=torch.uint8)
                else:
                    kwargs["rgb_frames"] = torch.ones((self._subseq_len), dtype=torch.uint8)

            if self._subseq_len == -1:
                kwargs['img_feature'] = torch.zeros((clip_len, 1280))
            else:
                kwargs['img_feature'] = torch.zeros((self._subseq_len, 1280))
            kwargs['img_feature'] = kwargs['img_feature'].cpu()
        return HandTrainingData(**kwargs)
    
    def __len__(self) -> int:
        return self.N

    def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp",resize=None,from_mano=False) -> None:
        os.makedirs(out_dir, exist_ok=True)
        sample = self.__getitem__(index, resize=resize)
        intrinsics = sample.intrinsics.numpy()
        border_color = [255, 0, 0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
        out = cv2.VideoWriter(out_dir+"/gt_joints.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))

        if from_mano:
            _, mano_joint_3d = self.get_vertices_joints(index)
        else:
            mano_joint_3d = sample.mano_joint_3d
        projected_joints = np.zeros((mano_joint_3d.shape[0],mano_joint_3d.shape[1], 2))
        # if self.dataset_name == "arctic":
        intrinsics = torch.tensor(intrinsics)
        # (64,1)*(64,21) -> (64,21)
        projected_joints[...,0] = intrinsics[:,0:1] * (mano_joint_3d[...,0] / mano_joint_3d[...,2]) + intrinsics[:,2:3]
        projected_joints[...,1] = intrinsics[:,1:2] * (mano_joint_3d[...,1] / mano_joint_3d[...,2]) + intrinsics[:,3:4]
        # else:
        #     projected_joints[...,0] = intrinsics[0] * (mano_joint_3d[...,0] / mano_joint_3d[...,2]) + intrinsics[2]
        #     projected_joints[...,1] = intrinsics[1] * (mano_joint_3d[...,1] / mano_joint_3d[...,2]) + intrinsics[3]
        # projected_joints = projected_joints[:, :11, :]  # only visualize first 11 joints
        for i in tqdm(range(sample.mask.sum().item())):
            image = sample.rgb_frames[i].numpy().astype(np.uint8)
            image = image[:,:,::-1]
            composited = image.copy()
            for j in range(projected_joints.shape[1]):
                # the color of the circle depends on the joint index
                cv2.circle(composited, (int(projected_joints[i,j,0]), int(projected_joints[i,j,1])), 3, (int(255 * j / projected_joints.shape[1]), 0, int(255 * (projected_joints.shape[1] - j) / projected_joints.shape[1])), -1)
            composited = np.concatenate([image, composited], axis=1)
            out.write(composited)
        out.release()

    def visualize_manos_in_rgb(self, index: int,out_dir:str = "tmp",resize=None,debug=False) -> None:
        os.makedirs(out_dir, exist_ok=True)
        mano_vertices, _ = self.get_vertices_joints(index)
        # if debug:
        #     verts_json = "/public/home/group_ucb/yunqili/.cache/huggingface/hub/datasets--AnnieLiyunqi--verts_ho3d/snapshots/103cbce86985aa35997f18496f1cd3b02ede0fea/evaluation_verts.json"
        #     with open(verts_json, 'r') as f:
        #         verts_data = json.load(f)
        #     verts_data = np.array(verts_data)
        #     verts_data[:,:,1] *= -1  # flip y axis
        #     verts_data[:,:,2] *= -1  # flip z axis
        #     vertices = torch.tensor(verts_data[659:659+64])  # (T, 778, 3)
        # else:
        vertices = mano_vertices  # (T, 778, 3)
        sample = self.__getitem__(index, resize=resize)
        
        faces = self.get_mano_faces(mano_side="right" if sample.mano_side.item()==1 else "left")
        intrinsics = sample.intrinsics.numpy()
        border_color = [255, 0, 0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
        out = cv2.VideoWriter(out_dir+"/gt.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))
        for i in tqdm(range(sample.mask.sum().item())):
            image = sample.rgb_frames[i].numpy().astype(np.uint8)
            image = image[:,:,::-1]
            # if self.dataset_name == "arctic":
            intr = intrinsics[i]
            # else:
            #     intr = intrinsics
            render_rgb, rend_depth, render_mask = render_joint(vertices[i].numpy(), faces.numpy(),
                                                               intr, h=sample.rgb_frames.shape[1], w=sample.rgb_frames.shape[2])
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
            #iio.imwrite(os.path.join(out_dir, f"{i:03d}.jpg"), composited)
            out.write(composited)
        out.release()

    def get_vertices_joints(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.__getitem__(index)
        if sample.mano_side.item() == 0:
            vertices, joints = self.left_mano_layer(
                sample.mano_pose[:,:48],
                sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
                sample.mano_pose[:,48:51]
            )
        elif sample.mano_side.item() == 1:
            vertices, joints = self.right_mano_layer(
                sample.mano_pose[:,:48],
                sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
                sample.mano_pose[:,48:51]
            )
        else:
            print("Error: mano_side should be 0 or 1")
            return None
        vertices = vertices/ 1000  # Convert to meters
        joints = joints / 1000  # Convert to meters
        # print((joints - sample.mano_joint_3d)[0])
        
        return vertices, joints
    
    def get_mano_faces(self,mano_side='left'):
        if mano_side=='right':
            return self.right_mano_layer.th_faces
        elif mano_side=='left':
            return self.left_mano_layer.th_faces
        else:
            print("Error: mano_side should be 'left' or 'right'")
            return None
    
    # def hamer_output(self,index):
    #     sample = self.__getitem__(index, resize=None)
    #     hamer_helper = HamerHelper()

    #     hamer_output_seq = []
    #     print("Start calculating joints from hamer")
    #     start_time = time.time()
    #     for i in range(self._subseq_len):
    #         hamer_out_frame = {}            
    #         hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
    #             sample.rgb_frames[i].numpy().astype(np.uint8),
    #             focal_length=sample.intrinsics[0].item(),
    #         )

    #         if hamer_out_left is None:
    #             hamer_out_frame["left"] = None
    #         else:
    #             hamer_out_frame["left"] = {
    #                 "verts": hamer_out_left["verts"],
    #                 "keypoints_3d": hamer_out_left["keypoints_3d"],
    #                 "mano_poses": hamer_out_left["mano_hand_pose"],
    #                 "mano_betas": hamer_out_left["mano_hand_betas"],
    #                 "global_orientation": hamer_out_left["mano_hand_global_orient"],
    #                 "camera_pose": hamer_out_left["camera_pose"],
    #             }

    #         if hamer_out_right is None:
    #             hamer_out_frame["right"] = None
    #         else:
    #             hamer_out_frame["right"] = {
    #                 "verts": hamer_out_right["verts"],
    #                 "keypoints_3d": hamer_out_right["keypoints_3d"],
    #                 "mano_poses": hamer_out_right["mano_hand_pose"],
    #                 "mano_betas": hamer_out_right["mano_hand_betas"],
    #                 "global_orientation": hamer_out_right["mano_hand_global_orient"],
    #                 "camera_pose": hamer_out_right["camera_pose"],
    #             }

    #         hamer_output_seq.append(hamer_out_frame)        
    #     print(f"Hamer output time: {time.time() - start_time:.2f} seconds")
    #     return hamer_output_seq
    
    # def visualize_hamer_output(self,sample_index:int,out_dir:str="tmp_hamer"):
    #     os.makedirs(out_dir, exist_ok=True)
    #     sample = self.__getitem__(sample_index, resize=None)
    #     hamer_helper = HamerHelper()
    #     hamer_out_frams = []
    #     for i in range(self._subseq_len):   
    #         rgb_image = sample.rgb_frames[i].numpy().astype(np.uint8)      
    #         hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
    #             rgb_image,
    #             focal_length=sample.intrinsics[0].item(),
    #         )
    #         h, w = rgb_image.shape[0], rgb_image.shape[1]

            

    #         composited = rgb_image
            
    #         intrinsics = sample.intrinsics.numpy()
    #         if hamer_out_left is not None:
    #             for index in range(hamer_out_left["verts"].shape[0]):
    #                 vertices = hamer_out_left["verts"][index]
    #                 faces = hamer_out_left["faces"]
    #                 render_rgb, rend_depth, render_mask = render_joint(vertices, faces,
    #                                                             intrinsics, h=sample.rgb_frames.shape[1], w=sample.rgb_frames.shape[2])
    #                 border_width = 10
    #                 border_color = (255, 100, 100)
    #                 composited = np.where(
    #                     binary_dilation(
    #                         render_mask, np.ones((border_width, border_width), dtype=bool)
    #                     )[:, :, None],
    #                     np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
    #                     composited,
    #                 )
    #                 composited = np.where(render_mask[:, :, None], render_rgb, composited)
    #         if hamer_out_right is not None:
    #             for index in range(hamer_out_right["verts"].shape[0]):
    #                 vertices = hamer_out_right["verts"][index]
    #                 faces = hamer_out_right["faces"]
    #                 render_rgb, rend_depth, render_mask = render_joint(vertices, faces,
    #                                                             intrinsics, h=sample.rgb_frames.shape[1], w=sample.rgb_frames.shape[2])
    #                 border_width = 10
    #                 border_color = (100, 100, 255)
    #                 composited = np.where(
    #                     binary_dilation(
    #                         render_mask, np.ones((border_width, border_width), dtype=bool)
    #                     )[:, :, None],
    #                     np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
    #                     composited,
    #                 )
    #                 composited = np.where(render_mask[:, :, None], render_rgb, composited)

    #         hamer_out_frams.append(np.concatenate([rgb_image, composited], axis=1))

    #     # iio write video
    #     iio.imwrite(os.path.join(out_dir, f"hamer_output_{sample_index}.mp4"), hamer_out_frams, fps=30, macro_block_size=None)


from torch.utils.data import ConcatDataset

class HandHdf5Dataset(torch.utils.data.Dataset[HandTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file.

    Args:
        split: "train", "val" or "test" splits to include in the dataset. For ho3d,
            use "train" and "evaluation".
        dataset_name: Name of the dataset to load. Options are "all", "dexycb", "interhand26m", "arctic" and "ho3d".
        vis: Whether to load RGB frames for visualization.
        subseq_len: Length of subsequences to sample from the dataset.
        clip_stride: Stride between subsequences.
        min_len: Minimum length of subsequences to include.
    """
    def __init__(self,split: Literal["train", "val", "test"] = "train", dataset_name: Literal["all","dexycb", "arctic", "interhand26m", "ho3d"] = 'all', vis=False, subseq_len: int = 64, clip_stride: int = 8,min_len:int=32) -> None:
        if dataset_name == "all":
            dataset_ih26 = HandHdf5EachDataset(split=split, dataset_name="interhand26m", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len)
            dataset_dexycb = HandHdf5EachDataset(split=split, dataset_name="dexycb", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len)
            dataset_arctic = HandHdf5EachDataset(split=split, dataset_name="arctic", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len)
            if split == "train":
                dataset_ho3d = HandHdf5EachDataset(split=split, dataset_name="ho3d", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len)
                self.dataset = ConcatDataset([dataset_ih26, dataset_dexycb, dataset_arctic, dataset_ho3d])
            else:
                self.dataset = ConcatDataset([dataset_ih26, dataset_dexycb, dataset_arctic])
        else:
            self.dataset = ConcatDataset([HandHdf5EachDataset(split=split, dataset_name=dataset_name, vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len)])
        

    def __getitem__(self, index: int,resize=(512,512)) -> HandTrainingData:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.__getitem__(index,resize=resize)
            else:
                index -= len(ds)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_vertices_joints(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.get_vertices_joints(index)
            else:
                index -= len(ds)
    def get_mano_faces(self,mano_side='left'):
        for ds in self.dataset.datasets:
            return ds.get_mano_faces(mano_side=mano_side)
        
    def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp",resize=None,from_mano=False) -> None:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.visualize_joints_in_rgb(index,out_dir=out_dir,resize=resize,from_mano=from_mano)
            else:
                index -= len(ds)
    def visualize_manos_in_rgb(self, index: int,out_dir:str = "tmp",resize=None,debug=False) -> None:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.visualize_manos_in_rgb(index,out_dir=out_dir,resize=resize,debug=debug)
            else:
                index -= len(ds)

    # def hamer_output(self,index):
    #     for ds in self.dataset.datasets:
    #         if index < len(ds):
    #             return ds.hamer_output(index)
    #         else:
    #             index -= len(ds)
    
    # def visualize_hamer_output(self,index:int,out_dir:str="tmp_hamer"):
    #     for ds in self.dataset.datasets:
    #         if index < len(ds):
    #             return ds.visualize_hamer_output(index,out_dir=out_dir)
    #         else:
    #             index -= len(ds)
    
if __name__ == "__main__":    
    # dataset = HandHdf5Dataset(split='evaluation', dataset_name='ho3d', vis=True, subseq_len=64, clip_stride=4, min_len=32)
    # dataset.visualize_joints_in_rgb(10, out_dir="tmp_ho3d_joints", resize=None, from_mano=True)
    # dataset.visualize_manos_in_rgb(10, out_dir="tmp_ho3d_manos", resize=None)
    # dataset.visualize_joints_in_rgb(10, out_dir="tmp_ho3d_joints", resize=(512,512))
    # dataset.visualize_manos_in_rgb(10, out_dir="tmp_ho3d_manos", resize=(512,512))

    # dataset = HandHdf5Dataset(split='test', dataset_name='interhand26m', vis=True, subseq_len=64, clip_stride=4, min_len=32)
    # dataset.visualize_joints_in_rgb(10, out_dir="tmp_interhand_joints", resize=None)
    # dataset.visualize_manos_in_rgb(10, out_dir="tmp_interhand_manos", resize=None)

    dataset = HandHdf5Dataset(split='test', dataset_name='arctic', vis=True,subseq_len=64, clip_stride=4, min_len=32)
    dataset.visualize_joints_in_rgb(11, out_dir="tmp_arctic0_joints", resize=None)
    dataset.visualize_manos_in_rgb(11, out_dir="tmp_arctic0_manos", resize=(512,512))


    # dataset = HandHdf5Dataset(split='train', dataset_name='dexycb', vis=True, subseq_len=64, clip_stride=4)
    # dataset.visualize_joints_in_rgb(10, out_dir="tmp_dexycb_joints", resize=None)
    # dataset.visualize_manos_in_rgb(10, out_dir="tmp_dexycb_manos", resize=None)

    # less =0
    # more =0
    for i in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(i, resize=None)
        # if less ==0 and sample.mask.sum().item() < 64:
        #     dataset.visualize_joints_in_rgb(i, out_dir="tmp_dexycb_joints", resize=None)
        #     dataset.visualize_manos_in_rgb(i, out_dir="tmp_dexycb_manos", resize=None)
        #     less +=1
        #     if more >0:
        #         break
        # if more ==0 and sample.mask.sum().item() > 64:
        #     dataset.visualize_joints_in_rgb(i, out_dir="tmp_dexycb_joints_more", resize=None)
        #     dataset.visualize_manos_in_rgb(i, out_dir="tmp_dexycb_manos_more", resize=None)
        #     more +=1
        #     if less >0:
        #         break
