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

import imageio.v3 as iio
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from hamer_helper import HamerHelper
# from .dataclass import HandTrainingData
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

# def composite_detections(
#     image: np.ndarray,
#     detections: HandOutputsWrtCamera | None,
#     border_color: tuple[int, int, int],
# ) -> np.ndarray:
#     render_rgb, _, render_mask = render_detection(
#         detections, hand_index=0, h=h, w=w, focal_length=None
#     )
#     border_width = 15
#     image = np.where(
#         binary_dilation(
#             render_mask, np.ones((border_width, border_width), dtype=bool)
#         )[:, :, None],
#         np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
#         image,
#     )
#     image = np.where(render_mask[:, :, None], render_rgb, image)

#     return image


class HandHdf5EachDataset(torch.utils.data.Dataset[HandTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file.

    Args:
        hdf5_path: Path to the HDF5 file containing the dataset.
        splits: "train", "val" or "test" splits to include in the dataset.
        subseq_len: Length of subsequences to sample from the dataset.
        cache_files: Whether to cache the entire dataset in memory.
        deterministic_slices: Set to True to always use the same slices. This
            is useful for reproducible eval.
    """

    def __init__(
        self,
        dataset_name: str = "dexycb", # dexycb, interhand26m, arctic
        # hdf5_path: Path="/public/datasets/handdata/dexycb_v2.hdf5", # 
        split: Literal["train", "val", "test"] = "train",
        vis: bool = False,
        # file_list_path: Path,
        # splits: tuple[
        #     Literal["train", "val", "test"], ...
        # ],
        # slice_strategy: Literal[
        #     "deterministic", "random_uniform_len", "random_variable_len"
        # ] = "deterministic",
        # subseq_len: int=64, # Length of subsequences to sample.
        # cache_files: bool,
        # min_subseq_len: int | None = None,
        # random_variable_len_proportion: float = 0.3,
        # random_variable_len_min: int = 16,
    ) -> None:
        if dataset_name=="dexycb":
            self._hdf5_path = "/public/datasets/handdata/dexycb_v5.hdf5"
            self.video_root = "/public/datasets/handdata/dexycb/videos_v4"
            self.img_feat_root = "/public/datasets/handdata/dexycb/img_feats"
        elif dataset_name=="interhand26m":
            self._hdf5_path = "/public/datasets/handdata/interhand26m_v3.hdf5"
            self.video_root = "/public/datasets/handdata/interhand26m/data/picked_videos"
        elif dataset_name=="arctic":
            self._hdf5_path = "/public/datasets/handdata/arctic_v3.hdf5"
            self.video_root = "/public/datasets/handdata/arctic/picked_videos"
        else:
            raise ValueError("dataset_name should be dexycb, interhand26m or arctic")
        # self.rgb_format = "color_{:06d}.jpg"
        self.split = split
        self.dataset_name = dataset_name
        self.vis = vis
        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            # ]
            dataset = hdf5_file[self.split]
            self._subseq_len = 64  # Default subsequence length, can be overridden.
            self.N = dataset['mano_side'].shape[0]
            # self.keys = list(dataset.keys())
        # self.left_mano_layer = ManoLayer(
        #     use_pca=False,
        #     flat_hand_mean=True,
        #     ncomps=45,
        #     side='left',
        #     mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        # )
        # self.right_mano_layer = ManoLayer(
        #     use_pca=False,
        #     flat_hand_mean=True,
        #     ncomps=45,
        #     side='right',
        #     mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        # )
        
    def __getitem__(self, index: int,resize=(512,512)) -> HandTrainingData:
        kwargs: dict[str, Any] = {}

        with h5py.File(self._hdf5_path, "r") as f:
            dataset = f[self.split]
            # ndarray to tensor
            kwargs["mano_betas"]=torch.from_numpy(dataset['mano_betas'][index])
            kwargs["mano_pose"]=torch.from_numpy(dataset['mano_poses'][index,:,0])
            timesteps= kwargs["mano_pose"].shape[0]
            kwargs["mano_joint_3d"] = torch.from_numpy(dataset['mano_joint_3d'][index,:,0])
            if self.dataset_name=="arctic":
                kwargs["mano_joint_3d"] = kwargs["mano_joint_3d"] / 1000  # Convert to meters
            kwargs["intrinsics"]= torch.from_numpy(dataset['intrinsics'][index])
            kwargs["extrinsics"] =torch.from_numpy(dataset['extrinsics'][index])
            if dataset['mano_side'][index][0].decode('utf-8') == 'left':
                kwargs["mano_side"] = torch.zeros(1)
            else:
                kwargs["mano_side"] = torch.ones(1)
            
            kwargs["mask"] = torch.from_numpy(dataset['mask'][index]).type(torch.bool)
            assert os.path.exists(os.path.join(self.video_root, self.split, dataset['video_name'][index][0].decode('utf-8'))), f"Video not found: {os.path.join(self.video_root, self.split, dataset['video_name'][index][0].decode('utf-8'))}"
            # print("Video name: ", dataset['video_name'][index][0].decode('utf-8'))
            if self.vis:
                if 'rgb_frames' in dataset:
                    kwargs["rgb_frames"] = torch.tensor(dataset['rgb_frames'][index].transpose(0,2,3,1))
                elif 'video_name' in dataset:
                    video_name = dataset['video_name'][index][0].decode('utf-8')
                    # video_path = os.path.join("/public/datasets/handdata/interhand26m/data/picked_videos", self.split, video_name)
                    video_path = os.path.join(self.video_root, self.split, video_name)
                    reader = iio.imread(video_path)
                    rgb_frames = []
                    for frame in reader:
                        rgb_frames.append(torch.from_numpy(frame))
                    # padding if needed
                    if len(rgb_frames) < timesteps:
                        rgb_frames.extend([torch.zeros_like(rgb_frames[0])] * (timesteps - len(rgb_frames)))
                    kwargs["rgb_frames"] = torch.stack(rgb_frames, dim=0)

                else:
                    raise ValueError("No rgb_frames or video_name in test dataset")
                # resize
                if resize is not None:
                    resized_frames = []
                    for i in range(kwargs["rgb_frames"].shape[0]):
                        frame = kwargs["rgb_frames"][i].numpy().astype(np.uint8)
                        frame_resized = cv2.resize(frame, resize)
                        resized_frames.append(torch.from_numpy(frame_resized))
                    kwargs["rgb_frames"] = torch.stack(resized_frames, dim=0)
            else:
                kwargs["rgb_frames"] = torch.ones((timesteps,), dtype=torch.bool)
            # if no mask in dataset, set mask to all ones
            kwargs['img_feature'] = torch.load(os.path.join(self.img_feat_root, self.split, f'imgfeat_{index}.pt')).squeeze(1) # T,1280
            if kwargs['img_feature'].shape[0] < timesteps:
                pad_len = timesteps - kwargs['img_feature'].shape[0]
                kwargs['img_feature'] = torch.cat([kwargs['img_feature'], torch.zeros((pad_len, kwargs['img_feature'].shape[1]))], dim=0)
        return HandTrainingData(**kwargs)
    
    def __len__(self) -> int:
        return self.N
    
    def hamer_output(self,index):
        sample = self.__getitem__(index)
        hamer_helper = HamerHelper()

        hamer_output_seq = []
        print("Start calculating joints from hamer")
        start_time = time.time()
        for i in range(self._subseq_len):
            hamer_out_frame = {}            
            hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
                sample.rgb_frames[i].numpy().astype(np.uint8),
                focal_length=sample.intrinsics[0].item(),
            )

            if hamer_out_left is None:
                hamer_out_frame["left"] = None
            else:
                hamer_out_frame["left"] = {
                    "verts": hamer_out_left["verts"],
                    "keypoints_3d": hamer_out_left["keypoints_3d"],
                    "mano_poses": hamer_out_left["mano_hand_pose"],
                    "mano_betas": hamer_out_left["mano_hand_betas"],
                    "global_orientation": hamer_out_left["mano_hand_global_orient"],
                    "camera_pose": hamer_out_left["camera_pose"],
                }

            if hamer_out_right is None:
                hamer_out_frame["right"] = None
            else:
                hamer_out_frame["right"] = {
                    "verts": hamer_out_right["verts"],
                    "keypoints_3d": hamer_out_right["keypoints_3d"],
                    "mano_poses": hamer_out_right["mano_hand_pose"],
                    "mano_betas": hamer_out_right["mano_hand_betas"],
                    "global_orientation": hamer_out_right["mano_hand_global_orient"],
                    "camera_pose": hamer_out_right["camera_pose"],
                }

            hamer_output_seq.append(hamer_out_frame)        
        print(f"Hamer output time: {time.time() - start_time:.2f} seconds")
        return hamer_output_seq
    
    # def get_vertices_faces(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    #     sample = self.__getitem__(index)
    #     if sample.mano_side.item() == 0:
    #         vertices, joints = self.left_mano_layer(
    #             sample.mano_pose[:,:48],
    #             sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
    #             sample.mano_pose[:,48:51]
    #         )
    #     elif sample.mano_side.item() == 1:
    #         vertices, joints = self.right_mano_layer(
    #             sample.mano_pose[:,:48],
    #             sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
    #             sample.mano_pose[:,48:51]
    #         )
    #     else:
    #         print("Error: mano_side should be 0 or 1")
    #         return None
    #     vertices = vertices/ 1000  # Convert to meters
        
    #     return vertices
    
    # def get_mano_faces(self,mano_side='left'):
    #     if mano_side=='right':
    #         return self.right_mano_layer.th_faces
    #     elif mano_side=='left':
    #         return self.left_mano_layer.th_faces
    #     else:
    #         print("Error: mano_side should be 'left' or 'right'")
    #         return None
        
    # def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp") -> None:
    #     os.makedirs(out_dir, exist_ok=True)
    #     sample = self.__getitem__(index)
    #     intrinsics = sample.intrinsics.numpy()
    #     border_color = [255, 0, 0]
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
    #     out = cv2.VideoWriter(out_dir+"/gt_joints.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))
    #     projected_joints = np.zeros((sample.mano_joint_3d.shape[0],sample.mano_joint_3d.shape[1], 2))
    #     projected_joints[...,0] = intrinsics[0] * (sample.mano_joint_3d[...,0] / sample.mano_joint_3d[...,2]) + intrinsics[2]
    #     projected_joints[...,1] = intrinsics[1] * (sample.mano_joint_3d[...,1] / sample.mano_joint_3d[...,2]) + intrinsics[3]
    #     for i in tqdm(range(sample.mask.sum().item())):
    #         image = sample.rgb_frames[i].numpy().astype(np.uint8)
    #         image = image[:,:,::-1]
    #         composited = image.copy()
    #                 # sample.mano_joint_3d # (batch, timesteps, 21, 3)
    #                 # to(batch, timesteps, 21, 3)
    #         for j in range(projected_joints.shape[1]):
    #             cv2.circle(composited, (int(projected_joints[i,j,0]), int(projected_joints[i,j,1])), 3, border_color, -1)
    #         composited = np.concatenate([image, composited], axis=1)
    #         #iio.imwrite(os.path.join(out_dir, f"{i:03d}.jpg"), composited)
    #         out.write(composited)
    #     out.release()

    # def visualize_manos_in_rgb(self, index: int,out_dir:str = "tmp") -> None:
    #     os.makedirs(out_dir, exist_ok=True)
    #     vertices = self.get_vertices_faces(index)
    #     faces = self.get_mano_faces()
    #     sample = self.__getitem__(index)
    #     intrinsics = sample.intrinsics.numpy()
    #     border_color = [255, 0, 0]
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
    #     out = cv2.VideoWriter(out_dir+"/gt.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))
    #     for i in range(sample.mask.sum().item()):
    #         image = sample.rgb_frames[i].numpy().astype(np.uint8)
    #         image = image[:,:,::-1]
    #         render_rgb, rend_depth, render_mask = render_joint(vertices[i].numpy(), faces.numpy(),
    #                                                            intrinsics, h=sample.rgb_frames.shape[1], w=sample.rgb_frames.shape[2])
    #         border_width = 10
    #         composited = np.where(
    #             binary_dilation(
    #                 render_mask, np.ones((border_width, border_width), dtype=bool)
    #             )[:, :, None],
    #             np.zeros_like(render_rgb) + np.array(border_color, dtype=np.uint8),
    #             image,
    #         )
    #         composited = np.where(render_mask[:, :, None], render_rgb, image)
    #         composited = np.concatenate([image, composited], axis=1)
    #         #iio.imwrite(os.path.join(out_dir, f"{i:03d}.jpg"), composited)
    #         out.write(composited)
    #     out.release()

from torch.utils.data import ConcatDataset

class HandHdf5Dataset(torch.utils.data.Dataset[HandTrainingData]):
    # concate HandHdf5EachDataset(dexycb) and HandHdf5EachDataset(interhand26m)
    # to a single dataset
    def __init__(self,split: Literal["train", "val", "test"] = "train", dataset_name: Literal["all","dexycb", "arctic", "interhand26m"] = 'all', vis=False) -> None:
        # if dataset_name is None and split == "test":
        #     dataset_dexycb = HandHdf5EachDataset(split=split, dataset_name="dexycb")
        #     self.dataset = dataset_dexycb
        if dataset_name == "all":
            dataset_ih26 = HandHdf5EachDataset(split=split, dataset_name="interhand26m", vis=vis)
            dataset_dexycb = HandHdf5EachDataset(split=split, dataset_name="dexycb", vis=vis)
            dataset_arctic = HandHdf5EachDataset(split=split, dataset_name="arctic", vis=vis)
        
            self.dataset = ConcatDataset([dataset_ih26, dataset_dexycb, dataset_arctic])
        else:
            self.dataset = HandHdf5EachDataset(split=split, dataset_name=dataset_name, vis=vis)
        
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
        return self.dataset.__getitem__(index, resize=resize)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_vertices_faces(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        
        return vertices
    
    def get_mano_faces(self,mano_side='left'):
        if mano_side=='right':
            return self.right_mano_layer.th_faces
        elif mano_side=='left':
            return self.left_mano_layer.th_faces
        else:
            print("Error: mano_side should be 'left' or 'right'")
            return None
        
    def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp") -> None:
        os.makedirs(out_dir, exist_ok=True)
        sample = self.__getitem__(index, resize=None)
        intrinsics = sample.intrinsics.numpy()
        border_color = [255, 0, 0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
        out = cv2.VideoWriter(out_dir+"/gt_joints.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))
        projected_joints = np.zeros((sample.mano_joint_3d.shape[0],sample.mano_joint_3d.shape[1], 2))
        projected_joints[...,0] = intrinsics[0] * (sample.mano_joint_3d[...,0] / sample.mano_joint_3d[...,2]) + intrinsics[2]
        projected_joints[...,1] = intrinsics[1] * (sample.mano_joint_3d[...,1] / sample.mano_joint_3d[...,2]) + intrinsics[3]
        for i in tqdm(range(sample.mask.sum().item())):
            image = sample.rgb_frames[i].numpy().astype(np.uint8)
            image = image[:,:,::-1]
            composited = image.copy()
                    # sample.mano_joint_3d # (batch, timesteps, 21, 3)
                    # to(batch, timesteps, 21, 3)
            for j in range(projected_joints.shape[1]):
                cv2.circle(composited, (int(projected_joints[i,j,0]), int(projected_joints[i,j,1])), 3, border_color, -1)
            composited = np.concatenate([image, composited], axis=1)
            #iio.imwrite(os.path.join(out_dir, f"{i:03d}.jpg"), composited)
            out.write(composited)
        out.release()

    def visualize_manos_in_rgb(self, index: int,out_dir:str = "tmp") -> None:
        os.makedirs(out_dir, exist_ok=True)
        vertices = self.get_vertices_faces(index)
        sample = self.__getitem__(index, resize=None)
        
        faces = self.get_mano_faces(mano_side="right" if sample.mano_side.item()==1 else "left")
        intrinsics = sample.intrinsics.numpy()
        border_color = [255, 0, 0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
        out = cv2.VideoWriter(out_dir+"/gt.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))
        for i in tqdm(range(sample.mask.sum().item())):
            image = sample.rgb_frames[i].numpy().astype(np.uint8)
            image = image[:,:,::-1]
            render_rgb, rend_depth, render_mask = render_joint(vertices[i].numpy(), faces.numpy(),
                                                               intrinsics, h=sample.rgb_frames.shape[1], w=sample.rgb_frames.shape[2])
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
    
if __name__ == "__main__":
    # Example usage
    # dataset = HandHdf5Dataset(split='test', dataset_name='dexycb')
    # print(f"Dataset length: {len(dataset)}")
    # dataset.visualize_manos_in_rgb(115, out_dir="tmp")

    # for dataset_name in ['dexycb', 'interhand26m', 'arctic']:
    for dataset_name in ['dexycb']:
        # for split in ['test','train','val']:
        split='train'
        img_shape = set()
        dataset = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True)
        for i in tqdm(range(100)):
            sample = dataset[i]
            breakpoint()
            img_shape.add(sample.rgb_frames.shape[1:3])
        print(f"Dataset: {dataset_name}, Image shapes: {img_shape}")
            # assert sample.mano_betas not all zero
            # assert sample.mano_betas.abs().sum() > 1e-6, f"Sample {i} in {split} of {dataset_name} has all zero mano_betas"

    # joint_3d_calculated = mano_poses2joints_3d(
    #     mano_pose=sample.mano_pose,
    #     mano_betas=sample.mano_betas,
    #     mano_side=sample.mano_side,
    #     intrinsics=sample.intrinsics,
    #     extrinsics=sample.extrinsics,
    # )
    # sample.mano_joint_3d[0,0]
    # joint_3d_calculated[0,0]
    # hamer_output = dataset.hamer_output(0)
    # print(f"Sample mano_betas shape: {sample.mano_betas.shape}")
    # print(f"Sample dir: {dir(sample)}")
    # print(sample.mano_betas.shape,sample.mano_pose.shape)
    # print(f"Sample rgb_frames shape: {sample.rgb_frames.shape}")
    # print(f"Sample mano_side: {sample.mano_side}")
