import sys

from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
import imageio.v3 as iio
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

class DexYCBHdf5Dataset(torch.utils.data.Dataset[HandTrainingData]):
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
        hdf5_path: Path="/public/datasets/handdata/dexycb_s_all_unseen.hdf5",
        split: Literal["train", "val", "test"] = "train",
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
        self._hdf5_path = hdf5_path
        self.rgb_format = "color_{:06d}.jpg"
        self.split = split

        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            # ]
            dataset = hdf5_file[self.split]  # Assuming splits is a single split.
            # self._subseq_len = subseq_len
            self._subseq_len = 64  # Default subsequence length, can be overridden.
            self.N = dataset['data_dir'].shape[0]
            self.keys = list(dataset.keys())

    def __getitem__(self, index: int) -> HandTrainingData:
        kwargs: dict[str, Any] = {}

        with h5py.File(self._hdf5_path, "r") as f:
            dataset = f[self.split]
            # ndarray to tensor
            kwargs["mano_betas"]=torch.from_numpy(dataset['mano_betas'][index])
            kwargs["mano_pose"]=torch.from_numpy(dataset['mano_poses'][index,:,0])
            timesteps= kwargs["mano_pose"].shape[0]
            kwargs["mano_joint_3d"] = torch.from_numpy(dataset['mano_joint_3d'][index,:,0])
            kwargs["intrinsics"]= torch.from_numpy(dataset['intrinsics'][index])
            kwargs["extrinsics"] =torch.from_numpy(dataset['extrinsics'][index])
            if dataset['mano_side'][index][0].decode('utf-8') == 'left':
                kwargs["mano_side"] = torch.zeros(1)
            else:
                kwargs["mano_side"] = torch.ones(1)
            data_dir = dataset['data_dir'][index][0].decode('utf-8')
            # firstly ignore the video data
            '''
            start_frame = dataset['start_frame'][index]
            rgb_frames = []
            for i in range(start_frame[0], start_frame[0] + self._subseq_len):
                rgb_frame = self.rgb_format.format(i)
                rgb_path = os.path.join(data_dir, rgb_frame)
                # open the image and convert it to tensor
                rgb_image = iio.imread(rgb_path)
                rgb_frames.append(torch.from_numpy(rgb_image))
            '''
            kwargs["rgb_frames"] = torch.ones((timesteps,))  # torch.stack(rgb_frames)
            kwargs["mask"] = torch.ones((timesteps,), dtype=torch.bool)
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
                    "mano_hand_pose": hamer_out_left["mano_hand_pose"],
                    "mano_hand_betas": hamer_out_left["mano_hand_betas"],
                    "mano_hand_global_orient": hamer_out_left["mano_hand_global_orient"],
                }

            if hamer_out_right is None:
                hamer_out_frame["right"] = None
            else:
                hamer_out_frame["right"] = {
                    "verts": hamer_out_right["verts"],
                    "keypoints_3d": hamer_out_right["keypoints_3d"],
                    "mano_hand_pose": hamer_out_right["mano_hand_pose"],
                    "mano_hand_betas": hamer_out_right["mano_hand_betas"],
                    "mano_hand_global_orient": hamer_out_right["mano_hand_global_orient"],
                }

            hamer_output_seq.append(hamer_out_frame)        
        print(f"Hamer output time: {time.time() - start_time:.2f} seconds")
        return hamer_output_seq
    
    def get_vertices_faces(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.__getitem__(index)
        mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side=mano_side_str[(int)(sample.mano_side.numpy())],
            mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        )
        vertices, joints = mano_layer(
            sample.mano_pose[:,:48],
            sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
            sample.mano_pose[:,48:51]
        )
        vertices = vertices/ 1000  # Convert to meters
        faces_m = mano_layer.th_faces
        return vertices, faces_m

    def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp") -> None:
        os.makedirs(out_dir, exist_ok=True)
        vertices, faces = self.get_vertices_faces(index)
        sample = self.__getitem__(index)
        intrinsics = sample.intrinsics.numpy()
        border_color = [255, 0, 0]
        for i in range(self._subseq_len):
            image = sample.rgb_frames[i].numpy().astype(np.uint8)
            # render_rgb, render_depth, render_mask = render_joint(
            #     vertices[i].numpy(),
            #     faces.numpy(),
            #     intrinsics,
            #     h=image.shape[0],
            #     w=image.shape[1],
            # )
            # # Convert to uint8
            # render_rgb = (render_rgb * 255).astype(np.uint8)
            render_rgb, rend_depth, render_mask = render_joint(vertices[i].numpy(), faces.numpy(),
                                                               intrinsics, h=sample.rgb_frames.shape[1], w=sample.rgb_frames.shape[2])
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

def mano_poses2joints_3d(mano_pose: torch.FloatTensor, mano_betas: torch.FloatTensor, mano_side: str, extrinsics: torch.FloatTensor, intrinsics: torch.FloatTensor) -> torch.FloatTensor:
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
    # joints = joints.permute(0, 2, 1)  # (64, 3, 21)
    # R = extrinsics[:,:3]
    # t = extrinsics[:,3:]    # (3,1)
    # invert to world->camera:
    # R = R.T
    # t = -R @ t

    # 3*3 @ 64*3*21 + 64*3 = 64*3*21
    # joint_3d = torch.bmm(R.unsqueeze(0).repeat(joints.shape[0], 1, 1), joints) + t.unsqueeze(0).repeat(joints.shape[0], 1, joints.shape[2])
    # return joint_3d.permute(0, 2, 1)  # (64, 21, 3)
    return joints
    

if __name__ == "__main__":
    # Example usage
    dataset = DexYCBHdf5Dataset()
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    dataset.visualize_joints_in_rgb(0, out_dir="tmp")

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
    print(f"Sample dir: {dir(sample)}")
    print(sample.mano_betas.shape,sample.mano_pose.shape)
    print(f"Sample rgb_frames shape: {sample.rgb_frames.shape}")
    print(f"Sample mano_side: {sample.mano_side}")