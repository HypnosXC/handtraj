import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"]="1"
import json

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

import imageio.v3 as iio
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
# from hamer_helper import HamerHelper
# from .dataclass import HandTrainingData
from egoallo.data.dataclass import HandTrainingData, collate_dataclass
from egoallo.data.data_util import project_3d_to_2d,normalize_2d_keypoints

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
        split: Literal["train", "val", "test"] = "train",
        vis: bool = False,
        subseq_len: int = 64,
        clip_stride: int = 64,
        min_len: int = 32,
        speed_augment = None, # (0.9, 1.0)
        flip_augment = False,
        use_feature = "visual_token" # None, "visual_token", "cls_token"
    ) -> None:
        if speed_augment is not None:
            assert isinstance(speed_augment, tuple) and len(speed_augment) == 2, "speed_augment should be a tuple of (min_speed, max_speed)"
        self.augment = {
            'speed': speed_augment,
            'flip': flip_augment
        }
        self.split = split
        self.dataset_name = dataset_name
        self.use_token = use_feature
        self.archive: h5py.File | None = None
        self.feature_archive: h5py.File | None = None
        assert self.use_token in [None, "visual_token", "cls_token"], "use_feature should be None, 'visual_token' or 'cls_token'"
        if self.use_token is not None:
            assert speed_augment is None or speed_augment == (1.0, 1.0), "use_feature should be None if speed_augment is not (1.0, 1.0)"
            assert flip_augment is False, "use_feature should be None if flip_augment is True"
        assert flip_augment is False, "flip_augment is not supported yet"
        data_root = "/public/home/annie/preprocessed"
        # data_root = "/data/lingang_data/data1/handdata/"
        feat_root = "/public/home/annie/preprocessed/dino_feats/"
        # feat_root = "/data/annie/dino_feats"
        if dataset_name=="dexycb":
            self._hdf5_path = os.path.join(data_root, "dexycb_v6.hdf5")
            self.video_root = os.path.join(data_root, "dexycb/videos_v4")
        elif dataset_name=="interhand26m":
            self._hdf5_path = os.path.join(data_root, "interhand26m_v4.hdf5")
            self.video_root = os.path.join(data_root, "interhand26m/data/picked_videos")
        elif dataset_name=="arctic":
            self._hdf5_path = os.path.join(data_root, "arctic_v8.hdf5")
            self.video_root = os.path.join(data_root, "arctic/picked_videos")
        elif dataset_name == "ho3d":
            self.video_root = os.path.join(data_root, "HO3D_v3_new/picked_videos_v2")
            if split == "train":
                self._hdf5_path = os.path.join(data_root, "ho3d_v3.hdf5")
            elif split == "test":
                self._hdf5_path = os.path.join(data_root, "ho3d_evaluation_v11.hdf5")
            else:
                raise ValueError("For ho3d, split should be train or test")
        else:
            raise ValueError("dataset_name should be dexycb, interhand26m, arctic or ho3d")

        self._feature_hdf5 = os.path.join(feat_root, f"{self.dataset_name}_{self.split}_dino_fpn.hdf5")
        # Per-group .npy dir (float16, much faster I/O than HDF5).
        self._feature_npy_dir = os.path.join(feat_root + "_npy", f"{self.dataset_name}_{self.split}")
        self._use_npy_features = os.path.isdir(self._feature_npy_dir)
          
        # self.img_feat_root = os.path.join(img_feat_root[dataset_name], split)
        self.dataset_name = dataset_name
        self.vis = vis
        self._clip_stride = clip_stride
        self._subseq_len = subseq_len
        self._min_len = min_len
        json_dir = self._hdf5_path.replace('.hdf5','_json')
        

        
        json_file_path = os.path.join(json_dir, f"{self.split}.json")
        self._mapping = []
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        for group_name in json_data.keys():
            group_length = json_data[group_name]
            if self._subseq_len == -1:
                self._mapping.append((group_name, 0, group_length, group_length)) # group_name, start_idx, after_len, ori_len
            else:
                if group_length < self._min_len:
                    continue
                if group_length <= self._subseq_len:
                    self._mapping.append((group_name, 0, group_length, group_length)) # group_name, start_idx, after_len, ori_len
                else:
                    start_idx = 0
                    while start_idx + self._subseq_len < group_length:
                        self._mapping.append((group_name, start_idx, group_length - start_idx, group_length))
                        start_idx += self._clip_stride
                    # last clip
                    self._mapping.append((group_name, group_length - self._subseq_len, self._subseq_len, group_length))

        self.N = len(self._mapping)
        self.left_mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='left',
            mano_root="/data/xuchen",
        )
        self.right_mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='right',
            mano_root="/data/xuchen",
        )
        self.archive: h5py.File | None = None
        self.feature_archive: h5py.File | None = None

        # Preload small per-group metadata into memory to avoid repeated HDF5 reads.
        self._group_cache: dict[str, dict] = {}
        self._preload_group_cache()
        

    def _preload_group_cache(self) -> None:
        """Preload ALL per-group data (poses, joints, metadata) into CPU memory.
        After this, __getitem__ only reads visual features from disk."""
        archive = h5py.File(self._hdf5_path, 'r', swmr=True, libver='latest')
        group_names = set(m[0] for m in self._mapping)
        for gn in tqdm(group_names, desc=f"Preloading {self.dataset_name}/{self.split}"):
            ds = archive[gn]
            video_name = ds['video_name'][()].decode('utf-8')
            video_path = os.path.join(self.video_root, self.split, video_name)
            if os.path.exists(video_path):
                props = iio.improps(video_path, plugin="pyav")
                img_shape = (int(props.shape[1]), int(props.shape[2]))  # (h, w)
            else:
                img_shape = (256, 256)
            self._group_cache[gn] = {
                "mano_betas": torch.from_numpy(ds['mano_betas'][:]),
                "mano_poses": torch.from_numpy(ds['mano_poses'][:, 0]),  # (T, 51)
                "mano_joint_3d": torch.from_numpy(ds['mano_joint_3d'][:, 0]),  # (T, 21, 3)
                "extrinsics": torch.from_numpy(ds['extrinsics'][:]),
                "mano_side": 0 if ds['mano_side'][()].decode('utf-8') == 'left' else 1,
                "video_name": video_name,
                "intrinsics": torch.from_numpy(ds['intrinsics'][:]),
                "img_shape": img_shape,
            }
        archive.close()

    def __getitem__(self, index: int,resize=(512,512)) -> HandTrainingData:
        if self.use_token is not None and self.feature_archive is None and not self._use_npy_features:
            self.feature_archive = h5py.File(self._feature_hdf5, 'r', swmr=True, libver='latest')
        kwargs: dict[str, Any] = {}
        group_name, start_idx, after_len, ori_len = self._mapping[index]
        if self.augment['speed'] is not None:
            speed_aug = self.augment['speed']
            speed_factor = np.random.uniform(speed_aug[0], speed_aug[1])
        else:
            speed_factor = 1.0
        if self._subseq_len == -1:
            clip_len = after_len
        else:
            clip_len = min(int(self._subseq_len * speed_factor), after_len)

        # All data from CPU memory cache — no HDF5 reads for poses/joints.
        cached = self._group_cache[group_name]
        kwargs["mano_betas"] = cached["mano_betas"].clone()
        kwargs["mano_pose"] = cached["mano_poses"][start_idx:start_idx+clip_len].clone()
        kwargs["extrinsics"] = cached["extrinsics"].clone()
        kwargs["mano_joint_3d"] = cached["mano_joint_3d"][start_idx:start_idx+clip_len].clone()
        if cached["mano_side"] == 0:
            kwargs["mano_side"] = torch.zeros(1)
        else:
            kwargs["mano_side"] = torch.ones(1)

        if self.augment['speed'] is not None:
            ori_mano_pose = kwargs["mano_pose"].clone()
            ori_len = ori_mano_pose.shape[0]
            new_len = int(ori_len / speed_factor)
            ori_mano_trans = ori_mano_pose[:,48:51]
            mano_rots = ori_mano_pose[:,:48].reshape(-1,3)
            mano_quat = R.from_rotvec(mano_rots.numpy()).as_quat()
            ori_mano_quat = mano_quat.reshape(ori_len, -1,4)

            # linear interpolate on ori_mano_trans
            new_indices = np.linspace(0, ori_len - 1, new_len)
            new_mano_quat = np.zeros((new_len, ori_mano_quat.shape[1],4),dtype=np.float32)
            new_mano_trans = np.zeros((new_len, ori_mano_trans.shape[1]),dtype=np.float32)

            for t in range(1, ori_len):
                dots = np.sum(ori_mano_quat[t] * ori_mano_quat[t-1], axis=-1)
                neg_indices = dots < 0
                
                if np.any(neg_indices):
                    ori_mano_quat[t, neg_indices] *= -1.0

            # slerp interpolation on ori_mano_quat
            for i in range(ori_mano_quat.shape[1]):
                key_rots = R.from_quat(ori_mano_quat[:,i,:])
                slerp = Slerp(np.arange(ori_len), key_rots)
                interp_rots = slerp(new_indices)
                new_mano_quat[:,i,:] = interp_rots.as_quat()


            # for i in range(ori_mano_quat.shape[1]):
                # for j in range(4):
                #     new_mano_quat[:,i,j] = np.interp(new_indices, np.arange(ori_len), ori_mano_quat[:,i,j])
            

            norm = np.linalg.norm(new_mano_quat, axis=-1, keepdims=True)
            new_mano_quat = new_mano_quat / norm
            # Convert back to rotation vectors
            new_mano_rots = R.from_quat(new_mano_quat.reshape(-1,4)).as_rotvec().astype(np.float32)
            new_mano_rots = new_mano_rots.reshape(new_len, -1)
            for i in range(new_mano_trans.shape[1]):
                new_mano_trans[:,i] = np.interp(new_indices, np.arange(ori_len), ori_mano_trans.numpy()[:,i])
            # new_mano_trans = np.interp(new_indices, np.arange(ori_len), ori_mano_trans.numpy())
            if self._subseq_len != -1:
                kwargs["mano_pose"] = torch.cat([torch.from_numpy(new_mano_rots), torch.from_numpy(new_mano_trans)], dim=1)[:min(new_len, self._subseq_len), :]
            ori_joint_shape = kwargs["mano_joint_3d"].shape
            # _,new_3d_joint = self.get_vertices_joints_from_mano(kwargs["mano_pose"], kwargs["mano_betas"], is_right=(kwargs["mano_side"].item()==1))
            # kwargs["mano_joint_3d"] = new_3d_joint
            joint_interpolator = interp1d(np.arange(ori_len), kwargs["mano_joint_3d"].numpy(), axis=0, kind='linear')
            new_joints = torch.from_numpy(joint_interpolator(new_indices))
            if self._subseq_len != -1:
                kwargs["mano_joint_3d"] = new_joints[:min(new_len, self._subseq_len), :]
            else:
                kwargs["mano_joint_3d"] = new_joints

            clip_len = kwargs["mano_pose"].shape[0]

        if self._subseq_len == -1:
            pad_len = 0
        else:
            pad_len = self._subseq_len - clip_len

        kwargs["intrinsics"] = cached["intrinsics"].clone()
        if len(kwargs["intrinsics"].shape) == 1:
            if self._subseq_len !=-1: 
                kwargs["intrinsics"] = kwargs["intrinsics"].unsqueeze(0).expand(self._subseq_len, -1)
            else:
                kwargs["intrinsics"] = kwargs["intrinsics"].unsqueeze(0).expand(clip_len, -1)
        else:
            kwargs["intrinsics"] = kwargs["intrinsics"][start_idx:start_idx+clip_len]
            if self._subseq_len != -1: 
                kwargs["intrinsics"] = torch.cat([kwargs["intrinsics"], torch.zeros((pad_len, kwargs["intrinsics"].shape[1]))], dim=0)
        
        
        # kwargs["mask"] = torch.from_numpy(dataset['mask'][index]).type(torch.bool)
        kwargs["mask"] = torch.ones((clip_len), dtype=torch.bool)
        # pad if needed
        if pad_len > 0:
            kwargs["mano_pose"] = torch.cat([kwargs["mano_pose"], torch.zeros((pad_len, *kwargs["mano_pose"].shape[1:]))], dim=0)
            kwargs["mano_joint_3d"] = torch.cat([kwargs["mano_joint_3d"], torch.zeros((pad_len, *kwargs["mano_joint_3d"].shape[1:]))], dim=0)
            kwargs['mask'] = torch.cat([kwargs['mask'], torch.zeros((pad_len), dtype=torch.bool)], dim=0)
        video_name = cached["video_name"]
        image_height, image_width = cached["img_shape"]
        video_path = os.path.join(self.video_root, self.split, video_name)
        if self.vis:
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
                
        # load img features
        # img_feat_path = os.path.join(self.img_feat_root, f'imgfeat_{group_name}.pt')
        # assert os.path.exists(img_feat_path), f"Image feature file not found: {img_feat_path}"
        # img_features = torch.load(img_feat_path)  # T, feat_dim(768)
        # if self._subseq_len == -1:
        #     kwargs['img_feature'] = img_features
        # else:
        #     kwargs['img_feature'] = img_features[start_idx:start_idx+clip_len]

        if self.use_token is None:
            kwargs['img_feature'] = torch.zeros((1,768))  # dummy feature
        elif self.use_token == "cls_token":
            kwargs['img_feature'] = torch.from_numpy(self.feature_archive[group_name]["cls_token"][start_idx:start_idx+clip_len]) # T,768
            if self._subseq_len != -1:
                if pad_len > 0:
                    kwargs['img_feature'] = torch.cat([kwargs['img_feature'], torch.zeros((pad_len, kwargs['img_feature'].shape[1]))], dim=0)
        elif self.use_token == "visual_token":
            if self._use_npy_features:
                # Fast path: LRU-cached mmap per group (float16).
                # Same group is accessed many times per epoch (different start_idx),
                # so the mmap handle + OS page cache make subsequent reads near-free.
                npy_path = os.path.join(self._feature_npy_dir, f"{group_name}.npy")
                feat = _load_npy_mmap(npy_path)
                # Keep float16 through dataloader; model's .float() converts on GPU.
                kwargs['img_feature'] = torch.from_numpy(np.array(feat[start_idx:start_idx+clip_len]))
            else:
                # Fallback: HDF5 (slower with multiple workers).
                assert self.feature_archive[group_name]["layer_11"].shape[0] == ori_len, f"Feature length mismatch: index{index}, dataset_name{self.dataset_name}, {self.feature_archive[group_name]['layer_11'].shape[0]} vs {after_len}"
                kwargs['img_feature'] = torch.from_numpy(self.feature_archive[group_name]["layer_11"][start_idx:start_idx+clip_len]) # T,768,16,16
            if self._subseq_len != -1:
                if pad_len > 0:
                    kwargs['img_feature'] = torch.cat([kwargs['img_feature'], torch.zeros((pad_len, kwargs['img_feature'].shape[1], kwargs['img_feature'].shape[2], kwargs['img_feature'].shape[3]), dtype=kwargs['img_feature'].dtype)], dim=0)
        joints_3d = kwargs["mano_joint_3d"]   # (T, 21, 3)
        intr = kwargs["intrinsics"]           # (T, 4)
        mask = kwargs["mask"]                 # (T,)

        mano_joint_2d = project_3d_to_2d(joints_3d, intr)  # (T, 21, 2)
        # normalized_joint_2d = normalize_2d_keypoints(mano_joint_2d, kwargs["rgb_frames"].shape[2], kwargs["rgb_frames"].shape[1])  # (T, 21, 2)
        normalized_joint_2d = normalize_2d_keypoints(mano_joint_2d, image_height, image_width) 
        # 将 padded 帧的 2D joints 置零（这些帧 z=0 会产生无意义值）
        mano_joint_2d[~mask] = 0.0
        kwargs["joint_2d"] = normalized_joint_2d
        # if self.dataset_name == "arctic":
        #     kwargs['img_shape'] = (1000,1000)
        # elif self.dataset_name=="dexycb" or self.dataset_name=="ho3d":
        #     kwargs['img_shape'] = (480,640)
        # else:
        #     kwargs['img_shape'] = (512, 344)
        kwargs['img_shape'] = torch.tensor([image_height, image_width])  # (h, w)
        return HandTrainingData(**kwargs)
    def __len__(self) -> int:
        return self.N

    def get_mapping(self):
        return self._mapping

    def build_gpu_cache(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Precompute ALL small fields for every sample and store on GPU.

        Returns a dict of tensors indexed by sample index:
            mano_betas:    (N, 10)
            mano_pose:     (N, subseq_len, 51)
            mano_joint_3d: (N, subseq_len, 21, 3)
            joint_2d:      (N, subseq_len, 21, 2)
            intrinsics:    (N, subseq_len, 4)
            mask:          (N, subseq_len)
            mano_side:     (N, 1)
            img_shape:     (N, 2)
        """
        from .data_util import project_3d_to_2d, normalize_2d_keypoints
        N = len(self)
        T = self._subseq_len
        assert T > 0, "build_gpu_cache requires fixed subseq_len"

        # Allocate CPU buffers, fill, then move to GPU once.
        buf = {
            "mano_betas": torch.zeros(N, 10),
            "mano_pose": torch.zeros(N, T, 51),
            "mano_joint_3d": torch.zeros(N, T, 21, 3),
            "joint_2d": torch.zeros(N, T, 21, 2),
            "intrinsics": torch.zeros(N, T, 4),
            "mask": torch.zeros(N, T, dtype=torch.bool),
            "mano_side": torch.zeros(N, 1),
            "extrinsics": torch.zeros(N, 3, 4),
            "img_shape": torch.zeros(N, 2),
        }

        for i in tqdm(range(N), desc=f"Building GPU cache ({self.dataset_name}/{self.split})"):
            group_name, start_idx, after_len, ori_len = self._mapping[i]
            clip_len = min(T, after_len)
            cached = self._group_cache[group_name]
            image_height, image_width = cached["img_shape"]

            buf["mano_betas"][i] = cached["mano_betas"]
            buf["mano_pose"][i, :clip_len] = cached["mano_poses"][start_idx:start_idx+clip_len]
            buf["mano_joint_3d"][i, :clip_len] = cached["mano_joint_3d"][start_idx:start_idx+clip_len]
            buf["extrinsics"][i] = cached["extrinsics"]
            buf["mano_side"][i] = float(cached["mano_side"])
            buf["mask"][i, :clip_len] = True
            buf["img_shape"][i] = torch.tensor([image_height, image_width])

            intr = cached["intrinsics"].clone()
            if len(intr.shape) == 1:
                buf["intrinsics"][i] = intr.unsqueeze(0).expand(T, -1)
            else:
                buf["intrinsics"][i, :clip_len] = intr[start_idx:start_idx+clip_len]

            # 2D joint projection
            joints_3d = buf["mano_joint_3d"][i]  # (T, 21, 3)
            intr_t = buf["intrinsics"][i]         # (T, 4)
            j2d = project_3d_to_2d(joints_3d, intr_t)
            j2d = normalize_2d_keypoints(j2d, image_height, image_width)
            j2d[~buf["mask"][i]] = 0.0
            buf["joint_2d"][i] = j2d

        # Move everything to GPU in one shot.
        return {k: v.to(device) for k, v in buf.items()}

    def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp",resize=None,from_mano=False) -> None:
        os.makedirs(out_dir, exist_ok=True)
        sample = self.__getitem__(index, resize=resize)
        intrinsics = sample.intrinsics.numpy()
        border_color = [255, 0, 0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
        out = cv2.VideoWriter(out_dir+"/gt_joints.mp4", fourcc, 10, (sample.rgb_frames.shape[2]*2, sample.rgb_frames.shape[1]))

        if from_mano:
            _, mano_joint_3d = self.get_vertices_joints_from_index(index)
            mpjpe_frames = torch.norm(mano_joint_3d - sample.mano_joint_3d, dim=-1)
            print("MPJPE per frame (m): ", mpjpe_frames.mean(dim=-1))
            print("all mean: ", mpjpe_frames.mean().item())
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

    def visualize_manos_in_rgb(self, index: int,out_dir:str = "tmp",resize=None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        mano_vertices, _ = self.get_vertices_joints_from_index(index)
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

    def get_vertices_joints_from_mano(self, mano_pose: torch.Tensor, mano_betas: torch.Tensor, is_right: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not is_right:
            vertices, joints = self.left_mano_layer(
                mano_pose[:,:48],
                mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
                mano_pose[:,48:51]
            )
        elif is_right:
            vertices, joints = self.right_mano_layer(
                mano_pose[:,:48],
                mano_betas.unsqueeze(0).repeat(mano_pose.shape[0], 1),
                mano_pose[:,48:51]
            )
        vertices = vertices/ 1000  # Convert to meters
        joints = joints / 1000  # Convert to meters
        return vertices, joints
    
    def get_vertices_joints_from_index(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.__getitem__(index)
        # if sample.mano_side.item() == 0:
        #     vertices, joints = self.left_mano_layer(
        #         sample.mano_pose[:,:48],
        #         sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
        #         sample.mano_pose[:,48:51]
        #     )
        # elif sample.mano_side.item() == 1:
        #     vertices, joints = self.right_mano_layer(
        #         sample.mano_pose[:,:48],
        #         sample.mano_betas.unsqueeze(0).repeat(sample.mano_pose.shape[0], 1),
        #         sample.mano_pose[:,48:51]
        #     )
        # else:
        #     print("Error: mano_side should be 0 or 1")
        #     return None
        # vertices = vertices/ 1000  # Convert to meters
        # joints = joints / 1000  # Convert to meters
        # print((joints - sample.mano_joint_3d)[0])
        return self.get_vertices_joints_from_mano(sample.mano_pose, sample.mano_betas, is_right=(sample.mano_side.item()==1))

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


@lru_cache(maxsize=128)
def _load_npy_mmap(npy_path: str) -> np.ndarray:
    """Module-level LRU-cached mmap loader (picklable for DataLoader workers)."""
    return np.load(npy_path, mmap_mode='r')


from torch.utils.data import ConcatDataset


class FeatureOnlyDataset(torch.utils.data.Dataset):
    """Lightweight dataset that only loads visual features + returns sample index.
    Used with GPU-cached small fields to eliminate all CPU→GPU transfer except features."""

    def __init__(self, parent_dataset: "HandHdf5Dataset") -> None:
        self.parent = parent_dataset

    def __len__(self) -> int:
        return len(self.parent)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        # Route to the correct sub-dataset to load only img_feature.
        offset = 0
        for ds in self.parent.dataset.datasets:
            if index - offset < len(ds):
                local_idx = index - offset
                group_name, start_idx, after_len, ori_len = ds._mapping[local_idx]
                clip_len = min(ds._subseq_len, after_len) if ds._subseq_len > 0 else after_len
                pad_len = ds._subseq_len - clip_len if ds._subseq_len > 0 else 0

                if ds.use_token == "visual_token":
                    if ds._use_npy_features:
                        npy_path = os.path.join(ds._feature_npy_dir, f"{group_name}.npy")
                        feat = _load_npy_mmap(npy_path)
                        img_feat = torch.from_numpy(np.array(feat[start_idx:start_idx+clip_len]))
                    else:
                        if ds.feature_archive is None:
                            ds.feature_archive = h5py.File(ds._feature_hdf5, 'r', swmr=True, libver='latest')
                        img_feat = torch.from_numpy(ds.feature_archive[group_name]["layer_11"][start_idx:start_idx+clip_len])
                    if pad_len > 0:
                        img_feat = torch.cat([img_feat, torch.zeros((pad_len, *img_feat.shape[1:]), dtype=img_feat.dtype)], dim=0)
                elif ds.use_token == "cls_token":
                    if ds.feature_archive is None:
                        ds.feature_archive = h5py.File(ds._feature_hdf5, 'r', swmr=True, libver='latest')
                    img_feat = torch.from_numpy(ds.feature_archive[group_name]["cls_token"][start_idx:start_idx+clip_len])
                    if pad_len > 0:
                        img_feat = torch.cat([img_feat, torch.zeros((pad_len, img_feat.shape[1]))], dim=0)
                else:
                    img_feat = torch.zeros((1, 768))

                return index, img_feat
            offset += len(ds)
        raise IndexError(f"Index {index} out of range")


class HandHdf5Dataset(torch.utils.data.Dataset[HandTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file.

    Args:
        split: "train", "val" or "test" splits to include in the dataset. For ho3d,
            use "train" and "test".
        dataset_name: Name of the dataset to load. Options are "all", "dexycb", "interhand26m", "arctic" and "ho3d".
        vis: Whether to load RGB frames for visualization.
        subseq_len: Length of subsequences to sample from the dataset.
        clip_stride: Stride between subsequences.
        min_len: Minimum length of subsequences to include.
    """
    def __init__(self,split: Literal["train", "val", "test"] = "train",
                 dataset_name: Literal["all","dexycb", "arctic", "interhand26m", "ho3d"] = 'all',
                 vis=False, subseq_len: int = 64,
                 clip_stride: int = 16,
                 min_len:int=32,
                 speed_augment = None,
                 flip_augment = False,
                 use_feature = "visual_token" # None, "visual_token", "cls_token"
                 ) -> None:
        if dataset_name == "all":
            dataset_ih26 = HandHdf5EachDataset(split=split, dataset_name="interhand26m", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len, speed_augment=speed_augment, flip_augment=flip_augment, use_feature=use_feature)
            dataset_dexycb = HandHdf5EachDataset(split=split, dataset_name="dexycb", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len, speed_augment=speed_augment, flip_augment=flip_augment, use_feature=use_feature)
            dataset_arctic = HandHdf5EachDataset(split=split, dataset_name="arctic", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len, speed_augment=speed_augment, flip_augment=flip_augment, use_feature=use_feature)
            if split == "train" or split == "test":
                dataset_ho3d = HandHdf5EachDataset(split=split, dataset_name="ho3d", vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len, speed_augment=speed_augment, flip_augment=flip_augment, use_feature=use_feature)
                self.dataset = ConcatDataset([dataset_ih26, dataset_dexycb, dataset_arctic, dataset_ho3d])
            else:
                self.dataset = ConcatDataset([dataset_ih26, dataset_dexycb, dataset_arctic])
        else:
            self.dataset = ConcatDataset([HandHdf5EachDataset(split=split, dataset_name=dataset_name, vis=vis, subseq_len=subseq_len, clip_stride=clip_stride,min_len=min_len, speed_augment=speed_augment, flip_augment=flip_augment, use_feature=use_feature)])
        

    def __getitem__(self, index: int,resize=(512,512)) -> HandTrainingData:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.__getitem__(index,resize=resize)
            else:
                index -= len(ds)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_vertices_joints_from_index(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.get_vertices_joints_from_index(index)
            else:
                index -= len(ds)

    def get_vertices_joints_from_mano(self, mano_pose: torch.Tensor, mano_betas: torch.Tensor, is_right: bool) -> tuple[torch.Tensor, torch.Tensor]:
        for ds in self.dataset.datasets:
            return ds.get_vertices_joints_from_mano(mano_pose, mano_betas, is_right)


    def get_mano_faces(self,mano_side='left'):
        for ds in self.dataset.datasets:
            return ds.get_mano_faces(mano_side=mano_side)

    def build_gpu_cache(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Build GPU cache across all sub-datasets, with global indexing."""
        caches = []
        for ds in self.dataset.datasets:
            caches.append(ds.build_gpu_cache(device))
        # Concatenate along sample dimension.
        return {k: torch.cat([c[k] for c in caches], dim=0) for k in caches[0]}

    def visualize_joints_in_rgb(self, index: int,out_dir:str = "tmp",resize=None,from_mano=False) -> None:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.visualize_joints_in_rgb(index,out_dir=out_dir,resize=resize,from_mano=from_mano)
            else:
                index -= len(ds)
    def visualize_manos_in_rgb(self, index: int,out_dir:str = "tmp",resize=None) -> None:
        for ds in self.dataset.datasets:
            if index < len(ds):
                return ds.visualize_manos_in_rgb(index,out_dir=out_dir,resize=resize)
            else:
                index -= len(ds)

    def get_mapping(self):
        mapping = []
        for ds in self.dataset.datasets:
            mapping.extend([(ds.dataset_name,item[0], item[1], item[2], item[3]) for item in ds.get_mapping()])
        return mapping

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
    start_time = time.time()
    dataset = HandHdf5Dataset(split='train', dataset_name='ho3d', vis=False, subseq_len=64, clip_stride=64, min_len=32, speed_augment=None,use_feature="visual_token")
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    load_start_time = time.time()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_dataclass)
    # print("Total samples in all dataset:", len(dataset))
    for batch in tqdm(dataloader):
        pass

    # for dataset_name in ['ho3d', 'dexycb', 'interhand26m', 'arctic']:
    # # for dataset_name in ['interhand26m']:
    #     this_dataset = HandHdf5Dataset(split='train', dataset_name=dataset_name, vis=True, subseq_len=64, clip_stride=64, min_len=32, speed_augment=None,use_feature=None)
    #     print(dataset_name)
    #     # randomly select 20 id from len(this_dataset) and visualize
    #     for i in np.random.choice(len(this_dataset), size=20, replace=False):
    #         sample = this_dataset.__getitem__(i,resize=None)
    #         # print(sample.video_name)
    #         print(sample.img_shape)



    # for i in tqdm(range(len(dataset))):
    #     sample = dataset.__getitem__(i)


    # demo_datas = [("ho3d", "train"), ("ho3d", "test")]
    # for dataset_name, split in demo_datas:
    #     dataset = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=False, subseq_len=64, clip_stride=64, min_len=32, speed_augment=None)
    #     print(f"Total samples in {dataset_name} {split} dataset:", len(dataset))
    #     for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name} {split}"):
    #         sample = dataset.__getitem__(i, resize=None)
    #     dataset_vis = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True, subseq_len=64, clip_stride=64, min_len=32, speed_augment=None)
    #     dataset_vis.visualize_joints_in_rgb(34, out_dir=f"vis_{dataset_name}_{split}", resize=None)

    # dataset = HandHdf5Dataset(split='train', dataset_name='ho3d', vis=False,subseq_len=64, clip_stride=64, min_len=32, speed_augment=None)
    # # dataset = HandHdf5Dataset(split='train', dataset_name='all', vis=False,subseq_len=64, clip_stride=64, min_len=32, speed_augment=None)
    # print("Total samples in all dataset:", len(dataset))
    # for i in tqdm(range(len(dataset))):
    #     sample = dataset.__getitem__(i, resize=None)

    # for dataset_name in ['ho3d', 'dexycb', 'interhand26m', 'arctic']:
        # dataset = HandHdf5Dataset(split='train', dataset_name=dataset_name, vis=True, subseq_len=64, clip_stride=64, min_len=32, speed_augment=None)
        # dataset.visualize_joints_in_rgb(4, out_dir=f"vis_{dataset_name}", resize=(512,512))
        # dataset.visualize_joints_in_rgb(4, out_dir=f"vis_{dataset_name}_from_mano_results", resize=(512,512),from_mano=True)

    # dataset = HandHdf5Dataset(split='test', dataset_name='arctic', vis=True,subseq_len=64, clip_stride=4, min_len=32, speed_augment=(0.5,0.8))
    # dataset.visualize_joints_in_rgb(7, out_dir="arctic_results", resize=None)
    # dataset.visualize_manos_in_rgb(7, out_dir="arctic_results", resize=(512,512))

    # for split in ['train','test']:
    #     dataset = HandHdf5Dataset(split=split, dataset_name='ho3d', vis=False, subseq_len=64, clip_stride=64, min_len=32)
    #     for i in tqdm(range(len(dataset))):
    #         sample = dataset.__getitem__(i, resize=None)

    # for split in ['train','val','test']:
        # dataset = HandHdf5Dataset(split=split, dataset_name='dexycb', vis=False, subseq_len=64, clip_stride=64, min_len=32)
        # for i in tqdm(range(len(dataset)), desc=f"Processing dexycb {split}"):
        #     sample = dataset.__getitem__(i, resize=None)
        # dataset = HandHdf5Dataset(split=split, dataset_name='interhand26m', vis=False, subseq_len=64, clip_stride=64, min_len=32)
        # for i in tqdm(range(len(dataset)), desc=f"Processing interhand26m {split}"):
        #     sample = dataset.__getitem__(i, resize=None)
        # dataset = HandHdf5Dataset(split=split, dataset_name='arctic', vis=False, subseq_len=-1)
        # for i in tqdm(range(len(dataset)), desc=f"Processing arctic {split}"):
        #     sample = dataset.__getitem__(i, resize=None)

    # dataset = HandHdf5Dataset(split='test', dataset_name='ho3d', vis=True, subseq_len=64, clip_stride=4, min_len=32)
    # # # dataset.visualize_joints_in_rgb(40, out_dir="ho3d_from_mano", resize=None, from_mano=True)
    # dataset.visualize_joints_in_rgb(4, out_dir="ho3d_test_results", resize=(512,512))
    # dataset.visualize_manos_in_rgb(4, out_dir="ho3d_test_results", resize=(512,512))
    # # # dataset.visualize_manos_in_rgb(4, out_dir="ho3d_gt", resize=(512,512), debug=True)

    # dataset = HandHdf5Dataset(split='train', dataset_name='ho3d', vis=True, subseq_len=64, clip_stride=64, min_len=32)
    # dataset.visualize_joints_in_rgb(10, out_dir="ho3d_train_results", resize=(512,512))
    # dataset.visualize_manos_in_rgb(10, out_dir="ho3d_train_results", resize=(512,512))


    # dataset = HandHdf5Dataset(split='test', dataset_name='interhand26m', vis=True, subseq_len=64, clip_stride=4, min_len=32)

    # dataset.visualize_joints_in_rgb(10, out_dir="interhand26m_results", resize=None)
    # dataset.visualize_manos_in_rgb(10, out_dir="interhand26m_results", resize=None)



    # dataset = HandHdf5Dataset(split='train', dataset_name='dexycb', vis=True, subseq_len=64, clip_stride=4)
    # dataset.visualize_joints_in_rgb(10, out_dir="dexycb_results", resize=None)
    # dataset.visualize_manos_in_rgb(10, out_dir="dexycb_results", resize=None)

    # less =0
    # more =0
    # for split in ['train','val','test']:
    #     dataset = HandHdf5Dataset(split=split, dataset_name='all', vis=False, subseq_len=64, clip_stride=64, min_len=32)
    #     for i in tqdm(range(len(dataset))):
    #         sample = dataset.__getitem__(i, resize=None)
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