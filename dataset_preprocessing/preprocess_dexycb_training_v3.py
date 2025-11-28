# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

import os
import yaml
import numpy as np
import torch
import json
import tqdm

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

  # 'mano_20200820_133405_subject-03_right', , 'mano_20200813_143449_subject-02_right', 'mano_20200903_101911_subject-04_right',  'mano_20201002_103251_subject-08_right', 'mano_20201022_105224_subject-10_right',, 'mano_20200918_110920_subject-06_right', 'mano_20200807_132210_subject-07_right', 'intrinsics', 'mano_20200908_140650_subject-05_right', 'mano_20200709_140042_subject-01_right', 'mano_20200514_142106_subject-09_right', 

_BOP_EVAL_SUBSAMPLING_FACTOR = 4

class DexYCBDataset():
  """DexYCB dataset."""
  ycb_classes = _YCB_CLASSES
  mano_joints = _MANO_JOINTS
  mano_joint_connect = _MANO_JOINT_CONNECT

  def __init__(self, split, data_dir="/public/datasets/handdata/dexycb", setup="s_all_unseen"):
    """Constructor.

    Args:
      setup: Setup name. 's0', 's1', 's2', 's3' or 's_all_unseen'.
      data_dir: Path to the dataset directory.
      split: Split name. 'train', 'val', or 'test'.
    """
    print(f"Starting DexYCBDataset with setup: {setup}, split: {split}, data_dir: {data_dir}")
    self._setup = setup
    self._split = split

    # assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
    self._data_dir = data_dir
    self._calib_dir = os.path.join(self._data_dir, "calibration")
    self._model_dir = os.path.join(self._data_dir, "models")

    self._color_format = "color_{:06d}.jpg"
    self._depth_format = "aligned_depth_to_color_{:06d}.png"
    self._label_format = "labels_{:06d}.npz"
    self._h = 480
    self._w = 640

    
    # self.max_len=0
    # self.min_len=1000000

    self._obj_file = {
        k: os.path.join(self._model_dir, v, "textured_simple.obj")
        for k, v in _YCB_CLASSES.items()
    }

    # Seen subjects, camera views, grasped objects.
    if self._setup == 's0':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 != 4]
      if self._split == 'val':
        subject_ind = [0, 1]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]
      if self._split == 'test':
        subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]

    # Unseen subjects.
    if self._setup == 's1':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
      if self._split == 'val':
        subject_ind = [6]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
      if self._split == 'test':
        subject_ind = [7, 8]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))

    # Unseen camera views.
    if self._setup == 's2':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5]
        sequence_ind = list(range(100))
      if self._split == 'val':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [6]
        sequence_ind = list(range(100))
      if self._split == 'test':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [7]
        sequence_ind = list(range(100))

    # Unseen grasped objects.
    if self._setup == 's3':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [
            i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
        ]
      if self._split == 'val':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
      if self._split == 'test':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

    # 新的设置：所有未见
    if self._setup == 's_all_unseen':
        if self._split == 'train':
            # 训练：使用部分受试者、部分相机、部分物体
            subject_ind = [0, 1, 2, 3, 4, 5,6,7,8]  # 减少训练受试者
            serial_ind = [0, 1, 2, 3, 4,5,6]      # 减少训练相机
            sequence_ind = [
                i for i in range(100) 
                if i // 5 not in (3,)  # 排除某些物体
            ]
            # sss_set = [(0,0,0), (1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6), (7,7,7)]
            all_arrangements = []
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        meta_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
                                                 sorted(os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub])))[id_seq],"meta.yml")
                        with open(meta_file, 'r') as f:
                            meta = yaml.load(f, Loader=yaml.FullLoader)
                        num_frames = meta['num_frames']
                        this_data_path = os.path.join(self._data_dir, _SUBJECTS[id_sub])
                        this_data_path = os.path.join(this_data_path, sorted(os.listdir(this_data_path))[id_seq], 
                                            _SERIALS[id_ser])
                        
                        start_frame = 0
                        for i in range(num_frames):
                            mano_labels_file = os.path.join(this_data_path, f'labels_{i:06d}.npz')
                            label = np.load(mano_labels_file)
                            mano_pose = label['pose_m']
                            if sum(mano_pose[0,:3])!=0:
                                start_frame = i
                                break
                        
                        real_num_frames = num_frames - start_frame
                        # if real_num_frames > 64:
                        #     all_arrangements.append((id_sub, id_ser, id_seq, start_frame,64))
                        #     all_arrangements.append((id_sub, id_ser, id_seq, num_frames - 64,64))
                        # else:
                        all_arrangements.append((id_sub, id_ser, id_seq, start_frame, real_num_frames))
            self._mapping = np.array(all_arrangements)
        if self._split == 'val':
            subject_ind = [9]  # 验证使用单个受试者
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]  # 验证使用所有相机
            sequence_ind = list(range(100))  # 验证使用所有序列
            all_arrangements = []
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        meta_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
                                                 sorted(os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub])))[id_seq],"meta.yml")
                        with open(meta_file, 'r') as f:
                            meta = yaml.load(f, Loader=yaml.FullLoader)
                        num_frames = meta['num_frames']
                        this_data_path = os.path.join(self._data_dir, _SUBJECTS[id_sub])
                        this_data_path = os.path.join(this_data_path, sorted(os.listdir(this_data_path))[id_seq], 
                                            _SERIALS[id_ser])
                        
                        start_frame = 0
                        for i in range(num_frames):
                            mano_labels_file = os.path.join(this_data_path, f'labels_{i:06d}.npz')
                            label = np.load(mano_labels_file)
                            mano_pose = label['pose_m']
                            if sum(mano_pose[0,:3])!=0:
                                start_frame = i
                                break
                        
                        real_num_frames = num_frames - start_frame
                        # if real_num_frames > 64:
                        #     all_arrangements.append((id_sub, id_ser, id_seq, start_frame,64))
                        #     all_arrangements.append((id_sub, id_ser, id_seq, num_frames - 64,64))
                        # else:
                        all_arrangements.append((id_sub, id_ser, id_seq, start_frame, real_num_frames))
            self._mapping = np.array(all_arrangements)
        if self._split == 'test':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 测试使用多个受试者
            serial_ind = [7]  # 测试使用所有相机
            sequence_ind = list(range(100))  # 测试使用所有序列
            all_arrangements = []
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        meta_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
                                                 sorted(os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub])))[id_seq],"meta.yml")
                        with open(meta_file, 'r') as f:
                            meta = yaml.load(f, Loader=yaml.FullLoader)
                        num_frames = meta['num_frames']
                        this_data_path = os.path.join(self._data_dir, _SUBJECTS[id_sub])
                        this_data_path = os.path.join(this_data_path, sorted(os.listdir(this_data_path))[id_seq], 
                                            _SERIALS[id_ser])
                        
                        start_frame = 0
                        for i in range(num_frames):
                            mano_labels_file = os.path.join(this_data_path, f'labels_{i:06d}.npz')
                            label = np.load(mano_labels_file)
                            mano_pose = label['pose_m']
                            if sum(mano_pose[0,:3])!=0:
                                start_frame = i
                                break
                        
                        real_num_frames = num_frames - start_frame
                        # if real_num_frames > 64:
                        #     all_arrangements.append((id_sub, id_ser, id_seq, start_frame,64))
                        #     all_arrangements.append((id_sub, id_ser, id_seq, num_frames - 64,64))
                        # else:
                        all_arrangements.append((id_sub, id_ser, id_seq, start_frame, real_num_frames))

                        # all_arrangements.append((id_sub, id_ser, id_seq))
            sequence_ind = [i for i in range(100) if i // 5 in (3,)]
            serial_ind = [0, 1, 2, 3, 4, 5, 6]
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        meta_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
                                                 sorted(os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub])))[id_seq],"meta.yml")
                        with open(meta_file, 'r') as f:
                            meta = yaml.load(f, Loader=yaml.FullLoader)
                        num_frames = meta['num_frames']
                        this_data_path = os.path.join(self._data_dir, _SUBJECTS[id_sub])
                        this_data_path = os.path.join(this_data_path, sorted(os.listdir(this_data_path))[id_seq], 
                                            _SERIALS[id_ser])
                        
                        start_frame = 0
                        for i in range(num_frames):
                            mano_labels_file = os.path.join(this_data_path, f'labels_{i:06d}.npz')
                            label = np.load(mano_labels_file)
                            mano_pose = label['pose_m']
                            if sum(mano_pose[0,:3])!=0:
                                start_frame = i
                                break
                        
                        real_num_frames = num_frames - start_frame
                        # if real_num_frames > 64:
                        #     all_arrangements.append((id_sub, id_ser, id_seq, start_frame,64))
                        #     all_arrangements.append((id_sub, id_ser, id_seq, num_frames - 64,64))
                        # else:
                        all_arrangements.append((id_sub, id_ser, id_seq, start_frame, real_num_frames))
            self._mapping = np.array(all_arrangements)
            sequence_ind = list(range(100))

    self._subjects = [_SUBJECTS[i] for i in subject_ind]

    self._serials = [_SERIALS[i] for i in serial_ind]
    self._intrinsics = []
    self.extrinsics = []
    self._mano_side = []
    self._mano_betas = []
    self._start_frames = []
    self.poses = []
    self.hand_3d_joints = []

    # go through the mapping
    for id_ser in range(8):
        intr_file = os.path.join(self._calib_dir, "intrinsics",
                                "{}_{}x{}.yml".format(_SERIALS[id_ser], self._w, self._h))
        with open(intr_file, 'r') as f:
            intr = yaml.load(f, Loader=yaml.FullLoader)
        intr = intr['color']
        self._intrinsics.append([
            intr['fx'], intr['fy'], intr['ppx'], intr['ppy']
        ])

    for id_sub, id_ser, id_seq, start_frame, seq_len in tqdm.tqdm(self._mapping, desc="Loading dataset"):
        meta_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
                                 sorted(os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub])))[id_seq],"meta.yml")
        # pose_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
        #                          os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub]))[id_seq], "pose.npz")
        with open(meta_file, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        # self._num_frames.append(meta['num_frames'])
        # num_frames = meta['num_frames']

        # start_frame = num_frames // 2 - 32
        self._start_frames.append(start_frame)
        # if meta['num_frames'] > self.max_len:
        #     self.max_len = meta['num_frames']
        # if meta['num_frames'] < self.min_len:
        #     self.min_len = meta['num_frames']
        # self._ycb_ids.append(meta['ycb_ids'])
        # self._ycb_grasp_ind.append(meta['ycb_grasp_ind'])
        self._mano_side.append(meta['mano_sides'][0])
        mano_calib_file = os.path.join(self._data_dir, "calibration",
                                       "mano_{}".format(meta['mano_calib'][0]),
                                       "mano.yml")
        extrinsic_file = os.path.join(self._data_dir, "calibration", 
                                      "extrinsics_{}".format(meta["extrinsics"]),"extrinsics.yml")
        with open(extrinsic_file, 'r') as f:
            extrinsics = yaml.load(f, Loader=yaml.FullLoader)
        T = extrinsics['extrinsics']
        # data = np.load(pose_file)
        # self.poses.append(data['pose_m'][start_frame:start_frame + 64, :, :])
        np_T = np.array(T[_SERIALS[id_ser]])

        self.extrinsics.append(np_T.reshape(3, 4))
        with open(mano_calib_file, 'r') as f:
            mano_calib = yaml.load(f, Loader=yaml.FullLoader)
        self._mano_betas.append(mano_calib['betas'])

  def __getitem__(self, idx):
    id_sub, id_ser, id_seq, start_frame, seq_len = self._mapping[idx]
    d = os.path.join(self._data_dir, _SUBJECTS[id_sub])
    d = os.path.join(d, sorted(os.listdir(d))[id_seq], 
                     _SERIALS[id_ser])
    sample = {
        'data_dir': d,
        # 'num_frames': self._num_frames[idx],
        'start_frame': start_frame,
        'seq_len': seq_len,
        'intrinsics': self._intrinsics[id_ser],
        'mano_side': self._mano_side[idx],
        'mano_betas': self._mano_betas[idx],
        'extrinsics': self.extrinsics[idx],
        # 'mano_poses': self.poses[idx]
        # 'hand_3d_joints': self.hand_3d_joints[idx],
        'video_name': f'{_SUBJECTS[id_sub]}_{_SERIALS[id_ser]}_{id_seq}_{start_frame}_{seq_len}.mp4',
    }
    return sample
  
  def __len__(self):
    """Returns the number of samples in the dataset."""
    return len(self._mapping)

import h5py
import multiprocessing as mp
from manopth.manolayer import ManoLayer
mano_model_folder = "/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano"

@torch.no_grad()
def pca_coeffs_to_axis45(coeffs_pca: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    """
    coeffs_pca: (B, x)  的 PCA x系数
    返回:     (B, 45)  的 axis-angle（仅手指15关节；不含全局3维）
    """
    basis = layer.th_selected_comps
    mean  = layer.th_hands_mean     
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    if basis.shape == (coeffs_pca.shape[-1], 45):    
        pose45 = coeffs_pca @ basis                  
    elif basis.shape == (45, coeffs_pca.shape[-1]):  
        pose45 = coeffs_pca @ basis.t()              
    else:
        raise ValueError(f"Unexpected basis shape: {basis.shape}")
    pose45 = pose45 + mean                       
    return pose45

import imageio.v2 as iio
import json
# import threading

def write_videos(video_args):
    split,this_data_dir,start_frame,seq_len,video_name = video_args
    video_path = os.path.join("/public/datasets/handdata/dexycb/videos_v4", split, video_name)
    writer = iio.get_writer(video_path, fps=30,macro_block_size=None)
    for i in range(start_frame, start_frame + seq_len):
        img_path = os.path.join(this_data_dir, f"color_{i:06d}.jpg")
        rgb_img = iio.imread(img_path)
        writer.append_data(rgb_img)
    writer.close()

import pickle

def _init_worker():
    worker_train_dataset = DexYCBDataset(split='train', setup='s_all_unseen')
    worker_val_dataset = DexYCBDataset(split='val', setup='s_all_unseen')
    worker_test_dataset = DexYCBDataset(split='test', setup='s_all_unseen')
    global worker_datasets
    worker_datasets = {
            'train': worker_train_dataset,
            'val': worker_val_dataset,
            'test': worker_test_dataset
    }
    global mano_layer_pca_coeff_left
    global mano_layer_pca_coeff_right
    mano_layer_pca_coeff_left = ManoLayer(
        flat_hand_mean=False,
        ncomps=45,
        side='left',
        mano_root=mano_model_folder,
    )
    mano_layer_pca_coeff_right = ManoLayer(
        flat_hand_mean=False,
        ncomps=45,
        side='right',
        mano_root=mano_model_folder,
    )


tmp_pkl_dir = "/public/datasets/handdata/tmp_dexycb"
os.makedirs(tmp_pkl_dir, exist_ok=True)
def worker(arg):
    split, idx = arg
    pkl_dict = {}
    sample = worker_datasets[split][idx]
    group_name = f"{split}_{idx}"
    poses_list = []
    joints_list = []
    
    for i in range(sample['start_frame'], sample['start_frame'] + sample['seq_len']):
        mano_labels_file = os.path.join(sample['data_dir'], f'labels_{i:06d}.npz')
        label = np.load(mano_labels_file)
        mano_pose = label['pose_m']
        mano_joint = label['joint_3d']
        assert mano_pose.shape == (1, 51), f"Expected mano_pose shape (1, 51), got {mano_pose.shape}"
        assert mano_joint.shape == (1,21, 3), f"Expected mano_joint shape (21, 3), got {mano_joint.shape}"
        poses_list.append(mano_pose)
        joints_list.append(mano_joint)
    # print("!!!   ", idx)
    video_name = sample['video_name']
    pkl_dict['video_name'] = video_name
    # Write poses
    np_pose = np.array(poses_list)
    pose_45_batch=torch.Tensor(np_pose[:,0,3:48])
    # print("!!!   ", idx)
    if sample['mano_side'] == 'left':
        pose_45 = pca_coeffs_to_axis45(pose_45_batch, mano_layer_pca_coeff_left)
    else:
        pose_45 = pca_coeffs_to_axis45(pose_45_batch, mano_layer_pca_coeff_right)
    # print("!!   ", idx)
    np_pose[:,0,3:48] = pose_45.numpy()
    np_joints = np.array(joints_list)

    pkl_dict['mano_poses'] = np_pose
    pkl_dict['mano_joint_3d'] = np_joints

    # Write intrinsics
    pkl_dict['intrinsics'] = sample['intrinsics']

    # Write mano_side
    pkl_dict['mano_side'] = sample['mano_side']

    # Write mano_betas
    pkl_dict['mano_betas'] = sample['mano_betas']

    # Write extrinsics
    pkl_dict['extrinsics'] = sample['extrinsics']

    pkl_path = os.path.join(tmp_pkl_dir, f"{group_name}.pkl")

    with open(pkl_path, 'wb') as pf:
        pickle.dump(pkl_dict, pf)
    # print("!!   ", idx)
    return 0




def save_splits2hdf5(hdf5_path = ''):
    # json_path = {
    #    'train': hdf5_path.replace('.hdf5','_json/train.json'),
    #    'val': hdf5_path.replace('.hdf5', '_json/val.json'),
    #    'test': hdf5_path.replace('.hdf5', '_json/test.json'),
    # }
    video_todo = []
    train_dataset = DexYCBDataset(split='train', setup='s_all_unseen')
    val_dataset = DexYCBDataset(split='val', setup='s_all_unseen')
    test_dataset = DexYCBDataset(split='test', setup='s_all_unseen')
    datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
    }
    json_dir = hdf5_path.replace('.hdf5','_json')
    os.makedirs(json_dir, exist_ok=True)
    # file_list={
    #     'train': [],
    #     'val': [],
    #     'test': []
    # } 

    # output_hdf5 = h5py.File(hdf5_path, "w")


    # shapes = {
    #     # 'index': (1,),
    #     # 'data_dir': (1,),  # single data directory path
    #     'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
    #     # 'start_frame': (1,),  # single start frame
    #     'intrinsics': (4,),  # fx,fy,ppx,ppy
    #     'mano_side': (1,),  # single side (left or right)
    #     'mano_betas': (10,),  # 10 betas for MANO model 
    #     'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
    #     'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
    #     # 'rgb_frames': (64, 3, 480, 640),  # 64 frames, RGB images of size 480x640
    #     'video_name': (1,),  # single video name
    #     'mask': (64,),  # mask for valid frames
    # }
    str_dt = h5py.string_dtype('utf-8', None)
    dtypes = {
        # 'index': 'i4',
        # 'data_dir':str_dt,
        # 'start_frame':'i4',
        'intrinsics':'f4',
        'mano_side':str_dt,
        'mano_betas':'f4',
        'extrinsics':'f4',
        'mano_poses':'f4',
        'mano_joint_3d':'f4',
        # 'rgb_frames':'f4',
        'video_name': str_dt,
        # 'mask':'i1',
    }
    todo_list=[]

    for split in ['test','train', 'val' ]:
        for idx in range(len(datasets[split])):
            todo_list.append((split,idx))

    with mp.Pool(processes=16, initializer=_init_worker, initargs=()) as pool:
        # tqdm pool
        for _ in tqdm.tqdm(pool.imap_unordered(worker, todo_list), total=len(todo_list), desc="Processing samples"):
            pass

    dict_file_list={
        'train': {},
        'val': {},
        'test': {}
    }
    with h5py.File(hdf5_path, 'w') as output_hdf5:
        for split, idx in tqdm.tqdm(todo_list, desc="Writing to HDF5"):
            group_name = f"{split}_{idx}"
            group = output_hdf5.create_group(group_name)
            pkl_path = os.path.join(tmp_pkl_dir, f"{group_name}.pkl")
            with open(pkl_path, 'rb') as pf:
                pkl_dict = pickle.load(pf)
            dict_file_list[split][group_name] = len(pkl_dict['mano_poses'])
            if not os.path.exists(os.path.join("/public/datasets/handdata/dexycb/videos_v4", split, pkl_dict['video_name'])):
                video_todo.append((split,datasets[split][idx]['data_dir'],datasets[split][idx]['start_frame'],datasets[split][idx]['seq_len'],pkl_dict['video_name']))
            for key in dtypes.keys():
                data = pkl_dict[key]
                group.create_dataset(key, data=data, dtype=dtypes[key])

    for split in ['train', 'val', 'test']:
        with open(os.path.join(json_dir, f"{split}.json"), 'w') as jf:
            json.dump(dict_file_list[split], jf)  
    
            
    # for split in ['test','train', 'val' ]:
    #     for idx in tqdm.tqdm(range(len(datasets[split])), desc=f"Collecting file list for {split} split"):
    #         sample = datasets[split][idx]
    #         group_name = f"{split}_{idx}"
    #         file_list[split].append(group_name)
    #         group = output_hdf5.create_group(group_name)
    #         # group.create_dataset('index', data=idx, dtype=dtypes['index'])
    #         for key in dtypes.keys():
    #             data = sample[key]
    #             group.create_dataset(key, data=data, dtype=dtypes[key])



    # breakpoint()  # Debugging breakpoint to inspect the datasets

    # print(len(datasets['train']))

    # data['pose_m'].shape
    # (74, 1, 51)
    # (Pdb) data['pose_y'].shape
    # (74, 5, 7)

    # breakpoint()

    # video_todo = []

    # video_root = "/public/datasets/handdata/dexycb/videos_v4"
    # os.makedirs(video_root, exist_ok=True)
    # with h5py.File(hdf5_path, 'w') as f:
    #     for split in ['test','train', 'val' ]:
    #         # os.makedirs(os.path.join(video_root, split), exist_ok=True)
    #         # print(f"Saving split '{split}' to {hdf5_path}...")
    #         N = len(datasets[split])
    #         grp = f.create_group(split)              

    #         for key in shapes.keys():
    #             shape = (N,) + shapes[key]
    #             chunks = (1,) + shapes[key]
    #             ds = grp.create_dataset(
    #                 name=key,
    #                 shape=shape,
    #                 dtype=dtypes[key],
    #                 compression=compression,
    #                 chunks=chunks
    #             )

    #         # 按索引写入剩余样本
    #         for idx, sample in enumerate(tqdm.tqdm(datasets[split], desc=f"Processing {split} split")):
    #             # Write data_dir
    #             # ds_data_dir = grp['data_dir']
    #             # ds_data_dir[idx] = sample['data_dir']

    #             # ds_start_frame = grp['start_frame']
    #             # ds_start_frame[idx] = sample['start_frame']

    #             poses_list = []
    #             joints_list = []
    #             # img_list = []
    #             for i in range(sample['start_frame'], sample['start_frame'] + sample['seq_len']):
    #                 mano_labels_file = os.path.join(sample['data_dir'], f'labels_{i:06d}.npz')
    #                 # mano_joint_file = os.path.join(sample['data_dir'], 'mano_joints', f'joints_{i:06d}.npy')
    #                 label = np.load(mano_labels_file)
    #                 mano_pose = label['pose_m']
    #                 mano_joint = label['joint_3d']
    #                 assert mano_pose.shape == (1, 51), f"Expected mano_pose shape (1, 51), got {mano_pose.shape}"
    #                 assert mano_joint.shape == (1,21, 3), f"Expected mano_joint shape (21, 3), got {mano_joint.shape}"
    #                 poses_list.append(mano_pose)
    #                 joints_list.append(mano_joint)
    #                 # img_path = os.path.join(sample['data_dir'], rgb_format.format(i))
    #                 # rgb_img = iio.imread(img_path)
    #                 # img_list.append(rgb_img)

    #             # ds_rgb_frames = grp['rgb_frames']
    #             # ds_rgb_frames[idx] = np.array(img_list).transpose(0,3,1,2)

    #             ds_video_name = grp['video_name']
    #             video_name = sample['video_name']
    #             ds_video_name[idx] = video_name
    #             video_todo.append((split,sample['data_dir'],sample['start_frame'],sample['seq_len'],video_name))
    #             # Write poses
    #             ds_mano_poses = grp['mano_poses']
    #             np_pose = np.array(poses_list)
    #             pose_45_batch=torch.Tensor(np_pose[:,0,3:48])
    #             if sample['mano_side'] == 'left':
    #                 pose_45 = pca_coeffs_to_axis45(pose_45_batch, mano_layer_pca_coeff_left)
    #             else:
    #                 pose_45 = pca_coeffs_to_axis45(pose_45_batch, mano_layer_pca_coeff_right)
    #             np_pose[:,0,3:48] = pose_45.numpy()
    #             np_joints = np.array(joints_list)

    #             mask_np = np.ones((sample['seq_len']), dtype=np.uint8)
    #             if sample['seq_len'] < 64:
    #                 pad_len = 64 - sample['seq_len']
    #                 pad_np = np.zeros((pad_len, 1, 51), dtype=np.float32)
    #                 np_pose = np.concatenate([np_pose, pad_np], axis=0)
    #                 pad_np = np.zeros((pad_len, 1, 21, 3), dtype=np.float32)
    #                 np_joints = np.concatenate([np_joints, pad_np], axis=0)
    #                 mask_pad = np.zeros((pad_len,), dtype=np.uint8)
    #                 mask_np = np.concatenate([mask_np, mask_pad], axis=0)


    #             ds_mano_poses[idx] = np_pose

    #             ds_mano_joint_3d = grp['mano_joint_3d']
    #             ds_mano_joint_3d[idx] = np_joints

    #             # Write intrinsics
    #             ds_intrinsics = grp['intrinsics']
    #             ds_intrinsics[idx] = sample['intrinsics']

    #             # Write mano_side
    #             ds_mano_side = grp['mano_side']
    #             ds_mano_side[idx] = sample['mano_side']

    #             # Write mano_betas
    #             ds_mano_betas = grp['mano_betas']
    #             ds_mano_betas[idx] = sample['mano_betas']

    #             # Write extrinsics
    #             ds_extrinsics = grp['extrinsics']
    #             ds_extrinsics[idx] = sample['extrinsics']

    #             ds_mask = grp['mask']
    #             ds_mask[idx] = mask_np

    with mp.Pool(processes=64) as pool:
        list(tqdm.tqdm(pool.imap(write_videos, video_todo), total=len(video_todo), desc="Writing videos"))

if __name__ == "__main__":
    save_splits2hdf5(
        hdf5_path="/public/datasets/handdata/dexycb_v6.hdf5",
        # txt_path="/public/datasets/handdata/dexycb_v6.txt",
        # compression='gzip'
    )
    print("HDF5 file created successfully.")
    # for split in ['train', 'val', 'test']:
    #   dataset = DexYCBDataset(split=split, setup='s_all_unseen', data_dir="/public/home/group_ucb/yunqili/data/dexycb")
    #   # breakpoint()

    #   print('Dataset size: {}'.format(len(dataset)))

    #   sample = dataset[554]
    #   print('555th sample:')
    #   # breakpoint()
    #   print(json.dumps(sample, indent=4))
    #   print(f"Max length: {dataset.max_len}, Min length: {dataset.min_len}")
    
