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

    
    self.max_len=0
    self.min_len=1000000

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
                        all_arrangements.append((id_sub, id_ser, id_seq))
            self._mapping = np.array(all_arrangements)
        if self._split == 'val':
            subject_ind = [9]  # 验证使用单个受试者
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]  # 验证使用所有相机
            sequence_ind = list(range(100))  # 验证使用所有序列
            all_arrangements = []
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        all_arrangements.append((id_sub, id_ser, id_seq))
            self._mapping = np.array(all_arrangements)
        if self._split == 'test':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 测试使用多个受试者
            serial_ind = [7]  # 测试使用所有相机
            sequence_ind = list(range(100))  # 测试使用所有序列
            all_arrangements = []
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        all_arrangements.append((id_sub, id_ser, id_seq))
            sequence_ind = [i for i in range(100) if i // 5 in (3,)]
            serial_ind = [0, 1, 2, 3, 4, 5, 6]
            for id_sub in subject_ind:
                for id_ser in serial_ind:
                    for id_seq in sequence_ind:
                        all_arrangements.append((id_sub, id_ser, id_seq))
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

    for id_sub, id_ser, id_seq in tqdm.tqdm(self._mapping, desc="Loading dataset"):
        meta_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
                                 os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub]))[id_seq],"meta.yml")
        # pose_file = os.path.join(self._data_dir, _SUBJECTS[id_sub],
        #                          os.listdir(os.path.join(self._data_dir, _SUBJECTS[id_sub]))[id_seq], "pose.npz")
        with open(meta_file, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        # self._num_frames.append(meta['num_frames'])
        num_frames = meta['num_frames']
        start_frame = num_frames // 2 - 32
        self._start_frames.append(start_frame)
        if meta['num_frames'] > self.max_len:
            self.max_len = meta['num_frames']
        if meta['num_frames'] < self.min_len:
            self.min_len = meta['num_frames']
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
    id_sub, id_ser, id_seq = self._mapping[idx]
    d = os.path.join(self._data_dir, _SUBJECTS[id_sub])
    d = os.path.join(d, os.listdir(d)[id_seq], 
                     _SERIALS[id_ser])
    sample = {
        'data_dir': d,
        # 'num_frames': self._num_frames[idx],
        'start_frame': self._start_frames[idx],
        'intrinsics': self._intrinsics[id_ser],
        'mano_side': self._mano_side[idx],
        'mano_betas': self._mano_betas[idx],
        'extrinsics': self.extrinsics[idx],
        # 'mano_poses': self.poses[idx]
        # 'hand_3d_joints': self.hand_3d_joints[idx],
    }
    return sample
  
  def __len__(self):
    """Returns the number of samples in the dataset."""
    return len(self._mapping)

import h5py

def save_splits2hdf5(hdf5_path = '', compression: str = 'gzip'):
    train_dataset = DexYCBDataset(split='train', setup='s_all_unseen')
    val_dataset = DexYCBDataset(split='val', setup='s_all_unseen')
    test_dataset = DexYCBDataset(split='test', setup='s_all_unseen')
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    shapes = {
        'data_dir': (1,),  # single data directory path
        'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
        'start_frame': (1,),  # single start frame
        'intrinsics': (4,),  # fx,fy,ppx,ppy
        'mano_side': (1,),  # single side (left or right)
        'mano_betas': (10,),  # 10 betas for MANO model 
        'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
        'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
        # 'rgb_frames': (64, 3, 480, 640),  # 64 frames, RGB images of size 480x640
    }

    str_dt = h5py.string_dtype('utf-8', None)
    dtypes = {
        'data_dir':str_dt,
        'start_frame':'i4',
        'intrinsics':'f4',
        'mano_side':str_dt,
        'mano_betas':'f4',
        'extrinsics':'f4',
        'mano_poses':'f4',
        'mano_joint_3d':'f4',
        # 'rgb_frames':'f4',
    }

    # breakpoint()  # Debugging breakpoint to inspect the datasets

    print(len(datasets['train']))

    # data['pose_m'].shape
    # (74, 1, 51)
    # (Pdb) data['pose_y'].shape
    # (74, 5, 7)

    # breakpoint()

    

    with h5py.File(hdf5_path, 'w') as f:
        for split in ['train', 'val', 'test']:
            print(f"Saving split '{split}' to {hdf5_path}...")
            N = len(datasets[split])
            grp = f.create_group(split)              

            for key in shapes.keys():
                shape = (N,) + shapes[key]
                chunks = (1,) + shapes[key]
                ds = grp.create_dataset(
                    name=key,
                    shape=shape,
                    dtype=dtypes[key],
                    compression=compression,
                    chunks=chunks
                )

            # 按索引写入剩余样本
            for idx, sample in enumerate(tqdm.tqdm(datasets[split], desc=f"Processing {split} split")):
                # Write data_dir
                ds_data_dir = grp['data_dir']
                ds_data_dir[idx] = sample['data_dir']

                ds_start_frame = grp['start_frame']
                ds_start_frame[idx] = sample['start_frame']

                poses_list = []
                joints_list = []
                for i in range(sample['start_frame'], sample['start_frame'] + 64):
                    mano_labels_file = os.path.join(sample['data_dir'], f'labels_{i:06d}.npz')
                    # mano_joint_file = os.path.join(sample['data_dir'], 'mano_joints', f'joints_{i:06d}.npy')
                    label = np.load(mano_labels_file)
                    mano_pose = label['pose_m']
                    mano_joint = label['joint_3d']
                    assert mano_pose.shape == (1, 51), f"Expected mano_pose shape (1, 51), got {mano_pose.shape}"
                    assert mano_joint.shape == (1,21, 3), f"Expected mano_joint shape (21, 3), got {mano_joint.shape}"
                    poses_list.append(mano_pose)
                    joints_list.append(mano_joint)

                # Write poses
                ds_mano_poses = grp['mano_poses']
                ds_mano_poses[idx] = np.array(poses_list)

                ds_mano_joint_3d = grp['mano_joint_3d']
                ds_mano_joint_3d[idx] = np.array(joints_list)

                # Write intrinsics
                ds_intrinsics = grp['intrinsics']
                ds_intrinsics[idx] = sample['intrinsics']

                # Write mano_side
                ds_mano_side = grp['mano_side']
                ds_mano_side[idx] = sample['mano_side']

                # Write mano_betas
                ds_mano_betas = grp['mano_betas']
                ds_mano_betas[idx] = sample['mano_betas']

                # Write extrinsics
                ds_extrinsics = grp['extrinsics']
                ds_extrinsics[idx] = sample['extrinsics']

            print(f"→ Split '{split}' saved {N} samples.")


if __name__ == "__main__":
    save_splits2hdf5(
        hdf5_path="/public/datasets/handdata/dexycb_s_all_unseen.hdf5",
        compression='gzip'
    )
    # for split in ['train', 'val', 'test']:
    #   dataset = DexYCBDataset(split=split, setup='s_all_unseen', data_dir="/public/home/group_ucb/yunqili/data/dexycb")
    #   # breakpoint()

    #   print('Dataset size: {}'.format(len(dataset)))

    #   sample = dataset[554]
    #   print('555th sample:')
    #   # breakpoint()
    #   print(json.dumps(sample, indent=4))
    #   print(f"Max length: {dataset.max_len}, Min length: {dataset.min_len}")
    
