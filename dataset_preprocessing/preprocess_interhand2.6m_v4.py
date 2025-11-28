import os
# import yaml
import numpy as np
# import torch
# import json
import tqdm

import h5py
# from data_loaders.mano_layer import loadManoHandModel
from manopth.manolayer import ManoLayer
import imageio.v3 as iio
import json
import torch
from pathlib import Path
from typing import Any, Literal
import time
# with open("/public/datasets/handdata/interhand26m/anno/annotation/val/InterHand2.6M_val_data.json", 'r') as f:
#     val_data = json.load(f)

import json
class Interhand2_6M(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir: Path="/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images",
        anno_dir: Path="/public/datasets/handdata/interhand26m/anno/annotation",
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        start_time = time.time()
        self.save_mp4_dir = os.path.join("/public/datasets/handdata/interhand26m/data/picked_videos",split)
        self.mp4_format = "original_{}_Capture{}_{}_cam{}_{}_{}_{}.mp4" # cap_idx, seq, cam, start_idx, length_this, hand_type
        record_split_seqs_32_list = os.path.join(anno_dir, "..", "resplit_ih26_v4.json")
        cam_mano_dir = os.path.join(anno_dir, split, f'seq_{split}_camera_mano_resplit.json')
        cam_joint_dir = os.path.join(anno_dir, split, f'seq_{split}_camera_joint_resplit.json')
        cam_dirs = {
            "train":os.path.join(anno_dir, 'train', 'InterHand2.6M_train_camera.json'),
            "val":os.path.join(anno_dir, 'val', 'InterHand2.6M_val_camera.json'),
            "test":os.path.join(anno_dir, 'test', 'InterHand2.6M_test_camera.json'),
        }
        with open(record_split_seqs_32_list, 'r') as f:
            # self.record_split_seqs_32 = json.load(f)[split]
            self.record_split_seqs_32 = json.load(f)[split]

        self.split = split
        print("Loading annotation files...")
        with open(cam_mano_dir, 'r') as f:
            self.cam_mano_data = json.load(f)
        print("Camera MANO data loaded.")
        with open(cam_joint_dir, 'r') as f:
            self.cam_joint_data = json.load(f)
        print("Camera joint data loaded.")
        self.camera_data = {}
        for split_cam in cam_dirs:
            with open(cam_dirs[split_cam], 'r') as f:
                self.camera_data[split_cam] = json.load(f)
        #     self.camera_data = json.load(f)
        print("Camera parameters data loaded.")
        self.N = len(self.cam_mano_data)
        self._subseq_len = 64
        self.left_mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='left',
            mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        )
        self.right_mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='right',
            mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        )
        print(f"Dataset initialized with {self.N} samples. Time taken: {time.time() - start_time:.2f} seconds")

    def __getitem__(self, index: int):
        # cap_idx, seq, cam, start_idx, save_length, hand_type = self._mapping[index]
        this_mano_data = self.cam_mano_data[index]
        this_seq_data = self.record_split_seqs_32[index]
        # this_joint_data = self.cam_joint_data[index]['joint']
        # cap_idx, seq, cam, start_idx, save_length, hand_type = this_mano_data['info']
        original_split, cap_idx, seq, cam, start_idx, save_length, hand_type = this_seq_data
        # assert this_mano_data['info'] == [cap_idx, seq, cam, start_idx, save_length, hand_type]

        kwargs: dict[str, Any] = {}
        # mp4_path = os.path.join(self.save_mp4_dir, self.mp4_format.format(cap_idx, seq, cam, start_idx, save_length, hand_type))

        # if not os.path.exists(mp4_path):
        #     breakpoint()

        # kwargs["rgb_frames"] = torch.from_numpy(video)
        # joint = new_joint_data[cap_id][cam_id][frame_id][hand_type]
        this_camera_data = self.camera_data[original_split][str(cap_idx)]
        kwargs['video_name'] = self.mp4_format.format(original_split,cap_idx, seq, cam, start_idx, save_length, hand_type)
        kwargs['mano_joint_3d'] = torch.FloatTensor(self.cam_joint_data[index]['joint']).reshape(-1,21,3)
        kwargs['mano_poses'] = torch.FloatTensor(this_mano_data['pose'])
        # mean of betas
        kwargs['mano_betas'] = torch.FloatTensor(this_mano_data['shape'])
        # cal mean
        kwargs['mano_betas'] = torch.mean(kwargs['mano_betas'], dim=0)
        kwargs['mano_side'] = hand_type
        kwargs['intrinsics'] = torch.FloatTensor([this_camera_data['focal'][cam][0], this_camera_data['focal'][cam][1], this_camera_data['princpt'][cam][0], this_camera_data['princpt'][cam][1]])

        camrot = np.array(this_camera_data['camrot'][cam]).reshape(3,3)
        campos = np.array(this_camera_data['campos'][cam]).reshape(3)/1000

        cam2world_rot = torch.FloatTensor(camrot)
        cam2world_trans = -cam2world_rot @ torch.FloatTensor(campos).reshape(3,1)

        extrinsics_matrix = torch.zeros((3,4))
        extrinsics_matrix[:3,:3] = cam2world_rot
        extrinsics_matrix[:3,3] = cam2world_trans.reshape(3)
        kwargs['extrinsics'] = extrinsics_matrix
        
        # pad mano_pose and mano_joint_3d to 64 and set mask
        # timesteps = kwargs['mano_poses'].shape[0]
        # if timesteps < self._subseq_len:
        #     pad_len = self._subseq_len - timesteps
        #     kwargs['mano_pose'] = torch.cat([kwargs['mano_pose'], torch.zeros((pad_len, 51))], dim=0)
        #     kwargs['mano_joint_3d'] = torch.cat([kwargs['mano_joint_3d'], torch.zeros((pad_len, 21, 3))], dim=0)
        #     # kwargs['rgb_frames'] = torch.cat([kwargs['rgb_frames'], torch.zeros((pad_len, kwargs['rgb_frames'].shape[1], kwargs['rgb_frames'].shape[2], 3), dtype=torch.uint8)], dim=0)
        #     kwargs['mask'] = torch.cat([torch.ones((timesteps,), dtype=torch.bool), torch.zeros((pad_len,), dtype=torch.bool)], dim=0)
        # else:
        kwargs['mask'] = torch.ones((self._subseq_len,), dtype=torch.bool)
        # with h5py.File(self._hdf5_path, "r") as f:
        #     dataset = f[self.split]
        #     # ndarray to tensor
        #     kwargs["mano_betas"]=torch.from_numpy(dataset['mano_betas'][index])
        #     kwargs["mano_pose"]=torch.from_numpy(dataset['mano_poses'][index,:,0])
        #     timesteps= kwargs["mano_pose"].shape[0]
        #     kwargs["mano_joint_3d"] = torch.from_numpy(dataset['mano_joint_3d'][index,:,0])
        #     kwargs["intrinsics"] = torch.from_numpy(dataset['intrinsics'][index])
        #     kwargs["extrinsics"] =torch.from_numpy(dataset['extrinsics'][index])
        #     if dataset['mano_side'][index][0].decode('utf-8') == 'left':
        #         kwargs["mano_side"] = torch.zeros(1)
        #     else:
        #         kwargs["mano_side"] = torch.ones(1)
        #     # data_dir = dataset['data_dir'][index][0].decode('utf-8')
        #     # breakpoint()
        #     # start_frame = dataset['start_frame'][index]
        #     # rgb_frames = []
        #     # for i in range(start_frame[0], start_frame[0] + self._subseq_len):
        #     #     rgb_frame = self.rgb_format.format(i)
        #     #     rgb_path = os.path.join(data_dir, rgb_frame)
        #     #     # open the image and convert it to tensor
        #     #     rgb_image = iio.imread(rgb_path)
        #     #     rgb_frames.append(torch.from_numpy(rgb_image))
            
        #     if self.split == 'test':
        #         kwargs["rgb_frames"] = torch.tensor(dataset['rgb_frames'][index].transpose(0,2,3,1))
        #     else:
        #         kwargs["rgb_frames"] = torch.ones((timesteps,), dtype=torch.bool)
        #     kwargs["mask"] = torch.ones((timesteps,), dtype=torch.bool)
        # breakpoint()
        return kwargs
    
    def __len__(self) -> int:
        return self.N
    
# def world2cam(world_coord, R, T):
#     cam_coord = np.dot(R, world_coord - T)
#     return cam_coord

import matplotlib.pyplot as plt
def plot_distribution(data):
    # frame_cnt_list = [idx for idx in frame_cnt_list if idx<=250]
    plt.hist(data, bins=50)
    plt.title(f"Frame count distribution")
    plt.xlabel("Frame count")
    plt.ylabel("Number of sequences")
    plt.savefig(f"frame_count_distribution.png")
    plt.clf()

def save_splits2hdf5(hdf5_path, compression: str = 'gzip'):
    json_dir = hdf5_path.replace('.hdf5','_json')
    os.makedirs(json_dir, exist_ok=True)
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
    dict_file_list={
        'train': {},
        'val': {},
        'test': {}
    }
    interhand2mano_idx = [20, 3, 2, 1, 0, 7, 6 ,5,4,11,10,9,8,15,14,13,12,19,18,17,16]

    with h5py.File(hdf5_path, 'w') as output_hdf5:
        for split in ['train', 'val', 'test']:
            # print(f"Processing split '{split}'...")
            dataset = Interhand2_6M(split=split)
            # N = len(dataset)
            # grp = f.create_group(split)
            # for key in shapes.keys():
            #     shape = (N,) + shapes[key]
            #     chunks = (1,) + shapes[key] 
            #     ds = grp.create_dataset(
            #         name=key,
            #         shape=shape,
            #         dtype=dtypes[key],
            #         compression=compression,
            #         chunks=chunks
            #     )
            for idx in tqdm.tqdm(range(len(dataset)), desc=f"Processing {split} split"):
                group_name = f"{split}_{idx}"
                sample = dataset[idx]
                dict_file_list[split][group_name] = len(sample['mano_poses'])
                group = output_hdf5.create_group(group_name)
                # Write data_dir
                key = 'video_name'
                group.create_dataset(key, data=sample['video_name'], dtype=dtypes[key])
                # ds_mano_poses = grp['mano_poses']
                # ds_mano_poses = sample['mano_pose'].unsqueeze(1)
                key = 'mano_poses'
                group.create_dataset(key, data=sample['mano_poses'].unsqueeze(1), dtype=dtypes[key])
                key = 'mano_joint_3d'
                group.create_dataset(key, data=sample['mano_joint_3d'].unsqueeze(1)[:,:,interhand2mano_idx,:], dtype=dtypes[key])
                # ds_mano_joint_3d = grp['mano_joint_3d']
                # ds_mano_joint_3d = sample['mano_joint_3d'].unsqueeze(1)[:,:,interhand2mano_idx,:]
                # ds_intrinsics = grp['intrinsics']
                # ds_intrinsics = sample['intrinsics']
                key = 'intrinsics'
                group.create_dataset(key, data=sample['intrinsics'], dtype=dtypes[key])
                # ds_mano_side = grp['mano_side']
                # ds_mano_side = sample['mano_side']
                # ds_mano_betas = grp['mano_betas']
                # ds_mano_betas = sample['mano_betas']
                # ds_extrinsics = grp['extrinsics']
                # ds_extrinsics = sample['extrinsics']
                key = 'mano_side'
                group.create_dataset(key, data=sample['mano_side'], dtype=dtypes[key])
                key = 'mano_betas'
                group.create_dataset(key, data=sample['mano_betas'], dtype=dtypes[key])
                key = 'extrinsics'
                group.create_dataset(key, data=sample['extrinsics'], dtype=dtypes[key])

    for split in ['train', 'val', 'test']:
        with open(os.path.join(json_dir, f"{split}.json"), 'w') as jf:
            json.dump(dict_file_list[split], jf)  

                    

    return
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    shapes = {
        # 'index': (1,),
        # 'data_dir': (1,),  # single data directory path
        'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
        # 'start_frame': (1,),  # single start frame
        'intrinsics': (4,),  # fx,fy,ppx,ppy
        'mano_side': (1,),  # single side (left or right)
        'mano_betas': (10,),  # 10 betas for MANO model 
        'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
        'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
        'rgb_frames': (64, 3, 480, 640),  # 64 frames, RGB images of size 480x640
    }

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
        'rgb_frames':'f4',
    }

    # breakpoint()  # Debugging breakpoint to inspect the datasets

    print(len(datasets['train']))

    # data['pose_m'].shape
    # (74, 1, 51)
    # (Pdb) data['pose_y'].shape
    # (74, 5, 7)

    # breakpoint()

    rgb_format = "image{:05d}.jpg"

    with h5py.File(hdf5_path, 'w') as f:
        for split in ['train', 'val', 'test']:
            img_root = os.path.join(interhand_img_dir, split)
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
        hdf5_path="/public/datasets/handdata/interhand26m_v4.hdf5",
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
    
