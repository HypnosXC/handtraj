import os
# import yaml
import numpy as np
# import torch
# import json
import tqdm

import h5py
from manopth.manolayer import ManoLayer
import imageio.v3 as iio
import json
import torch
from pathlib import Path
from typing import Any, Literal
import time
import pickle

import json
class HO3D_v3(torch.utils.data.Dataset):
    def __init__(
        self,
        # img_dir: Path="/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images",
        # anno_dir: Path="/public/datasets/handdata/interhand26m/anno/annotation",
        data_root="/public/datasets/handdata/HO3D_v3",
        split: Literal["train"] = "train",
    ) -> None:
        start_time = time.time()
        self.data_root = data_root
        self._mapping = []
        self._subseq_len = 64
        self.split = split
        tmp_good_list = []
        for seq_name in sorted(os.listdir(os.path.join(data_root, self.split))):
            start_tmp=-1
            tmp_length= 0 
            this_rgb_dir = os.path.join(data_root, self.split, seq_name, 'rgb')
            file_list = sorted(os.listdir(this_rgb_dir), key=lambda x: int(x[:-4]))
            for idx, file_name in enumerate(file_list):
                if file_name != f"{idx:04d}.jpg":
                    breakpoint()
                meta_file = os.path.join(data_root, self.split, seq_name, 'meta', f"{idx:04d}.pkl")
                with open(meta_file, 'rb') as f:
                    meta_data = pickle.load(f)
                    # if 'handPose' not in dict
                    # if 'handPose' not in meta_data:
                    # breakpoint()
                    if 'handPose' not in meta_data or meta_data['handPose'] is None:
                        if start_tmp != -1:
                            tmp_good_list.append((seq_name, start_tmp, tmp_length))
                            start_tmp = -1
                            tmp_length = 0
                    else:
                        if start_tmp == -1:
                            start_tmp = idx
                            tmp_length = 1
                        else:
                            tmp_length += 1
            if start_tmp != -1:
                tmp_good_list.append((seq_name, start_tmp, tmp_length))
                start_tmp = -1
                tmp_length = 0
        # breakpoint()
        for (seq_name, start_idx, length) in tmp_good_list:
            if length < self._subseq_len:
                if length >= self._subseq_len // 2:
                    self._mapping.append((seq_name, start_idx, length))
                continue
            else:
                num_subseq = length // self._subseq_len
                for sub_idx in range(num_subseq):
                    self._mapping.append((seq_name, start_idx + sub_idx * self._subseq_len, self._subseq_len))
                if length % self._subseq_len != 0:
                    self._mapping.append((seq_name, start_idx + length - self._subseq_len, self._subseq_len))

        self.mp4_format = "seq{}_start{}_len{}.mp4"
        print(f"HO3D_v3 Dataset '{self.split}' initialized with {len(self._mapping)} samples in {time.time() - start_time:.2f} seconds.")

    def __getitem__(self, index: int):
        seq, start_idx, save_length = self._mapping[index]
        hand_pose_list = []
        hand_beta_list = []
        hand_trans_list = []
        cam_mat_list = []
        hand_joints3d_list = []
        for idx in range(start_idx, start_idx + save_length):
            meta_file = os.path.join(self.data_root, self.split, seq, 'meta', f"{idx:04d}.pkl")
            with open(meta_file, 'rb') as f:
                meta_data = pickle.load(f)
                hand_pose_list.append(meta_data['handPose'])
                # if meta_data['handPose'] is None:
                #     breakpoint()
                hand_beta_list.append(meta_data['handBeta'])
                hand_trans_list.append(meta_data['handTrans'])
                cam_mat_list.append(meta_data['camMat'])
                hand_joints3d_list.append(meta_data['handJoints3D'])
        cam_mat_list = np.array(cam_mat_list[0])  # 3x3
        kwargs: dict[str, Any] = {}
        kwargs['video_name'] = self.mp4_format.format(seq, start_idx, save_length)
        # kwargs['mano_joint_3d'] = torch.FloatTensor(self.cam_joint_data[index]['joint']).reshape(-1,21,3)
        kwargs['mano_joint_3d'] = torch.FloatTensor(np.array(hand_joints3d_list)).reshape(-1,21,3)
        kwargs['mano_pose'] = torch.FloatTensor(np.array(hand_pose_list)).reshape(-1,48)
        kwargs['mano_trans'] = torch.FloatTensor(np.array(hand_trans_list)).reshape(-1,3)
        # mean of betas
        # breakpoint()
        kwargs['mano_betas'] = torch.FloatTensor(np.array(hand_beta_list[0])).reshape(10)
        # cal mean
        kwargs['mano_side'] = 'right'
        kwargs['intrinsics'] = torch.FloatTensor([cam_mat_list[0,0], cam_mat_list[1,1], cam_mat_list[0,2], cam_mat_list[1,2]])
        return kwargs
    
    def __len__(self) -> int:
        return len(self._mapping)
    

def cal_root_j(th_betas, layer: ManoLayer) -> torch.Tensor:
    th_v_shaped = torch.matmul(layer.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + layer.th_v_template
    th_j = torch.matmul(layer.th_J_regressor, th_v_shaped)
    return th_j[:, 0, :].contiguous().view(3, 1)

import matplotlib.pyplot as plt
import cv2
def save_splits2hdf5(hdf5_path, compression: str = 'gzip'):
    right_mano_layer = ManoLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        side='right',
        mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
    )
    str_dt = h5py.string_dtype('utf-8', None)
    dtypes = {
        'intrinsics':'f4',
        'mano_side':str_dt,
        'mano_betas':'f4',
        'extrinsics':'f4',
        'mano_poses':'f4',
        'mano_joint_3d':'f4',
        'video_name': str_dt,
        'mask':'i1',
    }
    shapes = {
        'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
        'intrinsics': (4,),  # fx,fy,ppx,ppy
        'mano_side': (1,),  # single side (left or right)
        'mano_betas': (10,),  # 10 betas for MANO model 
        'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
        'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
        'video_name': (1,),  # single video name
        'mask': (64,),  # mask for valid frames
    }
    with h5py.File(hdf5_path, 'w') as f:
        for split in ['train']:
            dataset = HO3D_v3(split=split)
            print(f"Processing split '{split}'...")
            N = len(dataset)
            # breakpoint()
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
            for idx, P_idx in enumerate(tqdm.tqdm(range(N), desc=f"Processing {split} split")):
                sample = dataset[P_idx]
                ds_data_dir = grp['video_name']
                ds_data_dir[idx] = sample['video_name']

                
                original_mano_pose_48 = sample['mano_pose']
                original_mano_trans = sample['mano_trans']
                original_mano_trans[:,1] = -original_mano_trans[:,1]
                original_mano_trans[:,2] = -original_mano_trans[:,2]
                pose_51_right_list = []
                joint_list = []
                for frame_id in range(original_mano_pose_48.shape[0]):
                    pose_48 = original_mano_pose_48[frame_id]
                    mano_rot_aa = pose_48[:3].reshape(1,3)
                    mano_rot_mat, _ = cv2.Rodrigues(mano_rot_aa.numpy())
                    mano_rot_mat[1,:] = -mano_rot_mat[1,:]
                    mano_rot_mat[2,:] = -mano_rot_mat[2,:]
                    mano_fix_aa,_ = cv2.Rodrigues(mano_rot_mat)
                    pose_48[:3] = torch.from_numpy(mano_fix_aa).reshape(3)
                    root_j = cal_root_j(sample['mano_betas'].unsqueeze(0), right_mano_layer)
                    delta = -root_j + torch.tensor([1,0,0,0,-1,0,0,0,-1],dtype=root_j.dtype).reshape(3,3) @ root_j
                    pose_51_right = torch.cat([pose_48, original_mano_trans[frame_id]+delta.reshape(3)], dim=0).unsqueeze(0)
                    pose_51_right_list.append(pose_51_right)

                    _, joints = right_mano_layer(
                        pose_51_right[:, :48],
                        sample['mano_betas'].repeat(pose_51_right.shape[0], 1),
                        pose_51_right[:, 48:]
                    )
                    joints = joints/1000.0  # to meters
                    joint_list.append(joints.detach().cpu().numpy())
                joint_array = np.array(joint_list)
                # pad to 64
                if joint_array.shape[0] < 64:
                    pad_len = 64 - joint_array.shape[0]
                    joint_array = np.pad(joint_array, ((0,pad_len),(0,0),(0,0),(0,0)), mode='edge')
                ds_mano_joint_3d = grp['mano_joint_3d']
                ds_mano_joint_3d[idx] = joint_array.reshape(64,1,21,3)

                pose_51_right_tensor = torch.cat(pose_51_right_list, dim=0)
                # pad to 64
                if pose_51_right_tensor.shape[0] < 64:
                    pad_len = 64 - pose_51_right_tensor.shape[0]
                    pad_tensor = pose_51_right_tensor[-1,:].unsqueeze(0).repeat(pad_len,1)
                    pose_51_right_tensor = torch.cat([pose_51_right_tensor, pad_tensor], dim=0)
                ds_mano_poses = grp['mano_poses']
                ds_mano_poses[idx] = pose_51_right_tensor.unsqueeze(1)

                ds_mano_side = grp['mano_side']
                ds_mano_side[idx] = 'right'

                ds_intrinsics = grp['intrinsics']
                ds_intrinsics[idx] = sample['intrinsics']

                ds_mano_betas = grp['mano_betas']
                ds_mano_betas[idx] = sample['mano_betas'].reshape(10)

                ds_extrinsics = grp['extrinsics']
                ds_extrinsics[idx] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32)

                ds_mask = grp['mask']
                if sample['mano_pose'].shape[0] < 64:
                    ds_mask[idx] = np.array([1]*sample['mano_pose'].shape[0] + [0]*(64 - sample['mano_pose'].shape[0]), dtype=np.int8)
                else:
                    ds_mask[idx] = np.ones((64,), dtype=np.int8)

            print(f"→ Split '{split}' saved {N} samples.")

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

def worker_save_video(split, seq, idx, length):
    video_root = "/public/datasets/handdata/HO3D_v3_new/picked_videos_v2"
    video_name = f"seq{seq}_start{idx}_len{length}.mp4"
    video_path = os.path.join(video_root, split, video_name)
    if os.path.exists(video_path):
        return
    # breakpoint()
    all_frames = []
    rgb_dir = os.path.join("/public/datasets/handdata/HO3D_v3", split, seq, 'rgb')
    for idx in range(idx, idx + length):
        frame_path = os.path.join(rgb_dir, f"{idx:04d}.jpg")
        image = iio.imread(frame_path)
        all_frames.append(image)
    iio.imwrite(video_path, all_frames, fps=30, codec="libx264", macro_block_size=None)

        

# A:
# manolayer(orient,pose)

mano_layer_pca_coeff_right = ManoLayer(
    flat_hand_mean=False,
    ncomps=15,
    use_pca=True,
    side='right',
    mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models'
)

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


def save_evaluation2hdf5(hdf5_path, compression: str = 'gzip'):
    # from mano_fitter import ManoSequenceFitter
    # fitter_right = ManoSequenceFitter(
    #     mano_model_dir='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',           # <-- change to your MANO folder
    #     mano_side='right',
    #     use_pca=True,                          # set True if you want PCA pose
    #     ncomps=15,
    #     iters=800,
    #     w_joint=1.0,
    #     w_pose_prior=0.001,
    #     w_shape_prior=0.04,
    #     w_trans_smooth=0.1,                    # try >0 for sequences
    #     w_pose_smooth=0.01
    # )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 1) 你的观测顶点 (1, 778, 3)；确保拓扑/顺序与 MANO 一致、单位一致（MANO默认mm）
    # V_obs = torch.from_numpy(vobs_np).float().to(device)  # <- 替换成你的数据

    back_layer = ManoLayer(
        flat_hand_mean=False,       # 和你训练/使用时的配置保持一致
        side='right',               # left/right 要对上
        mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',   # 模型路径
        ncomps=15,                  # 若用 PCA，这里要与下方 pose 维度匹配
        use_pca=True,              # 也可 False（45 轴角）
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)

    # 2) 待优化变量
    # rot_dim = 3                    # root 轴角
    # pose_dim = (15 if back_layer.use_pca else 45)  # 关节子空间
    # pose_params = torch.zeros(1, rot_dim + pose_dim, device=device, requires_grad=True)
    # betas       = torch.zeros(1, 10, device=device, requires_grad=True)   # 可先不优化
    # trans       = torch.zeros(1, 1, 3, device=device, requires_grad=True) # MANO不含平移，这里显式加

    # # 3) 优化器与损失权重
    # optim = torch.optim.Adam([pose_params, betas, trans], lr=5e-2)
    # λ_v, λ_p, λ_s = 1.0, 1e-3, 1e-4

    # def pose_prior(hand_pose_vec):
    #     # 简单 L2 先验；如果 use_pca=True，这里就是 PCA 系数的 L2
    #     return (hand_pose_vec[:, rot_dim:]**2).mean()

    # for it in range(800):  # 迭代次数按收敛情况调
    #     optim.zero_grad()
    #     V_pred, Jtr = layer(pose_params, th_betas=betas)   # V_pred: (1, 778, 3)
    #     V_pred = V_pred + trans                            # 显式全局平移

    #     loss_v = ((V_pred - V_obs)**2).mean()
    #     loss_p = pose_prior(pose_params)
    #     loss_s = (betas**2).mean()
    #     loss = λ_v*loss_v + λ_p*loss_p + λ_s*loss_s
    #     loss.backward()
    #     optim.step()

    #     if (it+1) % 100 == 0:
    #         print(f"iter {it+1}: loss {loss.item():.6f}, v {loss_v.item():.6f}")

    def vertices_to_pose_shape(V_obs: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
        # optim_pose = torch.optim.Adam, lr=5e-2
        # optim_betas = torch.optim.Adam, lr=1e-2
        rot_dim = 3
        pose_dim = (15 if layer.use_pca else 45)

        basis = layer.th_selected_comps
        mean  = layer.th_hands_mean  
        init_pose = (basis @ mean.reshape(-1,1)).reshape(pose_dim).repeat(V_obs.shape[0],1).to(device)
        init_rot = torch.zeros(V_obs.shape[0], 3).to(device)
        init_pose_params = torch.cat([init_rot, init_pose], dim=1)
        # print("Initial pose params norm:", init_pose_params)
        pose_params = init_pose_params.clone().detach().to(device).requires_grad_(True)


        # pose_params = torch.zeros(V_obs.shape[0], rot_dim + pose_dim, device=device, requires_grad=True)
        betas       = torch.zeros(1, 10, device=device, requires_grad=True)   # 可先不优化
        # trans       = torch.zeros(V_obs.shape[0], 1, 3, device=device, requires_grad=True) 
        # initialize trans with the mean of V_obs, with grad enabled
        trans_for_copy = torch.mean(V_obs, dim=1)
        trans = trans_for_copy.clone().detach().to(device).requires_grad_(True)
        optim = torch.optim.Adam([pose_params, betas, trans], lr=5e-2)
        λ_v, λ_p, λ_s = 1.0, 1e-5, 1e-5
        def pose_prior(hand_pose_vec):
            return (hand_pose_vec[:, rot_dim:] - init_pose).pow(2).mean()
        for it in range(800):  # 迭代次数按收敛情况调
            optim.zero_grad()
            V_pred, Jtr = layer(pose_params, betas.repeat(V_obs.shape[0],1), trans)
            loss_v = ((V_pred - V_obs*1000)**2).mean()
            # loss_p = pose_prior(pose_params)
            # loss_s = (betas**2).mean()
            # loss = λ_v*loss_v + λ_p*loss_p + λ_s*loss_s
            loss = loss_v
            loss.backward()
            optim.step()

            if it % 100 == 0:
                print(f"mean vertices: {(V_obs*1000-V_pred).abs().mean().item():.6f}")
                # print(f"iter {it+1}: loss_v {loss_v.item():.6f}, loss_p {loss_p.item():.6f}, loss_s {loss_s.item():.6f}")
                print(f"iter {it+1}: loss_v {loss_v.item():.6f}")
        return pose_params.detach().cpu(), betas.detach().cpu(), trans.detach().cpu()

    evaluation_txt = "/public/datasets/handdata/HO3D_v3/evaluation.txt"
    eval_file_list = []
    with open(evaluation_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('/')
            eval_file_list.append((line[0], int(line[1])))
    good_seqs = []
    tmp_seq = ''
    tmp_start_id = -1
    tmp_length = 0
    tmp_idx_evaluation = -1
    for idx, (seq_name, file_id) in enumerate(eval_file_list):
        
        if seq_name == tmp_seq and file_id == tmp_start_id + tmp_length:
            tmp_length += 1
        else:
            if tmp_seq != '':
                good_seqs.append((tmp_idx_evaluation, tmp_seq, tmp_start_id, tmp_length))
            tmp_idx_evaluation = idx
            tmp_seq = seq_name
            tmp_start_id = file_id
            tmp_length = 1
    if tmp_seq != '':
        good_seqs.append((tmp_idx_evaluation,tmp_seq, tmp_start_id, tmp_length))
    resplit_seqs = []
    for (idx_start_evaluation, seq_name, start_idx, length) in good_seqs:
        if length < 64:
            if length >= 32:
                resplit_seqs.append((idx_start_evaluation,seq_name, start_idx, length))
            continue
        else:
            num_subseq = length // 64
            for sub_idx in range(num_subseq):
                resplit_seqs.append((idx_start_evaluation + sub_idx*64, seq_name, start_idx + sub_idx * 64, 64))
            if length % 64 != 0:
                resplit_seqs.append((idx_start_evaluation + length - 64, seq_name, start_idx + length - 64, 64))
    mp4_format = "seq{}_start{}_len{}.mp4"
    str_dt = h5py.string_dtype('utf-8', None)
    dtypes = {
        'intrinsics':'f4',
        'mano_side':str_dt,
        'extrinsics':'f4',

        'mano_betas':'f4',
        'mano_poses':'f4',
        
        'mano_joint_3d':'f4',
        'video_name': str_dt,
        'mask':'i1',
    }
    
    shapes = {
        'intrinsics': (4,),  # fx,fy,ppx,ppy
        'mano_side': (1,),  # single side (left or right)
        'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
        'mano_betas': (10,),  # 10 betas for MANO model
        'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
        'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
        'video_name': (1,),  # single video name
        'mask': (64,),  # mask for valid frames
    }
    json_eval_file = "/public/home/group_ucb/yunqili/.cache/huggingface/hub/datasets--AnnieLiyunqi--verts_ho3d/snapshots/103cbce86985aa35997f18496f1cd3b02ede0fea/evaluation_verts.json"
    with open(json_eval_file, 'r') as f:
        json_data = json.load(f)
    
    joint_json_file = "/public/datasets/handdata/HO3D_v3/evaluation_xyz.json"
    with open(joint_json_file, 'r') as f:
        joint_json_data = json.load(f)

    # resplit_seqs = resplit_seqs[:11]  # debug use only first 1000 sequences
    with h5py.File(hdf5_path, 'w') as f:
        N = len(resplit_seqs)
        grp = f.create_group('evaluation')
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
        for idx, P_idx in enumerate(tqdm.tqdm(range(N))):
            evaluation_start_idx, seq_name, start_idx, save_length = resplit_seqs[P_idx]
            ds_data_dir = grp['video_name']
            ds_data_dir[idx] = mp4_format.format(seq_name, start_idx, save_length)

            vertices_list = []
            joint_list = []
            for evaluation_id in range(evaluation_start_idx, evaluation_start_idx + save_length):
                vertices_list.append(np.array(json_data[evaluation_id]))
                joint_list.append(np.array(joint_json_data[evaluation_id]))
            vertices_array = np.array(vertices_list)  # save_length x 21 x 3
            vertices_array[:,:,1] = -vertices_array[:,:,1]
            vertices_array[:,:,2] = -vertices_array[:,:,2]
            joint_array = np.array(joint_list)  # save_length x 21 x 3
            joint_array[:,:,1] = -joint_array[:,:,1]
            joint_array[:,:,2] = -joint_array[:,:,2]
            # joint rearrange to match MANO order
            mano_to_ho3d_idx = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            joint_array = joint_array[:,mano_to_ho3d_idx,:]
            # out = fitter_right.fit_sequence(joint_array.reshape(save_length,21,3), optimize_scale=False)
            # breakpoint()

            V_obs = torch.from_numpy(vertices_array).float().to(device)  # <- 替换成你的数据
            out_pose_params, out_betas, out_trans = vertices_to_pose_shape(V_obs, back_layer)
            # if idx==10:
            #     V_obs = V_obs.cpu()
            #     back_layer = back_layer.cpu()
            #     v,j = back_layer(out_pose_params, out_betas.repeat(V_obs.shape[0],1), out_trans)
            #     v = v/1000.0
            #     print(V_obs-v)

            
            
            if joint_array.shape[0] < 64:
                pad_len = 64 - joint_array.shape[0]
                joint_array = np.pad(joint_array, ((0,pad_len),(0,0),(0,0),(0,0)), mode='edge')
            ds_mano_joint_3d = grp['mano_joint_3d']
            ds_mano_joint_3d[idx] = joint_array.reshape(64,1,21,3)

            ds_mano_poses = grp['mano_poses']
            out_global_orient = out_pose_params[:, :3].cpu().numpy()  # save_length x 3
            out_hand_pose = out_pose_params[:, 3:18].cpu().numpy()  # save_length x 10
            out_transl = out_trans.cpu().numpy()  # save_length x 3
            # pca 10 to none pca 45
            out_hand_pose_45 = pca_coeffs_to_axis45(torch.FloatTensor(out_hand_pose), mano_layer_pca_coeff_right).reshape(save_length, 45)
            pose_51_right_list = torch.cat([torch.FloatTensor(out_global_orient), out_hand_pose_45, torch.FloatTensor(out_transl)], dim=1)
            # pad to 64
            if pose_51_right_list.shape[0] < 64:
                pad_len = 64 - pose_51_right_list.shape[0]
                pad_tensor = pose_51_right_list[-1,:].unsqueeze(0).repeat(pad_len,1)
                pose_51_right_list = torch.cat([pose_51_right_list, pad_tensor], dim=0)
            ds_mano_poses[idx] = pose_51_right_list.unsqueeze(1)

            ds_mano_betas = grp['mano_betas']
            ds_mano_betas[idx] = out_betas.reshape(10)

            ds_mano_side = grp['mano_side']
            ds_mano_side[idx] = 'right'
            data_root="/public/datasets/handdata/HO3D_v3"
            # for frame_id in range(start_idx, start_idx + save_length):
            meta_file = os.path.join(data_root, 'evaluation', seq_name, 'meta', f"{start_idx:04d}.pkl")
            with open(meta_file, 'rb') as f:
                meta_data = pickle.load(f)

            intrinsics = meta_data['camMat']

            ds_intrinsics = grp['intrinsics']
            ds_intrinsics[idx] = torch.FloatTensor([intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]])

            ds_extrinsics = grp['extrinsics']
            ds_extrinsics[idx] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32)

            ds_mask = grp['mask']
            if save_length < 64:
                ds_mask[idx] = np.array([1]*save_length + [0]*(64 - save_length), dtype=np.int8)
            else:
                ds_mask[idx] = np.ones((64,), dtype=np.int8)
    task_list = [( 'evaluation', seq, idx,length) for (eval_idx, seq, idx, length) in resplit_seqs]
    num_workers = 32
    # tqdm mp task_list
    os.makedirs(os.path.join("/public/datasets/handdata/HO3D_v3_new/picked_videos_v2", 'evaluation'), exist_ok=True)
    with mp.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.starmap(worker_save_video, task_list), total=len(task_list), desc=f"Saving evaluation videos"))

        
    # json_eval_file = "/public/datasets/handdata/HO3D_v3/evaluation_xyz.json"
    # with open(json_eval_file, 'r') as f:
    #     json_data = json.load(f)
    


    
import multiprocessing as mp

def save_videos(split = "train"):
    dataset = HO3D_v3(split=split)
    video_root = "/public/datasets/handdata/HO3D_v3_new/picked_videos_v2"
    split_video_root = os.path.join(video_root, split)
    os.makedirs(split_video_root, exist_ok=True)
    task_list = [(split, seq, idx,length) for (seq, idx, length) in dataset._mapping]
    num_workers = 32
    # tqdm mp task_list
    with mp.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.starmap(worker_save_video, task_list), total=len(task_list), desc=f"Saving {split} videos"))
    

if __name__ == "__main__":
    save_splits2hdf5(
        hdf5_path="/public/datasets/handdata/ho3d_v2.hdf5",
        compression='gzip'
    )
    save_evaluation2hdf5(
        hdf5_path="/public/datasets/handdata/ho3d_evaluation_v7.hdf5",
        compression='gzip'
    )
    save_videos(split='train')
    save_videos(split='test')
