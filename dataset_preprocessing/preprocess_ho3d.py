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
from torch import nn
import torch.nn.functional as F

import json

pca_dim=6


class HO3D_v3(torch.utils.data.Dataset):
    def __init__(
        self,
        # img_dir: Path="/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images",
        # anno_dir: Path="/public/datasets/handdata/interhand26m/anno/annotation",
        data_root="/data-share/share/HO3D/HO3D_v3",
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
            self._mapping.append((seq_name, start_idx, length))
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
    json_dir = hdf5_path.replace('.hdf5','_json')
    os.makedirs(json_dir, exist_ok=True)
    right_mano_layer = ManoLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        side='right',
        mano_root="/data-share/share/handdata/mano/",
    )
    dict_file_train = {}
    str_dt = h5py.string_dtype('utf-8', None)
    dtypes = {
        'intrinsics':'f4',
        'mano_side':str_dt,
        'mano_betas':'f4',
        'extrinsics':'f4',
        'mano_poses':'f4',
        'mano_joint_3d':'f4',
        'video_name': str_dt,
        # 'mask':'i1',
    }
    with h5py.File(hdf5_path, 'w') as output_hdf5:
        for split in ['train']:
            dataset = HO3D_v3(split=split)
            # print(f"Processing split '{split}'...")
            # N = len(dataset)
            # # breakpoint()
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
                dict_file_train[group_name] = len(sample['mano_pose'])
                group = output_hdf5.create_group(group_name)
                
                key = 'video_name'
                group.create_dataset(key, data=sample['video_name'], dtype=dtypes[key])
                
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

                key = 'mano_joint_3d'
                group.create_dataset(key, data=joint_array.reshape(-1,1,21,3), dtype=dtypes[key])

                pose_51_right_tensor = torch.cat(pose_51_right_list, dim=0)
                # pad to 64
                if pose_51_right_tensor.shape[0] < 64:
                    pad_len = 64 - pose_51_right_tensor.shape[0]
                    pad_tensor = pose_51_right_tensor[-1,:].unsqueeze(0).repeat(pad_len,1)
                    pose_51_right_tensor = torch.cat([pose_51_right_tensor, pad_tensor], dim=0)
                # ds_mano_poses = grp['mano_poses']
                # ds_mano_poses[idx] = pose_51_right_tensor.unsqueeze(1)
                key = 'mano_poses'
                group.create_dataset(key, data=pose_51_right_tensor.unsqueeze(1), dtype=dtypes[key])

                # ds_mano_side = grp['mano_side']
                # ds_mano_side[idx] = 'right'
                key = 'mano_side'
                group.create_dataset(key, data='right', dtype=dtypes[key])

                # ds_intrinsics = grp['intrinsics']
                # ds_intrinsics[idx] = sample['intrinsics']
                key = 'intrinsics'
                group.create_dataset(key, data=sample['intrinsics'], dtype=dtypes[key])

                # ds_mano_betas = grp['mano_betas']
                # ds_mano_betas[idx] = sample['mano_betas'].reshape(10)
                key = 'mano_betas'
                group.create_dataset(key, data=sample['mano_betas'].reshape(10), dtype=dtypes[key])

                # ds_extrinsics = grp['extrinsics']
                # ds_extrinsics[idx] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32)
                key = 'extrinsics'
                group.create_dataset(key, data=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32), dtype=dtypes[key])

            print(f"→ Split '{split}' saved {len(dataset)} samples to HDF5.")



    json_path = os.path.join(json_dir, 'train.json')
    with open(json_path, 'w') as f:
        json.dump(dict_file_train, f, indent=4)
    return

def worker_save_video(split, seq, idx, length):
    video_root = "/data-share/share/handdata/preprocessed/ho3d/picked_videos"
    video_name = f"seq{seq}_start{idx}_len{length}.mp4"
    video_path = os.path.join(video_root, split, video_name)
    if os.path.exists(video_path):
        return
    all_frames = []
    rgb_dir = os.path.join("/data-share/share/HO3D/HO3D_v3", split, seq, 'rgb')
    for idx in range(idx, idx + length):
        frame_path = os.path.join(rgb_dir, f"{idx:04d}.jpg")
        image = iio.imread(frame_path)
        all_frames.append(image)
    iio.imwrite(video_path, all_frames, fps=30, codec="libx264", macro_block_size=None)

        

# A:
# manolayer(orient,pose)



mano_layer_pca_coeff_right = ManoLayer(
    flat_hand_mean=False,
    ncomps=pca_dim if pca_dim>0 else 45,
    use_pca=True if pca_dim>0 else False,
    side='right',
    mano_root="/data-share/share/handdata/mano/"
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
        raise ValueError(f"Unexpected basis shape: {basis.shape}, coeffs_pca shape: {coeffs_pca.shape}")
    pose45 = pose45 + mean                       
    return pose45

def robust_l1(x, delta=0.01):
    # Huber loss with small delta (nearly L1 but smooth near 0)
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device))
    lin = absx - quad
    return 0.5 * quad**2 / delta + lin


def moving_average_T(x, window):
    """
    x: (T, D)
    window: int
    returns: (T, D)
    """
    T, D = x.shape
    pad = window // 2

    # conv1d needs (N, C, L)
    x_ = x.transpose(0, 1).unsqueeze(0)   # (1, D, T)

    kernel = torch.ones(D, 1, window, device=x.device) / window  # (D, 1, window)

    # depthwise convolution
    out = F.conv1d(x_, kernel, padding=pad, groups=D)  # (1, D, T)

    return out.squeeze(0).transpose(0, 1)  # back to (T, D)

def save_evaluation2hdf5(hdf5_path, compression: str = 'gzip'):
    json_dir = hdf5_path.replace('.hdf5','_json')
    os.makedirs(json_dir, exist_ok=True)
    dict_file_evaluation = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    back_layer = ManoLayer(
        flat_hand_mean=False,       # 和你训练/使用时的配置保持一致
        side='right',               # left/right 要对上
        mano_root="/data-share/share/handdata/mano/",   # 模型路径
        ncomps=pca_dim if pca_dim>0 else 45,
        use_pca=True if pca_dim>0 else False,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)
    
    def center_joints(joints: torch.Tensor, vertices:torch.Tensor) -> torch.Tensor:
        # joints: (B, 21, 3)
        center = joints[:, 0:1, :]  # 根关节位置
        centered_joints = joints - center  # 减去根关节位置
        centered_vertices = vertices - center  # 减去根关节位置
        return centered_joints, centered_vertices
    
    def vertices_to_pose_shape(V_obs: torch.Tensor, joint_arr, layer: ManoLayer) -> torch.Tensor:
        # optim_pose = torch.optim.Adam, lr=5e-2
        # optim_betas = torch.optim.Adam, lr=1e-2
        rot_dim = 3
        pose_dim = pca_dim if pca_dim>0 else 45

        basis = layer.th_selected_comps
        mean  = layer.th_hands_mean  

        if pca_dim > 0:
            init_pose = (basis @ mean.reshape(-1,1)).reshape(pose_dim).repeat(V_obs.shape[0],1).to(device)
        else:
            # init_pose = mean.reshape(pose_dim).repeat(V_obs.shape[0],1).to(device)
            init_pose = torch.zeros(V_obs.shape[0], pose_dim).to(device)
        init_rot = torch.zeros(V_obs.shape[0], 3).to(device)
        init_pose_params = torch.cat([init_rot, init_pose], dim=1)
        # pose_params = init_pose_params.clone().detach().to(device).requires_grad_(True)
        
        init_rot = init_rot.detach().to(device).requires_grad_(True)
        init_pose = init_pose.detach().to(device).requires_grad_(True)
        

        betas       = torch.zeros(1, 10, device=device, requires_grad=True)
        trans_for_copy = torch.mean(V_obs, dim=1)
        trans = trans_for_copy.clone().detach().to(device).requires_grad_(True)
        # optim_hand = torch.optim.Adam([pose_params, betas], lr=0.05)
        optim_hand = torch.optim.Adam([
            {"params": init_rot, "lr": 0.2},
            {"params": init_pose, "lr": 0.02},
            {"params": betas, "lr": 0.05},
        ], lr=0.05)
        optim_trans = torch.optim.Adam([trans], lr=0.02)
        # optim_both = torch.optim.Adam([pose_params, betas, trans], lr=0.003)
        # optim_both = torch.optim.Adam([init_rot, init_pose, betas, trans], lr=0.003)
        # λ_v, λ_p, λ_s = 1.0, 1e-5, 1e-5
        # def pose_prior(hand_pose_vec):
        #     return (hand_pose_vec[:, rot_dim:] - init_pose).pow(2).mean()
        for optim in [optim_hand, optim_trans]:
            if optim == optim_trans:
                print("Start optimizing translation...")
                joint_arr_to_use, V_obs_to_use = joint_arr.clone(), V_obs.clone()
            # elif optim == optim_both:
            #     print("Start optimizing hand pose, shape and translation...")
            #     joint_arr_to_use, V_obs_to_use = joint_arr.clone(), V_obs.clone()
            elif optim == optim_hand:
                print("Start optimizing hand pose and shape...")
                joint_arr_to_use, V_obs_to_use = center_joints(joint_arr.clone(), V_obs.clone())

                
            for it in range(301):
                # if optim == optim_hand and it == 200: pose_params[:, 3:] == pose_params[:, 3:].mean(dim=0, keepdim=True) # new initialization
                if optim == optim_hand and it == 200: init_pose == init_pose.mean(dim=0, keepdim=True) # new initialization
                
                optim.zero_grad()
                pose_params = torch.cat([init_rot, init_pose], dim=1)
                
                V_pred, Jtr = layer(pose_params, betas.repeat(V_obs_to_use.shape[0],1), trans)
                
                if optim == optim_hand:
                    Jtr, V_pred = center_joints(Jtr, V_pred)
                
                loss_v = robust_l1((V_obs_to_use*1000 - V_pred)).mean()
                loss_j = robust_l1((Jtr - joint_arr_to_use*1000)).mean()
                # loss_j = torch.where(loss_j > 3.0, loss_j*10.0, loss_j).mean()
                # .mean()
                loss_smooth = robust_l1(pose_params[:-1] - pose_params[1:]).mean()
                
                window_size=7
                # convolution
                smoothed_pose = moving_average_T(init_pose, window_size)
                loss_window_smooth = robust_l1(init_pose - smoothed_pose).mean()
                
                loss_prior = robust_l1(pose_params[:, 3:]-init_pose).mean()
                loss_beta_prior = (betas**2).mean()
                
                loss = loss_v + loss_j + loss_smooth + loss_prior + 1e-2*loss_beta_prior + 0.1 * loss_window_smooth
                # loss = loss_j
                loss.backward()
                optim.step()

                if it % 200 == 0:
                    print(f"mean vertices: {loss_v.item():.6f}")
                    print(f'mean jtr: {loss_j.item():.6f}')
                    # print(f"iter {it+1}: loss_v {loss_v.item():.6f}, loss_p {loss_p.item():.6f}, loss_s {loss_s.item():.6f}")
                    print(f"iter {it+1}: loss_v {loss_v.item():.6f}")
        pose_params = torch.cat([init_rot, init_pose], dim=1)
        return pose_params.detach().cpu(), betas.detach().cpu(), trans.detach().cpu()

    evaluation_txt = "/data-share/share/HO3D/HO3D_v3/evaluation.txt"
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
    resplit_seqs = good_seqs
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
    }
    
    json_eval_file = "/data-share/share/handdata/ho3d/evaluation_verts.json"
    with open(json_eval_file, 'r') as f:
        json_data = json.load(f)
    
    joint_json_file = "/data-share/share/handdata/ho3d/evaluation_xyz.json"
    with open(joint_json_file, 'r') as f:
        joint_json_data = json.load(f)

    # resplit_seqs = resplit_seqs[:1]  # debug use only first 10 sequences
    with h5py.File(hdf5_path, 'w') as output_hdf5:
        for idx in tqdm.tqdm(range(len(resplit_seqs)), desc="Processing evaluation split"):
            grp = output_hdf5.create_group(f"evaluation_{idx}")

            evaluation_start_idx, seq_name, start_idx, save_length = resplit_seqs[idx]
            dict_file_evaluation[f"evaluation_{idx}"] = save_length

            # ds_data_dir = grp['video_name']
            # ds_data_dir[idx] = mp4_format.format(seq_name, start_idx, save_length)
            key = 'video_name'
            grp.create_dataset(key, data=mp4_format.format(seq_name, start_idx, save_length), dtype=dtypes[key])

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
            key = 'mano_joint_3d'
            grp.create_dataset(key, data=joint_array.reshape(-1,1,21,3), dtype=dtypes[key])

            V_obs = torch.from_numpy(vertices_array).float().to(device)
            joint_array_tensor = torch.from_numpy(joint_array).float().to(device)
            out_pose_params, out_betas, out_trans = vertices_to_pose_shape(V_obs, joint_array_tensor, back_layer)     

            out_global_orient = out_pose_params[:, :3].cpu().numpy()  # save_length x 3
            out_hand_pose = out_pose_params[:, 3:].cpu().numpy()  # save_length x 10
            out_transl = out_trans.cpu().numpy()  # save_length x 3
            # pca 15 to none pca 45
            if pca_dim > 0:
                out_hand_pose_45 = pca_coeffs_to_axis45(torch.FloatTensor(out_hand_pose), mano_layer_pca_coeff_right).reshape(save_length, 45)
            else:
                out_hand_pose_45 = torch.FloatTensor(out_hand_pose)
            pose_51_right_list = torch.cat([torch.FloatTensor(out_global_orient), out_hand_pose_45, torch.FloatTensor(out_transl)], dim=1)
            key = 'mano_poses'
            grp.create_dataset(key, data=pose_51_right_list.unsqueeze(1), dtype=dtypes[key]) # (-1, 1, 51)
            key = 'mano_betas'
            grp.create_dataset(key, data=out_betas.reshape(10), dtype=dtypes[key])

            # pose_51_right_list = torch.zeros((save_length, 51))
            # key = 'mano_poses'
            # grp.create_dataset(key, data=pose_51_right_list.unsqueeze(1), dtype=dtypes[key]) # (-1, 1, 51)

            # key = 'mano_betas'
            # grp.create_dataset(key, data=torch.zeros((10,)), dtype=dtypes[key])


            key = 'mano_side'
            grp.create_dataset(key, data='right', dtype=dtypes[key])

            data_root="/data-share/share/HO3D/HO3D_v3"
            meta_file = os.path.join(data_root, 'evaluation', seq_name, 'meta', f"{start_idx:04d}.pkl")
            with open(meta_file, 'rb') as f:
                meta_data = pickle.load(f)

            intrinsics = meta_data['camMat']

            key = 'intrinsics'
            grp.create_dataset(key, data=torch.FloatTensor([intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]]), dtype=dtypes[key])


            key = 'extrinsics'
            grp.create_dataset(key, data=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32), dtype=dtypes[key])

    json_path = os.path.join(json_dir, 'evaluation.json')
    with open(json_path, 'w') as f:
        json.dump(dict_file_evaluation, f, indent=4)
    task_list = [( 'evaluation', seq, idx,length) for (eval_idx, seq, idx, length) in resplit_seqs]
    num_workers = 32
    os.makedirs(os.path.join("/data-share/share/handdata/preprocessed/ho3d/picked_videos", 'evaluation'), exist_ok=True)
    with mp.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.starmap(worker_save_video, task_list), total=len(task_list), desc=f"Saving evaluation videos"))


import multiprocessing as mp

def save_videos(split = "train"):
    dataset = HO3D_v3(split=split)
    video_root = "/data-share/share/handdata/preprocessed/ho3d/picked_videos"
    split_video_root = os.path.join(video_root, split)
    os.makedirs(split_video_root, exist_ok=True)
    task_list = [(split, seq, idx,length) for (seq, idx, length) in dataset._mapping]
    num_workers = 32
    # tqdm mp task_list
    with mp.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.starmap(worker_save_video, task_list), total=len(task_list), desc=f"Saving {split} videos"))


if __name__ == "__main__":
    # save_splits2hdf5(
    #     hdf5_path="/data-share/share/handdata/preprocessed/ho3d/ho3d_train.hdf5",
    #     compression='gzip'
    # )
    # save_evaluation2hdf5(
    #     hdf5_path="/data-share/share/handdata/preprocessed/ho3d/ho3d_evaluation.hdf5",
    #     compression='gzip'
    # )
    save_videos(split='train')
