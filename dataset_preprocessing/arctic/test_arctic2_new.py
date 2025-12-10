
import multiprocessing
from common.body_models import construct_layers
from src.arctic.preprocess_dataset import construct_loader
from glob import glob
from tqdm import tqdm
import os
# mano_ps = glob(f"/public/datasets/handdata/arctic/unpack/arctic_data/data/raw_seqs/*/*.mano.npy")
raw_seq_path = "/data-share/share/handdata/arctic/unpack/arctic_data/data/raw_seqs"
cropped_img_path = "/data-share/share/handdata/arctic/unpack/arctic_data/data/cropped_images"
# pbar = tqdm(mano_ps)
misc_p = "/data-share/share/handdata/arctic/unpack/arctic_data/data/meta/misc.json"
import json
with open(misc_p, "r") as f:
    misc = json.load(f)

data_split = {
    "train": ["s01", "s02", "s04", "s05", "s06", "s07", "s08"],
    "val": ["s09"],
    "test": ["s10"],
}

import h5py
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

hdf5_path="/public/datasets/handdata/arctic_v8.hdf5"

json_dir = hdf5_path.replace(".hdf5", "_json")
os.makedirs(json_dir, exist_ok=True)
dict_file_list={
    'train': {},
    'val': {},
    'test': {}
}

import imageio.v2 as iio
import cv2

resplit_p = "/data-share/share/handdata/preprocessed/record_arctic_valid_seqs_32_64_resplit_xxx.json"

with open(resplit_p, "r") as f:
    resplit_dict = json.load(f)

import torch
from manopth.manolayer import ManoLayer
import numpy as np

@torch.no_grad()
def add_mean_pose45(coeffs45: torch.Tensor, layer: ManoLayer) -> torch.Tensor:
    """
    coeffs45: (B, 45)  的 axis-angle（仅手指15关节；不含全局3维）
    返回:     (B, 45)  的 axis-angle（仅手指15关节；不含全局3维）
    """
    mean  = layer.th_hands_mean      # 形如 (1,45) 或 (45,)
    # 统一 mean 形状为 (1,45)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)  # (1,45)
    coeffs45 = coeffs45 + mean                          # 广播加均值
    return coeffs45

@torch.no_grad()
def cal_root_j(th_betas, layer: ManoLayer) -> torch.Tensor:
    th_v_shaped = torch.matmul(layer.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + layer.th_v_template
    th_j = torch.matmul(layer.th_J_regressor, th_v_shaped)
    return th_j[:, 0, :].contiguous().view(3, 1)

mano_model_folder = "/data-share/share/handdata/mano/"
mano_layer_right = ManoLayer(
    flat_hand_mean=False,
    ncomps=45,
    use_pca=False,
    side="right",
    mano_root=mano_model_folder,
)
mano_layer_left = ManoLayer(
    flat_hand_mean=False,
    ncomps=45,
    use_pca=False,
    side="left",
    mano_root=mano_model_folder,
)
mano_layer_joint_left = ManoLayer(
    flat_hand_mean=True,
    ncomps=45,
    use_pca=False,
    side="left",
    mano_root=mano_model_folder,
)
mano_layer_joint_right = ManoLayer(
    flat_hand_mean=True,
    ncomps=45,
    use_pca=False,
    side="right",
    mano_root=mano_model_folder,
)
compression = "gzip"

from multiprocessing.pool import ThreadPool

def worker(idx_info):
    try:
        this_grp = {
            'left':{
            'intrinsics':None,
            'mano_side':None,
            'mano_betas':None,
            'extrinsics':None,
            'mano_poses':None,
            'mano_joint_3d':None,
            'video_name': None,
        },
        'right':{
            'intrinsics':None,
            'mano_side':None,
            'mano_betas':None,
            'extrinsics':None,
            'mano_poses':None,
            'mano_joint_3d':None,
            'video_name': None,
        }}
        idx, s_idx, mano_name, start_idx, saved_length, cam_id = idx_info
        this_mano_p = os.path.join(raw_seq_path, s_idx, f"{mano_name}.mano.npy")
        this_processed_p = os.path.join('/data-share/share/handdata/arctic/outputs/processed/seqs', s_idx, f"{mano_name}.npy")
        # if not os.path.exists(this_processed_p):
        #     breakpoint()
        processed_data = np.load(this_processed_p, allow_pickle=True).item()
        bboxes = processed_data['bbox'][start_idx:start_idx+saved_length, cam_id, :]  # (T, 3)
        # processed_data['bbox'].shape
        # (617, 9, 3)
        loader = construct_loader(this_mano_p)
        batch = next(iter(loader))
        world2cam = np.array(misc[s_idx]['world2cam'][cam_id-1])

        # world2cam_rot = world2cam[:3,:3]
        # world2cam_T = world2cam[:3,3]
        # cam2world_rot = world2cam_rot.T
        # cam2world_T = -cam2world_rot @ world2cam_T
        cam2world_rot = world2cam[:3,:3]
        cam2world_T = world2cam[:3,3]
        extrinsics = np.concatenate([cam2world_rot, cam2world_T[:,None]], axis=1)
        for handtype in ['right', 'left']:
            if handtype == 'right':
                idx_save = idx*2
                mano_poses_45 = batch['pose_r'][start_idx:start_idx+saved_length]

                mano_betas = torch.mean(batch['shape_r'], dim=0)  # (10)
                mano_poses_45 = add_mean_pose45(mano_poses_45, mano_layer_right)
                root_j = cal_root_j(mano_betas[None,:], mano_layer_right)  # (3,1)
                world2mano_rot = batch['rot_r'][start_idx:start_idx+saved_length]
                world2mano_trans = batch['trans_r'][start_idx:start_idx+saved_length]
                mano_for_joint = mano_layer_joint_right
            else:
                idx_save = idx*2+1
                mano_poses_45 = batch['pose_l'][start_idx:start_idx+saved_length]
                mano_betas = torch.mean(batch['shape_l'], dim=0)  # (10)
                mano_poses_45 = add_mean_pose45(mano_poses_45, mano_layer_left)
                root_j = cal_root_j(mano_betas[None,:], mano_layer_left)  # (3,1)
                world2mano_rot = batch['rot_l'][start_idx:start_idx+saved_length]
                world2mano_trans = batch['trans_l'][start_idx:start_idx+saved_length]
                mano_for_joint = mano_layer_joint_left
            # if saved_length < 64:
            #     pad_len = 64 - saved_length
            #     mano_poses_45 = torch.cat([mano_poses_45, torch.zeros((pad_len, 45), dtype=mano_poses_45.dtype)], dim=0)
            #     world2mano_rot = torch.cat([world2mano_rot, torch.zeros((pad_len, 3), dtype=world2mano_rot.dtype)], dim=0)
            #     world2mano_trans = torch.cat([world2mano_trans, torch.zeros((pad_len, 3), dtype=world2mano_trans.dtype)], dim=0)
            this_grp[handtype]['video_name'] = f"{s_idx}_{mano_name}_{start_idx}_{saved_length}_{cam_id}.mp4"
            intr_mat = misc[s_idx]['intris_mat'][cam_id-1] # (3,3)list
            fx = intr_mat[0][0]
            fy = intr_mat[1][1]
            ppx = intr_mat[0][2]
            ppy = intr_mat[1][2]
            all_intrinsics = []
            for box in bboxes:
                crop_dim = box[2]*300
                s = 1000/crop_dim
                fx_s = fx * s
                fy_s = fy * s
                ppx_s = s * (ppx - box[0] + 0.5*crop_dim)
                ppy_s = s * (ppy - box[1] + 0.5*crop_dim)
                all_intrinsics.append([fx_s, fy_s, ppx_s, ppy_s])

            # ppx = 500
            # ppy = 500  # because the image is resized to half
            this_grp[handtype]['intrinsics'] = np.array(all_intrinsics)
            this_grp[handtype]['mano_side']= handtype
            this_grp[handtype]['mano_betas'] = mano_betas.numpy()
            this_grp[handtype]['extrinsics']= extrinsics
            world2cam_rot_matrix = []
            for w2m_rot in world2mano_rot:
                w2m_rot_matrix, _ = cv2.Rodrigues(w2m_rot.numpy())
                w2m_rot_matrix = torch.tensor(w2m_rot_matrix, dtype=torch.float32)
                world2cam_rot_matrix.append(w2m_rot_matrix)
            world2mano_rot_matrix = torch.stack(world2cam_rot_matrix, dim=0)
            cam2mano_rot = torch.tensor(cam2world_rot, dtype=torch.float32) @ world2mano_rot_matrix  # (T, 3, 3)
            cam2mano_rot_angle = []
            for c2m_rot in cam2mano_rot:
                c2m_rot_angle, _ = cv2.Rodrigues(c2m_rot.numpy())
                cam2mano_rot_angle.append(torch.tensor(c2m_rot_angle, dtype=torch.float32))
            cam2mano_rot_angle = torch.stack(cam2mano_rot_angle, dim=0)
            # (3,3)@(T,3,1) + (T,3,1) = (T,3,1)
            cam2mano_trans = torch.tensor(cam2world_rot, dtype=torch.float32) @ torch.tensor(world2mano_trans.numpy(), dtype=torch.float32).reshape(cam2mano_rot_angle.shape[0],3,1) + torch.tensor(cam2world_T, dtype=torch.float32).reshape(3,1)
            delta = -root_j + torch.tensor(cam2world_rot,dtype=root_j.dtype) @ root_j
            mano_pose_51 = torch.cat([cam2mano_rot_angle.squeeze(2), mano_poses_45, (cam2mano_trans+delta).squeeze(2)], dim=1)  # (T, 51)
            # if first 3 in cam2mano_rot_angle is 0
            # if torch.norm(cam2mano_rot_angle[0]) < 1e-6:
            #     breakpoint()
            this_grp[handtype]['mano_poses']= mano_pose_51.numpy().reshape(-1, 1, 51)
            
            _, joints = mano_for_joint(
                mano_pose_51[:,:48],
                mano_betas.repeat(mano_pose_51.shape[0], 1),
                mano_pose_51[:,48:51]
            )
            this_grp[handtype]['mano_joint_3d'] = joints.numpy().reshape(-1, 1, 21, 3)/1000
            # mask = np.zeros((64,), dtype=np.uint8)
            # mask[:saved_length] = 1
            # this_grp[handtype]['mask'] = mask
        return idx, this_grp
    except Exception as e:
        return idx, e


with h5py.File(hdf5_path, 'w') as f:
    for split in ['train', 'val', 'test']:
        split_N = 2*len(resplit_dict[split])
        # grp = f.create_group(split)
        # for key in shapes.keys():
        #     shape = (split_N,) + shapes[key]
        #     chunks = (1,) + shapes[key] 
        #     ds = grp.create_dataset(
        #         name=key,
        #         shape=shape,
        #         dtype=dtypes[key],
        #         compression=compression,
        #         chunks=chunks
        #     )
        task_list = []
        print(f"Starting processing {split} data...")
        for idx, info_dict in enumerate(resplit_dict[split]):
            s_idx, mano_name, start_idx, saved_length, cam_id = info_dict['info']
            task_list.append((idx, s_idx, mano_name, start_idx, saved_length, cam_id))
        print(f"Total {len(task_list)} tasks for {split} data.")
        with ThreadPool(16) as pool:
            print("Worker pool created.")
            results = pool.imap_unordered(worker, task_list)
            print("All tasks submitted.")
            for ret_idx, this_grp in tqdm(results, total=len(task_list), desc=f"Processing {split}"):
                # print(f"Writing result for task {ret_idx}")
                if type(this_grp) is Exception:
                    print(f"Error processing idx {ret_idx}: {this_grp}")
                    break
                for handtype in ['left', 'right']:
                    if handtype == 'left':
                        idx_save = ret_idx*2+1
                    else:
                        idx_save = ret_idx*2
                    group_name = f"{split}_{idx_save}"
                    dict_file_list[split][group_name] = len(this_grp[handtype]['intrinsics'])
                    grp = f.create_group(group_name)
                    for key in dtypes.keys():
                        grp.create_dataset(key, data=this_grp[handtype][key], dtype=dtypes[key])

        json_path = os.path.join(json_dir, f"{split}.json")
        with open(json_path, "w") as jf:
            json.dump(dict_file_list[split], jf)

        print(f"Finished writing {split} data to {hdf5_path}")
