# edit a hdf5 file, change a part of it, and save
import h5py
import numpy as np
import os
from src.arctic.preprocess_dataset import construct_loader
import tqdm

import multiprocessing as mp
from multiprocessing import Pool



file_path = '/public/datasets/handdata/arctic_v3.hdf5'
# with h5py.File(file_path, 'r+') as f:
#     for split in ['train', 'val', 'test']:
#         group = f[split]
#         video_names = [group['video_name'][index][0].decode('utf-8') for index in range(len(group['video_name']))]
#         # f"{s_idx}_{mano_name}_{start_idx}_{saved_length}_{cam_id}.mp4"
#         split_N = len(video_names)
#         ds_intrinsics = group['original_intrinsics']

#         # ds_original_intr = group.create_dataset(
#         #     name='original_intrinsics',
#         #     shape=ds_intrinsics.shape,
#         #     dtype=ds_intrinsics.dtype,
#         #     compression='gzip',
#         #     chunks=(1,4)
#         # )
#         # ds_original_intr[:] = ds_intrinsics[:]
#         # remove intrinsics in the group and create new intrinsics with new shape
#         # del group['intrinsics']
        
#         # shape = (split_N,64,4)
#         # chunks = (1,64,4)
#         # ds_new_intr = group.create_dataset(
#         #     name='intrinsics',
#         #     shape=shape,
#         #     dtype='f4',
#         #     compression='gzip',
#         #     chunks=chunks
#         # )
#         ds_new_intr = group['intrinsics']
#         for idx, video_name in tqdm.tqdm(enumerate(video_names), total=split_N):
#             s_idx = video_name.split('_')[0]
#             mano_name = video_name.split('_')[1:-3]
#             mano_name = '_'.join(mano_name)
#             start_idx = int(video_name.split('_')[-3])
#             saved_length = int(video_name.split('_')[-2])
#             cam_id = int(video_name.split('_')[-1].split('.')[0])
#             original_intrinsics = ds_intrinsics[idx]
#             this_processed_p = os.path.join('/public/datasets/handdata/arctic/outputs/processed/seqs', s_idx, f"{mano_name}.npy")
#             processed_data = np.load(this_processed_p, allow_pickle=True).item()
#             bboxes = processed_data['bbox'][start_idx:start_idx+saved_length, cam_id, :]
#             all_intrinsics = []
#             for box in bboxes:
#                 fx, fy, ppx, ppy = original_intrinsics
#                 crop_dim = box[2]*200
#                 s = 1000/crop_dim  # scale factor
#                 fx_s = fx * s
#                 fy_s = fy * s
#                 ppx_s = s * (ppx - box[0] + 0.5*crop_dim)
#                 ppy_s = s * (ppy - box[1] + 0.5*crop_dim)
#                 all_intrinsics.append([fx_s, fy_s, ppx_s, ppy_s])
#             all_intrinsics = np.array(all_intrinsics, dtype='f4')  # (saved_length, 4)
#             # pad to (64,4)
#             if saved_length < 64:
#                 pad_len = 64 - saved_length
#                 pad_intrinsics = np.zeros((pad_len,4), dtype='f4')
#                 all_intrinsics = np.concatenate([all_intrinsics, pad_intrinsics], axis=0)
#             ds_new_intr[idx] = all_intrinsics


# process and save to hdf5 simultaneously
def process_and_save(idx_video_tuple):
    idx, video_name = idx_video_tuple
    s_idx = video_name.split('_')[0]
    mano_name = video_name.split('_')[1:-3]
    mano_name = '_'.join(mano_name)
    start_idx = int(video_name.split('_')[-3])
    saved_length = int(video_name.split('_')[-2])
    cam_id = int(video_name.split('_')[-1].split('.')[0])
    original_intrinsics = ds_intrinsics[idx]
    this_processed_p = os.path.join('/public/datasets/handdata/arctic/outputs/processed/seqs', s_idx, f"{mano_name}.npy")
    processed_data = np.load(this_processed_p, allow_pickle=True).item()
    bboxes = processed_data['bbox'][start_idx:start_idx+saved_length, cam_id, :]
    all_intrinsics = []
    for box in bboxes:
        breakpoint()
        fx, fy, ppx, ppy = original_intrinsics
        crop_dim = box[2]*200
        s = 1000/crop_dim  # scale factor
        fx_s = fx * s
        fy_s = fy * s
        ppx_s = s * (ppx - box[0] + 0.5*crop_dim)
        ppy_s = s * (ppy - box[1] + 0.5*crop_dim)
        all_intrinsics.append([fx_s, fy_s, ppx_s, ppy_s])
    all_intrinsics = np.array(all_intrinsics, dtype='f4')  # (saved_length, 4)
    # pad to (64,4)
    if saved_length < 64:
        pad_len = 64 - saved_length
        pad_intrinsics = np.zeros((pad_len,4), dtype='f4')
        all_intrinsics = np.concatenate([all_intrinsics, pad_intrinsics], axis=0)
    return idx, all_intrinsics
with h5py.File(file_path, 'r+') as f:
    for split in ['train', 'val', 'test']:
        group = f[split]
        video_names = [group['video_name'][index][0].decode('utf-8') for index in range(len(group['video_name']))]
        split_N = len(video_names)
        ds_intrinsics = group['original_intrinsics']

        ds_original_intr = group.create_dataset(
            name='original_intrinsics',
            shape=ds_intrinsics.shape,
            dtype=ds_intrinsics.dtype,
            compression='gzip',
            chunks=(1,4)
        )
        ds_original_intr[:] = ds_intrinsics[:]
        # remove intrinsics in the group and create new intrinsics with new shape
        del group['intrinsics']
        
        shape = (split_N,64,4)
        chunks = (1,64,4)
        ds_new_intr = group.create_dataset(
            name='intrinsics',
            shape=shape,
            dtype='f4',
            compression='gzip',
            chunks=chunks
        )
        ds_new_intr = group['intrinsics']
        # ds_intrinsics = group['original_intrinsics']
        # ds_new_intr = group['intrinsics']
        
        idx_video_tuples = [(idx, video_name) for idx, video_name in enumerate(video_names)]
        with Pool(processes=1) as pool:
            results = list(tqdm.tqdm(pool.imap(process_and_save, idx_video_tuples), total=split_N))
        for idx, all_intrinsics in results:
            ds_new_intr[idx] = all_intrinsics