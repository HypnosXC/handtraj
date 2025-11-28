
from common.body_models import construct_layers
from src.arctic.preprocess_dataset import construct_loader
from glob import glob
from tqdm import tqdm
import os
# mano_ps = glob(f"/public/datasets/handdata/arctic/unpack/arctic_data/data/raw_seqs/*/*.mano.npy")
raw_seq_path = "/public/datasets/handdata/arctic/unpack/arctic_data/data/raw_seqs"
cropped_img_path = "/public/datasets/handdata/arctic/unpack/arctic_data/data/cropped_images"
# pbar = tqdm(mano_ps)
misc_p = "/public/datasets/handdata/arctic/unpack/arctic_data/data/meta/misc.json"
import json
with open(misc_p, "r") as f:
    misc = json.load(f)
    # breakpoint()
# misc.keys()
# dict_keys(['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10'])
# cam 0 : ego 
# cam 1-8: intr 0-7

# misc['s01'].keys()
# dict_keys(['gender', 'image_size', 'intris_mat', 'ioi_offset', 'world2cam'])
# misc['s01']['intris_mat']
# 8*3*3 list

# misc['s01']['intris_mat'][0]
# [[7270.5244140625, 0.0, 891.011962890625], [0.0, 7270.5244140625, 1358.8863525390625], [0.0, 0.0, 1.0]]

# len(misc['s01']['world2cam'])
# 8
# (Pdb) len(misc['s01']['world2cam'][0])
# 4
# (Pdb) len(misc['s01']['world2cam'][0][0])
# 4

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
}
shapes = {
    'mano_poses': (64, 1, 51),  # 64 frames, 1 hand, 51 joints
    'intrinsics': (4,),  # fx,fy,ppx,ppy
    'mano_side': (1,),  # single side (left or right)
    'mano_betas': (10,),  # 10 betas for MANO model 
    'extrinsics': (3, 4),  # 3x4 matrix for extrinsics
    'mano_joint_3d':(64, 1, 21, 3),  # 64 frames, 21 joints, 3D coordinates
    'video_name': (1,),  # single video name
}
hdf5_path="/public/datasets/handdata/arctic.hdf5"

# mappings={
#     'train': [],
#     'val': [],
#     'test': [],
# }

import imageio.v2 as iio
import cv2

# with h5py.File(hdf5_path, 'w') as f:
save_mp4_root = "/public/datasets/handdata/arctic/picked_videos"

resplit_p = "/public/datasets/handdata/arctic/record_arctic_valid_seqs_32_64_resplit_xxx.json"
img_root = "/public/datasets/handdata/arctic/unpack/arctic_data/data/cropped_images"


def write_video(args):
    good_list = []
    split,ioi_offset,s_idx, mano_name, start_frame, saved_length, split_mp4_dir, this_img_dir = args
    for cam_id in range(1,9):
        next_cam = False
        cam_img_dir = os.path.join(this_img_dir, str(cam_id))
        out_path = os.path.join(split_mp4_dir, f"{s_idx}_{mano_name}_{start_frame-ioi_offset}_{saved_length}_{cam_id}.mp4")
        if os.path.exists(out_path):
            good_list.append((split,s_idx, mano_name, start_frame-ioi_offset, saved_length, cam_id))
            continue

        frame_shape = None
        for frame_id in range(start_frame, start_frame+saved_length):
            img_path =os.path.join(cam_img_dir, f"{frame_id:05d}.jpg")
            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                # cancel video writing, including the written part
                if frame_shape is not None:
                    writer.close()
                    os.remove(os.path.join(split_mp4_dir, f"{s_idx}_{mano_name}_{start_frame-ioi_offset}_{saved_length}_{cam_id}.mp4"))
                next_cam = True
                break

            img = cv2.imread(img_path)
            # BGR to RGB
            if frame_shape is None:
                if img is None:
                    print(f"Image read error: {img_path}")
                    next_cam = True
                    break
                frame_shape = img.shape
                writer = iio.get_writer(out_path, fps=30, macro_block_size=None)
            else:
                if img is None or not img.shape == frame_shape:
                    # print(f"Image shape mismatch: {img_path}, expected {frame_shape}, got {img.shape}")
                    # cancel video writing, including the written part
                    writer.close()
                    os.remove(out_path)
                    next_cam = True
                    break
            img = img[:,:,::-1]
            writer.append_data(img)
        if next_cam:
            continue
        writer.close()
        good_list.append((split,s_idx, mano_name, start_frame-ioi_offset, saved_length, cam_id))
    return good_list


task_list = []
resplit_dict = {}
for split in ['train', 'val', 'test']:
    resplit_dict[split] = []
    split_mp4_dir = os.path.join(save_mp4_root, split)
    os.makedirs(split_mp4_dir, exist_ok=True)
    for s_idx in data_split[split]:
        ioi_offset = misc[s_idx]['ioi_offset']
        print(f"Processing subject {s_idx}")
        s_path = os.path.join(raw_seq_path, s_idx)
        manos = os.listdir(s_path)
        manos = [m for m in manos if m.endswith(".mano.npy")]
        mano_names = [m.replace(".mano.npy", "") for m in manos]
        for mano_name in tqdm(mano_names):
            mano_p = os.path.join(s_path, f"{mano_name}.mano.npy")
            # egocam_p = os.path.join(s_path, f"{mano_name}.egocam.dist.npy")
            # breakpoint()
            this_img_dir = os.path.join(img_root, s_idx, mano_name)
            # print(f"Processing {mano_p}")
            loader = construct_loader(mano_p)
            for idx,batch in enumerate(loader):
                assert idx == 0
                now_length = batch['pose_r'].shape[0]
                # for cam_id in range(9):
                #     frame_list = os.listdir(os.path.join(this_img_dir, str(cam_id)))
                #     if not len(frame_list) == batch['pose_r'].shape[0]:
                #         bad_file = set(frame_list)-
                start_idx=2
                now_length -= 2
                now_length-=ioi_offset
                if now_length <=0:
                    continue
                start_frame = start_idx + ioi_offset
                task_list.append((split,ioi_offset,s_idx, mano_name, start_frame, now_length, split_mp4_dir, this_img_dir))
                # for cam_id in range(9):
                #     this_dict = {"info": (s_idx, mano_name, start_idx,now_length, cam_id)}
                # resplit_dict[split].append(this_dict)
                # this_dict["frames"] = ...
                    

                # if now_length <32:
                #     continue
                # if now_length >=32 and now_length <64:
                #     start_frame = start_idx + ioi_offset
                #     task_list.append((split,ioi_offset,s_idx, mano_name, start_frame, now_length, split_mp4_dir, this_img_dir))
                #     # for cam_id in range(9):
                #     #     this_dict = {"info": (s_idx, mano_name, start_idx,now_length, cam_id)}
                #     # resplit_dict[split].append(this_dict)
                #     # this_dict["frames"] = ...
                #     continue

                # sub_clip_num = now_length // 64
                # for sub_idx in range(sub_clip_num):
                #     saved_start_idx = start_idx + sub_idx * 64
                #     start_frame = saved_start_idx + ioi_offset
                #     task_list.append((split,ioi_offset,s_idx, mano_name, start_frame, 64, split_mp4_dir, this_img_dir))
                # if now_length % 64 != 0:
                #     saved_start_idx = now_length - 64 + start_idx
                #     start_frame = saved_start_idx + ioi_offset
                #     task_list.append((split,ioi_offset,s_idx, mano_name, start_frame, 64, split_mp4_dir, this_img_dir))

                # while now_length != 0:
                #     if now_length >= 64:
                #         saved_start_idx = start_idx
                #         start_idx += 64
                #         now_length -= 64
                #     else:
                #         # the end 64 frames
                #         saved_start_idx = start_idx + now_length - 64
                #         now_length = 0

                #     start_frame = saved_start_idx + ioi_offset
                #     task_list.append((split,ioi_offset,s_idx, mano_name, start_frame, 64, split_mp4_dir, this_img_dir))
                    # for cam_id in range(9):
                    #     this_dict = {"info": (s_idx, mano_name, start_idx,saved_length, cam_id)}
                    # resplit_dict[split].append(this_dict)
                    
                    # this_dict["frames"] = ...

                # img_dir = os.path.join(cropped_img_path, s_idx, mano_name.replace(".mano.npy",""), str(idx))
                # img_list = os.listdir(img_dir)
                # print(f"Processing batch {idx}")
                # print(f"Found {len(img_list)} images")
                # breakpoint()

                #             batch.keys()
                # dict_keys(['rot_r', 'pose_r', 'trans_r', 'shape_r', 'fitting_err_r', 'rot_l', 'pose_l', 'trans_l', 'shape_l', 'fitting_err_l', 'smplx_transl', 'smplx_global_orient', 'smplx_body_pose', 'smplx_jaw_pose', 'smplx_leye_pose', 'smplx_reye_pose', 'smplx_left_hand_pose', 'smplx_right_hand_pose', 'obj_arti', 'obj_rot', 'obj_trans', 'world2ego', 'dist', 'K_ego', 'query_names'])


                # intr of ego
                # batch['K_ego'].shape
                # torch.Size([697, 3, 3])
                # batch['world2ego']
                # torch.Size([697, 4, 4])

                # batch['dist'] distortion parameters for egocentric camera
                # torch.Size([697, 8])

                # not flat hand
                # print(batch['rot_l'].shape)
                # breakpoint()

import multiprocessing
good_list = []
with multiprocessing.Pool(processes=16) as pool:
    # get return values to ensure all processes complete
    for ret in tqdm(pool.imap_unordered(write_video, task_list), total=len(task_list)):
        good_list.extend(ret)
for item in good_list:
    # remove bad item from resplit_dict
    split,s_idx, mano_name, start_idx, saved_length, cam_id = item
    resplit_dict[split].append({"info": (s_idx, mano_name, start_idx, saved_length, cam_id)})
    # s_idx, mano_name, start_frame, saved_length, cam_id

print(f"Total good clips: {len(resplit_dict['train'])+len(resplit_dict['val'])+len(resplit_dict['test'])}")
with open(resplit_p, "w") as f:
    json.dump(resplit_dict, f)