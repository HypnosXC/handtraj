import os
import json
import tqdm
bad_data_file = "/public/datasets/handdata/interhand26m/anno/interhand_bad_data.txt"
interhand_img_dir= "/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images"
# with open(bad_data_file,'r') as f:
#     bad_data_lines = f.readlines()
#     # split, cap_id, frame_id, hand_type
#     bad_data_set = set()
#     for line in bad_data_lines:
#         line = line.strip()
#         items = line.split(',')
#         bad_data_set.add((items[0], items[1], items[2], items[3]))
# print(f"Total {len(bad_data_set)} bad data entries loaded.")
# capture_cnt = {
#     "train":10,
#     "val":1,
#     "test":2
# }

# record_split_seqs_path = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs.json"
# split_seqs = {}
# right_num = 0
# left_num = 0
# for split in ["train","val","test"]:
#     split_seqs[split] = {}
#     cam_mano_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera_mano.json')
#     # cam_joint_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera_joint.json')
#     img_root = os.path.join(interhand_img_dir, split)
#     with open(cam_mano_dir, 'r') as f:
#         cam_mano_data = json.load(f)
#     for cap_idx in tqdm.tqdm(range(capture_cnt[split]), desc=f"Processing {split} split"):
#         split_seqs[split][str(cap_idx)] = {}
#         cap_str = f"Capture{cap_idx}"
#         for seq in os.listdir(os.path.join(img_root, cap_str)):
#             split_seqs[split][str(cap_idx)][seq] = {}
#             seq_dir = os.path.join(img_root, cap_str, seq)
#             for cam_str in os.listdir(seq_dir):
#                 cam = cam_str.split('cam')[-1]
#                 split_seqs[split][str(cap_idx)][seq][cam] = {"left":[], "right":[]}
#                 cam_dir = os.path.join(seq_dir, cam_str)
#                 frame_list = os.listdir(cam_dir)
#                 frame_list = [int(f.split('.')[0].split('image')[-1]) for f in frame_list]
#                 frame_list = sorted(frame_list)
#                 valid_left =(-1,0) # start_idx, length
#                 valid_right = (-1,0) # start_idx, length
#                 for idx,frame_id_int in enumerate(frame_list):
#                     frame_id = str(frame_id_int)
#                     left_mano = cam_mano_data[str(cap_idx)][cam].get(frame_id, {}).get('left', None)
#                     right_mano = cam_mano_data[str(cap_idx)][cam].get(frame_id, {}).get('right', None)
#                     if (split, str(cap_idx), frame_id, 'left') in bad_data_set or left_mano is None:
#                         # end left valid seq
#                         if valid_left[0] != -1:
#                             # record the valid seq
#                             start_idx, length = valid_left
#                             split_seqs[split][str(cap_idx)][seq][cam]['left'].append((start_idx, length))
#                             valid_left = (-1,0)
#                             if length >=32:
#                                 left_num += 1
#                     else:
#                         # continue left valid seq
#                         if valid_left[0] == -1:
#                             valid_left = (idx, 1)
#                         else:
#                             valid_left = (valid_left[0], valid_left[1]+1)   

#                     if (split, str(cap_idx), frame_id, 'right') in bad_data_set or right_mano is None:
#                         # end right valid seq
#                         if valid_right[0] != -1:
#                             # record the valid seq
#                             start_idx, length = valid_right
#                             split_seqs[split][str(cap_idx)][seq][cam]['right'].append((start_idx, length))
#                             valid_right = (-1,0)
#                             if length >=32:
#                                 right_num += 1
#                     else:
#                         # continue right valid seq
#                         if valid_right[0] == -1:
#                             valid_right = (idx, 1)
#                         else:
#                             valid_right = (valid_right[0], valid_right[1]+1)
#                 # end of the seq, check if there is valid seq not recorded
#                 if valid_left[0] != -1:
#                     start_idx, length = valid_left
#                     split_seqs[split][str(cap_idx)][seq][cam]['left'].append((start_idx, length))
#                 if valid_right[0] != -1:
#                     start_idx, length = valid_right
#                     split_seqs[split][str(cap_idx)][seq][cam]['right'].append((start_idx, length))

# print(f"Total {right_num} right hand valid seqs, {left_num} left hand valid seqs.")
# with open(record_split_seqs_path, 'w') as f:
#     json.dump(split_seqs, f, indent=4)
    
# print("Splitting valid seqs into 32-64 length...")
# num_hand_type={
#     "left":0,
#     "right":0
# }
# record_split_seqs_path = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs.json"
# record_split_seqs_32_list = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64.json"
# with open(record_split_seqs_path, 'r') as f:
#     split_seqs = json.load(f)
# split_seqs_32 = {}
save_mp4_dir = "/public/datasets/handdata/interhand26m/data/picked_videos"
for split in ["train","val","test"]:
    this_mp4_dir = os.path.join(save_mp4_dir, split)
    os.makedirs(this_mp4_dir, exist_ok=True)

import imageio.v2 as iio
def process_cam(worker_args):
    original_split, cap_idx, seq, cam, start_idx, save_length, hand_type, this_mp4_dir = worker_args
    # seq_32s = []
    this_dir = os.path.join(interhand_img_dir, original_split, f"Capture{cap_idx}", seq, f"cam{cam}")
    img_list = sorted(os.listdir(this_dir), key=lambda x: int(x.split('.')[0].split('image')[-1]))
    img_list = [os.path.join(this_dir, img_name) for img_name in img_list]
    writer = iio.get_writer(os.path.join(this_mp4_dir, f"original_{original_split}_Capture{cap_idx}_{seq}_cam{cam}_{start_idx}_{save_length}_{hand_type}.mp4"), fps=30,macro_block_size=None)
    for img_path in img_list[start_idx:start_idx+save_length]:
        img = iio.imread(img_path)
        writer.append_data(img)
    writer.close()
    # for hand_type in ["left","right"]:
    #     for (start_idx, length) in split_seqs[split][cap_idx][seq][cam][hand_type]:
    #         length_now = length
    #         while length_now >= 32:
    #             if length_now <=64:
    #                 # save to mp4
    #                 save_length = length_now
    #                 # this_dict = {"info":(cap_idx, seq, cam, start_idx, save_length, hand_type),"frames":[]}
    #                 # for img_path in img_list[start_idx:start_idx+save_length]:
    #                 #     this_dict["frames"].append(img_path.split('/')[-1].split('.')[0].split('image')[-1])
    #                 # seq_32s.append(this_dict)
    #                 # num_hand_type[hand_type] += 1
    #                 length_now = 0
    #             else:
    #                 save_length = 64
    #                 # this_dict = {"info":(cap_idx, seq, cam, start_idx, save_length, hand_type),"frames":[]}
    #                 # for img_path in img_list[start_idx:start_idx+save_length]:
    #                 #     this_dict["frames"].append(img_path.split('/')[-1].split('.')[0].split('image')[-1])
    #                 # seq_32s.append(this_dict)
    #                 # num_hand_type[hand_type] += 1
    #                 save_start = start_idx
    #                 start_idx += 56
    #                 length_now -= 56
    #             save_img_list = img_list[save_start:save_start+save_length]
    #             save_path = os.path.join(this_mp4_dir, f"Capture{cap_idx}_{seq}_cam{cam}_{save_start}_{save_length}_{hand_type}.mp4")
    #             writer = iio.get_writer(save_path, fps=30,macro_block_size=None)
    #             for img_path in save_img_list:
    #                 img = iio.imread(img_path)
    #                 writer.append_data(img)
    #             writer.close()
    return 

import shutil
# task_list = []
# for split in ["train","val","test"]:
#     this_mp4_dir = os.path.join(save_mp4_dir, split)
#     os.makedirs(this_mp4_dir, exist_ok=True)
#     split_seqs_32[split] = []
#     for cap_idx in split_seqs[split]:
#         # split_seqs_32[split][cap_idx] = {}
#         for seq in tqdm.tqdm(split_seqs[split][cap_idx], desc=f"Processing {split} Capture{cap_idx}"):
#             # split_seqs_32[split][cap_idx][seq] = {}
#             for cam in split_seqs[split][cap_idx][seq]:
#                 # split_seqs_32[split][cap_idx][seq][cam] = {"left":[],"right":[]}
#                 task_list.append((split, cap_idx, seq, cam, this_mp4_dir))
# task_list = []
new_record_split_seqs_32_list = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64_resplit.json"
# with open(new_record_split_seqs_32_list,'r') as f:
#     split_seqs_32 = json.load(f)
new_record_split_dict =  {}
original_record_split_seqs_32_list = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64.json"
with open(original_record_split_seqs_32_list,'r') as f:
    split_seqs_32 = json.load(f)
N_all = len(split_seqs_32['train']) + len(split_seqs_32['val']) + len(split_seqs_32['test'])
N_10 = N_all // 10
N_val_train = len(split_seqs_32['val']) - N_10
N_test_train = len(split_seqs_32['test']) - N_10
# move N_val_train from val to train
for split in ["train","val","test"]:
    original_split = split
    new_split = split
    new_record_split_dict[split] = []
    for idx,item in enumerate(tqdm.tqdm(split_seqs_32[split], desc=f"Resplitting {split} split")):
        if split == "val" and idx >= N_10:
            new_split = "train"
        if split == "test" and idx >= N_10:
            new_split = "train"
        new_item = item.copy()
        cap_idx, seq, cam, start_idx, save_length, hand_type = item['info']
        new_item['info'] = (original_split, cap_idx, seq, cam, start_idx, save_length, hand_type)
        new_record_split_dict[new_split].append(new_item)
        mp4_path = os.path.join(save_mp4_dir, new_split, f"original_{original_split}_Capture{cap_idx}_{seq}_cam{cam}_{start_idx}_{save_length}_{hand_type}.mp4")
        if not os.path.exists(mp4_path):
            wrong_path = os.path.join(save_mp4_dir, 'train', f"original_{original_split}_Capture{cap_idx}_{seq}_cam{cam}_{start_idx}_{save_length}_{hand_type}.mp4")
            if not os.path.exists(wrong_path):
                breakpoint()
            else:
                shutil.move(wrong_path, mp4_path)
                # move to correct directory
                
        # task_list.append((original_split, cap_idx, seq, cam, start_idx, save_length, hand_type, os.path.join(save_mp4_dir, new_split)))

with open(new_record_split_seqs_32_list, 'w') as f:
    json.dump(new_record_split_dict, f, indent=4)
# task_num = len(task_list)
# from multiprocessing import Pool
# with Pool(processes=64) as pool:
#     for ret in tqdm.tqdm(pool.imap_unordered(process_cam, task_list), total=task_num):
#         pass
        # split, seq_32s = ret
        # split_seqs_32[split].extend(seq_32s)

                
# print(f"Total {num_hand_type['right']} right hand valid seqs, {num_hand_type['left']} left hand valid seqs.")
# with open(record_split_seqs_32_list, 'w') as f:
#     json.dump(split_seqs_32, f, indent=4)
