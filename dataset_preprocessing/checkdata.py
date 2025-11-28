import os
import json
import tqdm
list_processed_path = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64_resplit.json"
anno_dir = "/public/datasets/handdata/interhand26m/anno/annotation"
mp4_dir = "/public/datasets/handdata/interhand26m/data/picked_videos"
with open(list_processed_path, 'r') as f:
    processed_list = json.load(f)
for split in ['train', 'val', 'test']:
    this_mp4_dir = os.path.join(mp4_dir, split)
    cam_mano_processed_path = os.path.join(anno_dir, split, f'seq_{split}_camera_mano_resplit.json')
    cam_joint_processed_path = os.path.join(anno_dir, split, f'seq_{split}_camera_joint_resplit.json')
    with open(cam_mano_processed_path, 'r') as f:
        cam_mano_dict = json.load(f)
    with open(cam_joint_processed_path, 'r') as f:
        cam_joint_dict = json.load(f)
    if not len(cam_mano_dict) == len(cam_joint_dict):
        breakpoint()
    if not len(processed_list[split]) == len(cam_mano_dict):
        breakpoint()
    mp4_list = os.listdir(this_mp4_dir)
    if not len(mp4_list) == len(processed_list[split]):
        breakpoint()
    for idx in tqdm.tqdm(range(len(processed_list[split])), desc=f"Checking {split} split"):
        item = processed_list[split][idx]
        original_split, cap_idx, seq, cam, start_idx, save_length, hand_type = item['info']
        expected_mp4_name = f"original_{original_split}_Capture{cap_idx}_{seq}_cam{cam}_{start_idx}_{save_length}_{hand_type}.mp4"
        if not expected_mp4_name in mp4_list:
            breakpoint()
        this_mano_dict = cam_mano_dict[idx]
        this_joint_dict = cam_joint_dict[idx]
        if not this_mano_dict['info'] == this_joint_dict['info']:
            breakpoint()
        if not this_mano_dict['info'] == [cap_idx, seq, cam, start_idx, save_length, hand_type]:
            breakpoint()

        