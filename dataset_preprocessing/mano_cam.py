import os
import json
import torch
from manopth.manolayer import ManoLayer
mano_model_folder = "/public/home/group_ucb/yunqili/code/hamer/_DATA/data/mano"
    
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

@torch.no_grad()
def cal_root_j(th_betas, layer: ManoLayer) -> torch.Tensor:
    th_v_shaped = torch.matmul(layer.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + layer.th_v_template
    th_j = torch.matmul(layer.th_J_regressor, th_v_shaped)
    return th_j[:, 0, :].contiguous().view(3, 1)

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

import numpy as np
import cv2
from tqdm import tqdm

# # range_hand = {
# #     "right":slice(0, 21),
# #     "left": slice(21, 42)
# # }
# # capture_cnt = {
# #     "train":10,
# #     "val":1,
# #     "test":2
# # }
# # record_bad_data = []
# # bad_data_file = "/public/datasets/handdata/interhand26m/anno/interhand_bad_data.txt"

list_processed_path = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64_resplit.json"
anno_dir = "/public/datasets/handdata/interhand26m/anno/annotation"
with open(list_processed_path, 'r') as f:
    processed_list = json.load(f)
for split in ["train","val","test"]:
    split_list = processed_list[split]
    unprocessed_mano_dir = os.path.join(anno_dir, split, f'InterHand2.6M_{split}_MANO_NeuralAnnot.json')
    with open(unprocessed_mano_dir, 'r') as f:
        unprocessed_mano = json.load(f)
    camera_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera.json')
    with open(camera_dir, 'r') as f:
        camera_data = json.load(f)
    original_cam_mano_dir = os.path.join(anno_dir, split, f'seq_{split}_camera_mano_resplit.json')
    with open(original_cam_mano_dir, 'r') as f:
        original_cam_mano = json.load(f)
    for idx in tqdm(range(len(split_list)), desc=f"Checking {split} split"):
        # new_cal_cam_mano = 
        item = processed_list[split][idx]
        frame_list = item['frames']
        original_split, cap_idx, seq, cam, start_idx, save_length, hand_type = item['info']
        unprocessed_this_mano = unprocessed_mano[str(cap_idx)][str(int(frame_list[start_idx]))][hand_type]
        original_this_cam_mano = original_cam_mano[idx]
        original_cam_mano_pose = original_this_cam_mano['pose'][0]
        unprocessed_mano_pose = torch.FloatTensor(unprocessed_this_mano['pose']).view(1,-1)
        camrot_list = camera_data[str(cap_idx)]['camrot'][cam]
        campos_list = camera_data[str(cap_idx)]['campos'][cam]
        camrot = np.array(camrot_list).reshape(3,3)
        campos = np.array(campos_list).reshape(3)/1000
        cam2world_rot = torch.FloatTensor(camrot)
        cam2world_trans = -cam2world_rot @ torch.FloatTensor(campos).reshape(3,1)
        
        # extrinsics_matrix = torch.zeros((4,4))
        # extrinsics_matrix[:3,:3] = cam2world_rot
        # extrinsics_matrix[:3,3] = cam2world_trans.reshape(3)


        breakpoint()
        mano_rot,_ = cv2.Rodrigues(unprocessed_mano_pose[0,:3].numpy())
        mano_rot = torch.FloatTensor(mano_rot)
        mano_trans = torch.FloatTensor(unprocessed_this_mano['trans']).reshape(3,1)
        cam2mano_rot = cam2world_rot @ mano_rot
        cam2mano_trans = cam2world_trans + cam2world_rot @ mano_trans
        mano_rot_angle,_ = cv2.Rodrigues(cam2mano_rot.numpy())
        mano_rot_angle = torch.FloatTensor(mano_rot_angle.reshape(3))
        mano_pose = torch.zeros((1,48))
        mano_pose[0,:3] = torch.FloatTensor(mano_rot_angle.reshape(3))
        if hand_type=='right':
            mano_pose[:,3:48] = add_mean_pose45(mano_pose[:,3:48], mano_layer_right)
            root_j = cal_root_j(torch.FloatTensor(unprocessed_this_mano['shape']).view(-1).unsqueeze(0), mano_layer_right)
        else:
            mano_pose[:,3:48] = add_mean_pose45(mano_pose[:,3:48], mano_layer_left)
            root_j = cal_root_j(torch.FloatTensor(unprocessed_this_mano['shape']).view(-1).unsqueeze(0), mano_layer_left)
        delta = -root_j + torch.tensor(cam2world_rot,dtype=root_j.dtype) @ root_j
        mano_pose = torch.cat([mano_pose,cam2mano_trans.reshape(1,3) + delta.reshape(1,3)], dim=1)
        breakpoint()


# #     cam_mano_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera_mano.json')
# #     original_joint_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_joint_3d.json')
# #     cam_joint_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera_joint.json')
# #     camera_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera.json')
# #     with open(original_mano_dir, 'r') as f:
# #         original_mano_data = json.load(f)
# #     with open(original_joint_dir, 'r') as f:
# #         original_joint_data = json.load(f)
# #     with open(camera_dir, 'r') as f:
# #         camera_data = json.load(f)
# #     new_mano_data = {}
# #     new_joint_data = {}
# #     for cap_id_int in range(capture_cnt[split]):
# #         cap_id = str(cap_id_int)
# #         print(f"Processing capture {cap_id}")
# #         new_mano_data[cap_id] = {}
# #         new_joint_data[cap_id] = {}
# #         for cam_id in tqdm(camera_data[cap_id]['camrot'].keys()):
# #             new_joint_data[cap_id][cam_id] = {}
# #             new_mano_data[cap_id][cam_id] = {}
# #             camrot_list = camera_data[cap_id]['camrot'][cam_id]
# #             campos_list = camera_data[cap_id]['campos'][cam_id]
# #             camrot = np.array(camrot_list).reshape(3,3)
# #             campos = np.array(campos_list).reshape(3)/1000
# #             cam2world_rot = torch.FloatTensor(camrot)
# #             cam2world_trans = -cam2world_rot @ torch.FloatTensor(campos).reshape(3,1)
            
# #             extrinsics_matrix = torch.zeros((4,4))
# #             extrinsics_matrix[:3,:3] = cam2world_rot
# #             extrinsics_matrix[:3,3] = cam2world_trans.reshape(3)
# #             for frame_id in original_joint_data[cap_id].keys():
# #                 try:
# #                     this_joint_data = original_joint_data[cap_id][frame_id]
# #                     this_mano_data = original_mano_data[cap_id][frame_id]
# #                 except:
# #                     record_bad_data.append((split, cap_id, frame_id, 'left'))
# #                     record_bad_data.append((split, cap_id, frame_id, 'right'))
# #                     new_joint_data[cap_id][cam_id][frame_id] = {'right':None,'left':None}
# #                     new_mano_data[cap_id][cam_id][frame_id] = {'right':None,'left':None}
# #                     continue
# #                 joint_world = np.array(this_joint_data['world_coord']).reshape(-1,3)
# #                 joint_valid = np.array(this_joint_data['joint_valid'])
# #                 new_joint_data[cap_id][cam_id][frame_id] = {}
# #                 new_mano_data[cap_id][cam_id][frame_id] = {}
# #                 for hand_type in ['right','left']:
# #                     joint_part = joint_world[range_hand[hand_type]]
# #                     if np.sum(joint_valid[range_hand[hand_type]])==0:
# #                         part_valid = False
# #                     else:
# #                         part_valid = True
# #                     if this_mano_data[hand_type] is None:
# #                         new_joint_data[cap_id][cam_id][frame_id][hand_type] = None
# #                         new_mano_data[cap_id][cam_id][frame_id][hand_type] = None
# #                         if part_valid:
# #                             record_bad_data.append((split, cap_id, frame_id, hand_type))
# #                         continue
# #                     new_mano_data[cap_id][cam_id][frame_id][hand_type] = {}
# #                     mano_betas = torch.FloatTensor(this_mano_data[hand_type]['shape']).view(-1)
# #                     mano_pose = torch.FloatTensor(this_mano_data[hand_type]['pose']).view(1,-1)
# #                     mano_rot,_ = cv2.Rodrigues(mano_pose[0,:3].numpy())
# #                     mano_rot = torch.FloatTensor(mano_rot)
# #                     mano_trans = torch.FloatTensor(this_mano_data[hand_type]['trans']).reshape(3,1)
# #                     cam2mano_rot = cam2world_rot @ mano_rot
# #                     cam2mano_trans = cam2world_trans + cam2world_rot @ mano_trans
# #                     mano_rot_angle,_ = cv2.Rodrigues(cam2mano_rot.numpy())
# #                     mano_rot_angle = torch.FloatTensor(mano_rot_angle.reshape(3))
# #                     mano_pose[0,:3] = torch.FloatTensor(mano_rot_angle.reshape(3))
# #                     if hand_type=='right':
# #                         mano_pose[:,3:48] = add_mean_pose45(mano_pose[:,3:48], mano_layer_right)
# #                         root_j = cal_root_j(mano_betas.unsqueeze(0), mano_layer_right)
# #                     else:
# #                         mano_pose[:,3:48] = add_mean_pose45(mano_pose[:,3:48], mano_layer_left)
# #                         root_j = cal_root_j(mano_betas.unsqueeze(0), mano_layer_left)
# #                     delta = -root_j + torch.tensor(cam2world_rot,dtype=root_j.dtype) @ root_j
# #                     mano_pose = torch.cat([mano_pose,cam2mano_trans.reshape(1,3) + delta.reshape(1,3)], dim=1)
# #                     new_mano_data[cap_id][cam_id][frame_id][hand_type] = {}
# #                     new_mano_data[cap_id][cam_id][frame_id][hand_type]['shape'] = mano_betas.numpy().reshape(-1).tolist()
# #                     new_mano_data[cap_id][cam_id][frame_id][hand_type]['pose'] = mano_pose.numpy().reshape(-1).tolist()
                    
# #                     # assert joint_world.shape[range_hand[hand_type]] not all zero
# #                     if not part_valid:
# #                         record_bad_data.append((split, cap_id, frame_id, hand_type))
# #                     joint_cam = cam2world_rot @ torch.FloatTensor(joint_part.T/1000) + cam2world_trans
# #                     joint_cam = joint_cam.T.numpy()
# #                     new_joint_data[cap_id][cam_id][frame_id][hand_type] = joint_cam.reshape(-1).tolist()

# #     with open(cam_mano_dir, 'w') as f:
# #         json.dump(new_mano_data, f)
# #     with open(cam_joint_dir, 'w') as f:
# #         json.dump(new_joint_data, f)

# # with open(bad_data_file, 'w') as f:
# #     for item in record_bad_data:
# #         f.write(f"{item[0]},{item[1]},{item[2]},{item[3]}\n")
# # print(f"Save bad data list to {bad_data_file}")

# record_split_seqs_32_list = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64.json"
# anno_dir ="/public/datasets/handdata/interhand26m/anno/annotation"
# # with open(record_split_seqs_32_list, 'r') as f:
# #     record = json.load(f)

# # for split in ["train","val","test"]:
# #     cam_mano_dir = os.path.join(anno_dir, split, f'InterHand2.6M_{split}_camera_mano.json')
# #     cam_joint_dir = os.path.join(anno_dir, split, f'InterHand2.6M_{split}_camera_joint.json')
# #     cam_mano_seq_path = os.path.join(anno_dir, split, f'seq_{split}_camera_mano.json')
# #     cam_joint_seq_path = os.path.join(anno_dir, split, f'seq_{split}_camera_joint.json')
# #     with open(cam_mano_dir, 'r') as f:
# #         cam_mano_data = json.load(f)
# #     with open(cam_joint_dir, 'r') as f:
# #         cam_joint_data = json.load(f)
# #     this_record = record[split]
# #     new_mano_data = []
# #     new_joint_data = []
# #     for item in tqdm(this_record):
# #         cap_idx, seq, cam, start_idx, length, hand_type = item['info']
# #         frames_list = item['frames']
# #         this_mano ={"info": item['info'],"shape": [],"pose": []}
# #         this_joint ={"info": item['info'],"joint": []}
# #         for frame_id in frames_list:
# #             frame_id_str = str(int(frame_id))
# #             this_mano["shape"].append(cam_mano_data[str(cap_idx)][cam][frame_id_str][hand_type]['shape'])
# #             this_mano["pose"].append(cam_mano_data[str(cap_idx)][cam][frame_id_str][hand_type]['pose'])
# #             this_joint["joint"].append(cam_joint_data[str(cap_idx)][cam][frame_id_str][hand_type])


# #         new_mano_data.append(this_mano)
# #         new_joint_data.append(this_joint)
# #     with open(cam_mano_seq_path, 'w') as f:
# #         json.dump(new_mano_data, f)
# #     with open(cam_joint_seq_path, 'w') as f:
# #         json.dump(new_joint_data, f)
# train_cam_mano_seq_path = os.path.join(anno_dir, 'train', 'seq_train_camera_mano.json')
# val_cam_mano_seq_path = os.path.join(anno_dir, 'val', 'seq_val_camera_mano.json')
# test_cam_mano_seq_path = os.path.join(anno_dir, 'test', 'seq_test_camera_mano.json')
# new_train_cam_mano_seq_path = os.path.join(anno_dir, 'train', 'seq_train_camera_mano_resplit.json')
# new_val_cam_mano_seq_path = os.path.join(anno_dir, 'val', 'seq_val_camera_mano_resplit.json')
# new_test_cam_mano_seq_path = os.path.join(anno_dir, 'test', 'seq_test_camera_mano_resplit.json')
# train_cam_joint_seq_path = os.path.join(anno_dir, 'train', 'seq_train_camera_joint.json')
# val_cam_joint_seq_path = os.path.join(anno_dir, 'val', 'seq_val_camera_joint.json')
# test_cam_joint_seq_path = os.path.join(anno_dir, 'test', 'seq_test_camera_joint.json')
# new_train_cam_joint_seq_path = os.path.join(anno_dir, 'train', 'seq_train_camera_joint_resplit.json')
# new_val_cam_joint_seq_path = os.path.join(anno_dir, 'val', 'seq_val_camera_joint_resplit.json')
# new_test_cam_joint_seq_path = os.path.join(anno_dir, 'test', 'seq_test_camera_joint_resplit.json')
# print("Resplitting train/val/test sets to 8:1:1 ratio")
# # with open(train_cam_mano_seq_path, 'r') as f:
# #     train_mano_data = json.load(f)
# # print("train mano data loaded")
# # with open(val_cam_mano_seq_path, 'r') as f:
# #     val_mano_data = json.load(f)
# # print("val mano data loaded")
# # with open(test_cam_mano_seq_path, 'r') as f:
# #     test_mano_data = json.load(f)
# # print("test mano data loaded")
# # print(f"Before split: train {len(train_mano_data)}, val {len(val_mano_data)}, test {len(test_mano_data)}")

# # move val_mano_data[all_cnt:] to train_mano_data
# # move test_mano_data[all_cnt_10:] to val_mano_data
# # train_mano_data = train_mano_data + val_mano_data[all_cnt_10:] + test_mano_data[all_cnt_10:]
# # val_mano_data = val_mano_data[:all_cnt_10]
# # test_mano_data = test_mano_data[:all_cnt_10]
# # print(f"After split: train {len(train_mano_data)}, val {len(val_mano_data)}, test {len(test_mano_data)}")
# # with open(new_train_cam_mano_seq_path, 'w') as f:
# #     json.dump(train_mano_data, f)
# # print(f"Save new train mano data to {new_train_cam_mano_seq_path}")
# # with open(new_val_cam_mano_seq_path, 'w') as f:
# #     json.dump(val_mano_data, f)
# # print(f"Save new val mano data to {new_val_cam_mano_seq_path}")
# # with open(new_test_cam_mano_seq_path, 'w') as f:
# #     json.dump(test_mano_data, f)
# # print(f"Save new test mano data to {new_test_cam_mano_seq_path}")

# print("Processing joint data")
# with open(train_cam_joint_seq_path, 'r') as f:
#     train_joint_data = json.load(f)
# print("train joint data loaded")
# with open(val_cam_joint_seq_path, 'r') as f:
#     val_joint_data = json.load(f)
# print("val joint data loaded")
# with open(test_cam_joint_seq_path, 'r') as f:
#     test_joint_data = json.load(f)
# print("test joint data loaded")
# # assert len(train_joint_data)==len(train_mano_data)
# # assert len(val_joint_data)==len(val_mano_data)
# # assert len(test_joint_data)==len(test_mano_data)
# all_cnt = len(train_joint_data) + len(val_joint_data) + len(test_joint_data)
# all_cnt_10 = all_cnt // 10
# print(f"Before split: train {len(train_joint_data)}, val {len(val_joint_data)}, test {len(test_joint_data)}")
# train_joint_data = train_joint_data + val_joint_data[all_cnt_10:] + test_joint_data[all_cnt_10:]
# val_joint_data = val_joint_data[:all_cnt_10]
# test_joint_data = test_joint_data[:all_cnt_10]
# print(f"After split: train {len(train_joint_data)}, val {len(val_joint_data)}, test {len(test_joint_data)}")
# with open(new_train_cam_joint_seq_path, 'w') as f:
#     json.dump(train_joint_data, f)
# print(f"Save new train joint data to {new_train_cam_joint_seq_path}")
# with open(new_val_cam_joint_seq_path, 'w') as f:
#     json.dump(val_joint_data, f)
# print(f"Save new val joint data to {new_val_cam_joint_seq_path}")
# with open(new_test_cam_joint_seq_path, 'w') as f:
#     json.dump(test_joint_data, f)
# print(f"Save new test joint data to {new_test_cam_joint_seq_path}")


# with open(record_split_seqs_32_list, 'r') as f:
#     record = json.load(f)
# resplit_record = {"train":[],
# "val":[],
# "test":[]}
# # assert len(record['train'])==len(train_mano_data)
# # assert len(record['val'])==len(val_mano_data)
# # assert len(record['test'])==len(test_mano_data)
# resplit_record['train'] = record['train'] + record['val'][all_cnt_10:] + record['test'][all_cnt_10:]
# resplit_record['val'] = record['val'][:all_cnt_10]
# resplit_record['test'] = record['test'][:all_cnt_10]
# new_record_split_seqs_32_list = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs_32_64_resplit.json"
# with open(new_record_split_seqs_32_list, 'w') as f:
#     json.dump(resplit_record, f)
# print(f"Save new record to {new_record_split_seqs_32_list}")