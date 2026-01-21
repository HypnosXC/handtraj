import json
import tqdm
import os 

record_split_seqs_path = "/public/datasets/handdata/interhand26m/anno/record_interhand_valid_seqs.json"
resplit_seq_path = "/public/datasets/handdata/interhand26m/anno/resplit_ih26_v4.json"
with open(record_split_seqs_path, 'r') as f:
    split_seqs = json.load(f)
capture_cnt = {
    "train":10,
    "val":1,
    "test":2
}

resplit_dict = {}
for split in ["train", "val", "test"]:
    print(f"Processing {split} set...")
    resplit_dict[split] = []
    for cap_idx in tqdm.tqdm(range(capture_cnt[split])):
        for seq in split_seqs[split][str(cap_idx)].keys():
            for cam in split_seqs[split][str(cap_idx)][seq].keys():
                for handtype in ['left', 'right']:
                    for clip in split_seqs[split][str(cap_idx)][seq][cam][handtype]:
                        # if clip[1]<32:
                        #     continue
                        # if clip[1]<=64:
                        resplit_dict[split].append((split, cap_idx, seq, cam, clip[0], clip[1], handtype))
                        # else:
                        #     sub_clip_num = clip[1] // 64
                        #     for sub_idx in range(sub_clip_num):
                        #         saved_start_idx = sub_idx * 64 + clip[0]
                        #         resplit_dict[split].append((split, cap_idx, seq, cam, saved_start_idx, 64, handtype))
                        #     if clip[1] % 64 != 0:
                        #         saved_start_idx = clip[1] - 64 +clip[0]
                        #         resplit_dict[split].append((split, cap_idx, seq, cam, saved_start_idx, 64, handtype))

N = len(resplit_dict['train']) + len(resplit_dict['val']) + len(resplit_dict['test'])
print("original count:\n train:", len(resplit_dict['train']), "\n val:", len(resplit_dict['val']), "\n test:", len(resplit_dict['test']))
N_10 = N//10
resplit_dict['train'].extend(resplit_dict['val'][N_10:])
resplit_dict['val'] = resplit_dict['val'][:N_10]
resplit_dict['train'].extend(resplit_dict['test'][N_10:])
resplit_dict['test'] = resplit_dict['test'][:N_10]
print("after 10% val/test to val/test:\n train:", len(resplit_dict['train']), "\n val:", len(resplit_dict['val']), "\n test:", len(resplit_dict['test']))

with open(resplit_seq_path, 'w') as f:
    json.dump(resplit_dict, f)

anno_dir="/public/datasets/handdata/interhand26m/anno/annotation"

mp4_format = "original_{}_Capture{}_{}_cam{}_{}_{}_{}.mp4" # original_split, cap_idx, seq, cam, start_idx, length_this, hand_type

# mp4_format.format(original_split,cap_idx, seq, cam, start_idx, save_length, hand_type)

cam_mano_split_dict = {}
cam_joint_split_dict = {}
for split in ['train', 'val', 'test']:
    cam_mano_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera_mano.json')
    cam_joint_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera_joint.json')
    with open(cam_mano_dir, 'r') as f:
        cam_mano_split_dict[split] = json.load(f)
    with open(cam_joint_dir, 'r') as f:
        cam_joint_split_dict[split] = json.load(f)

todo_video_list = []
for split in ['train', 'val', 'test']:
    save_mp4_dir = os.path.join("/public/datasets/handdata/interhand26m/data/picked_videos",split)
    os.makedirs(save_mp4_dir, exist_ok=True)
    # breakpoint()
    resplit_cam_mano_path = os.path.join(anno_dir, split, f'seq_{split}_camera_mano_resplit.json')       
    resplit_cam_joint_path = os.path.join(anno_dir, split, f'seq_{split}_camera_joint_resplit.json')
    resplit_cam_mano_dict = []
    resplit_cam_joint_dict = []
    # camera_dir = os.path.join("/public/datasets/handdata/interhand26m/anno/annotation", split, f'InterHand2.6M_{split}_camera.json')

    # with open(cam_mano_dir, 'r') as f:
    #     cam_mano_data = json.load(f)
#         cam_mano_data['0']['410234']['58997']['right'].keys()
# dict_keys(['shape', 'pose'])
    # with open(cam_joint_dir, 'r') as f:
    #     cam_joint_data = json.load(f)
#         len(cam_joint_data['0']['410234']['58997']['right'])
# 63
    print("Processing {} set...".format(split))
    for idx in tqdm.tqdm(range(len(resplit_dict[split])), desc=f"Processing {split} split"):
        item = resplit_dict[split][idx]
        original_split, cap_idx, seq, cam, start_idx, save_length, hand_type = item
        expected_mp4_name = mp4_format.format(original_split,cap_idx, seq, cam, start_idx, save_length, hand_type)
        save_mp4_path = os.path.join(save_mp4_dir, expected_mp4_name)
        if not os.path.exists(save_mp4_path):
            todo_video_list.append((split, original_split, cap_idx, seq, cam, start_idx, save_length, hand_type))
        
        cam_dir = os.path.join("/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images", original_split, f"Capture{cap_idx}", seq, f"cam{cam}")
        frame_list = os.listdir(cam_dir)
        frame_list = [f.split('.')[0].split('image')[-1] for f in frame_list]
        frame_list = sorted(frame_list, key=lambda x:int(x))[start_idx:start_idx+save_length]
        this_mano = {"info": item, "shape":[], "pose":[]}
        this_joint = {"info": item, "joint":[]}
        for frame_id in frame_list:
            frame_id_str = str(int(frame_id))
            this_mano["shape"].append(cam_mano_split_dict[original_split][str(cap_idx)][cam][frame_id_str][hand_type]['shape'])
            this_mano["pose"].append(cam_mano_split_dict[original_split][str(cap_idx)][cam][frame_id_str][hand_type]['pose'])
            this_joint["joint"].append(cam_joint_split_dict[original_split][str(cap_idx)][cam][frame_id_str][hand_type])
        
        resplit_cam_mano_dict.append(this_mano)
        resplit_cam_joint_dict.append(this_joint)
    with open(resplit_cam_mano_path, 'w') as f:
        json.dump(resplit_cam_mano_dict, f)
    with open(resplit_cam_joint_path, 'w') as f:
        json.dump(resplit_cam_joint_dict, f)

print("Total videos to be processed:", len(todo_video_list))
import multiprocessing as mp
import imageio.v2 as iio

def extract_video(item):
    split, original_split, cap_idx, seq, cam, start_idx, save_length, hand_type = item
    expected_mp4_name = mp4_format.format(original_split,cap_idx, seq, cam, start_idx, save_length, hand_type)
    save_mp4_dir = os.path.join("/public/datasets/handdata/interhand26m/data/picked_videos",split)
    save_mp4_path = os.path.join(save_mp4_dir, expected_mp4_name)
    cam_dir = os.path.join("/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images", original_split, f"Capture{cap_idx}", seq, f"cam{cam}")
    frame_list = os.listdir(cam_dir)
    frame_list = [f.split('.')[0].split('image')[-1] for f in frame_list]
    frame_list = sorted(frame_list, key=lambda x:int(x))[start_idx:start_idx+save_length]
    writer = iio.get_writer(save_mp4_path, fps=30,macro_block_size=None)
    for frame_id in frame_list:
        frame_path = os.path.join(cam_dir, f"image{frame_id}.jpg")
        image = iio.imread(frame_path)
        writer.append_data(image)
    writer.close()

with mp.Pool(processes=64) as pool:
    list(tqdm.tqdm(pool.imap(extract_video, todo_video_list), total=len(todo_video_list)))

        


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

                        
