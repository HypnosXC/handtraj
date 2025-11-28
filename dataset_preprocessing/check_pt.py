# pt_path = "../../../xuchen/handtraj/experiments/hand_train_cond_wrist_motion_more_data/v1/error_batch_529793.pt"
# pt_path="/public/datasets/handdata/dexycb/img_feats/train/imgfeat_0.pt"
pt_path = "/public/datasets/handdata/interhand26m/data/img_feats_dino/train/imgfeat_train_44.pt"
import torch
data = torch.load(pt_path,weights_only=False)
breakpoint()
print(data)

# import pickle
# # pt_path = "/public/datasets/handdata/HO3D_v3/evaluation/AP10/meta/0899.pkl"
# pt_path = "/public/datasets/handdata/HO3D_v3/train/ABF10/meta/0153.pkl"
# with open(pt_path, 'rb') as f:
#     data = pickle.load(f)
# breakpoint()
# print(data.keys())