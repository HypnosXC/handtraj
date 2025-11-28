img_root = "/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images"
import os
# new json file
import json
json_path = "/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/sort_all.json"
json_dict = {}
from tqdm import tqdm
import glob

cap_dir = os.path.join(img_root, "train", "Capture2")
# search in cap/*/cam400002/
cam_dir = glob.glob(os.path.join(cap_dir, "*", "cam400002"))
# print(f"Found camera directories: {cam_dir}")
for cam in cam_dir:
    img_paths = os.listdir(cam)
    img_paths = [img.split('image')[1].split('.jpg')[0] for img in img_paths]
    # find 62638
    if '62638' in img_paths:
        print(f"Found 62638 in {cam}")
breakpoint()
#'62638'
for split in ["train", "val", "test"]:
    print(f"Processing {split} set")
    img_dir = os.path.join(img_root, split)
    json_dict[split] = {}
    for capture in tqdm(os.listdir(img_dir)):
        capture_dir = os.path.join(img_dir, capture)
        json_dict[split][capture] = {}
        # depth 2 images in one list
        for seq in os.listdir(capture_dir):
            seq_dir = os.path.join(capture_dir, seq)
            for cam in os.listdir(seq_dir):
                if cam not in json_dict[split][capture]:
                    json_dict[split][capture][cam] = []
                cam_dir = os.path.join(seq_dir, cam)
                img_list = os.listdir(cam_dir)
                img_list = [os.path.join(capture, seq, cam, img) for img in img_list]
                json_dict[split][capture][cam] += img_list
        for cam_key in json_dict[split][capture].keys():
            json_dict[split][capture][cam_key].sort(key=lambda x: int(x.split('image')[-1].split('.')[0]))
        # json_dict[split][capture].sort(key=lambda x: (x[0].split('image')[0], int(x[0].split('image')[-1].split('.')[0])))
with open(json_path, "w") as f:
    json.dump(json_dict, f, indent=4)
print(f"Save sorted json to {json_path}")