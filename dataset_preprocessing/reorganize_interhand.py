v1_path = "/public/datasets/handdata/interhand26m/data/picked_videos"
v_fix_path = "/public/datasets/handdata/interhand26m/data/fixed_videos"
# copy all the videos from v_fix_path/train to v1_path/train (cover the video with the same name)
import os
import shutil
from tqdm import tqdm
split='train'
for video_name in tqdm(os.listdir(os.path.join(v_fix_path, split))):
    src_path =  os.path.join(v_fix_path, split, video_name)
    dst_path =  os.path.join(v1_path, split, video_name)
    shutil.copyfile(src_path, dst_path)