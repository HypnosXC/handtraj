    
import os
import imageio.v3 as iio
import h5py
from tqdm import tqdm
mp4_format = "original_{}_Capture{}_{}_cam{}_{}_{}_{}.mp4"
hdf5_path = "/public/datasets/handdata/interhand26m_v3.hdf5"
original_video_root = "/public/datasets/handdata/interhand26m/data/picked_videos"
new_video_root = "/public/datasets/handdata/interhand26m/data/fixed_videos"
train_new_video_root = os.path.join(new_video_root, 'train')
os.makedirs(train_new_video_root, exist_ok=True)
# /public/datasets/handdata/interhand26m/data/picked_videos/train/original_val_Capture0_ROM04_RT_Occlusion_cam400364_320_64_right.mp4
with h5py.File(hdf5_path, "r") as f:
    dataset = f['train']
    for index in tqdm(range(len(dataset['video_name']))):
        video_name = dataset['video_name'][index][0].decode('utf-8')
        try:
            iio.imread(os.path.join(original_video_root, 'train', video_name))
        except:
            original_split = video_name.split('_')[1]
            cap_idx = video_name.split('_')[2].replace('Capture','')
            cam_idx = video_name.split('_cam')[1].split('_')[0]
            manoside = video_name.split('_')[-1].split('.')[0]
            seq = '_'.join(video_name.split('_')[3:-4])
            start_idx = int(video_name.split('_')[-3])
            save_length = int(video_name.split('_')[-2])
            cam_dir = os.path.join("/public/datasets/handdata/interhand26m/data/InterHand2.6M_30fps_batch1/images", original_split, f"Capture{cap_idx}", seq, f"cam{cam_idx}")
            frame_list = os.listdir(cam_dir)
            frame_list = [f.split('.')[0].split('image')[-1] for f in frame_list]
            frame_list = sorted(frame_list, key=lambda x:int(x))[start_idx:start_idx+save_length]
            all_frames = []
            for frame_id in frame_list:
                frame_path = os.path.join(cam_dir, f"image{frame_id}.jpg")
                image = iio.imread(frame_path)
                all_frames.append(image)
            save_mp4_path = os.path.join(train_new_video_root, video_name)
            iio.imwrite(save_mp4_path, all_frames, fps=30, codec="libx264", macro_block_size=None)
            # print(f"Saved fixed video to {save_mp4