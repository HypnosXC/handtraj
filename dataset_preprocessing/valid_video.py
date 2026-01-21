# video_path='/public/datasets/handdata/interhand26m/data/picked_videos/train/original_val_Capture0_ROM04_RT_Occlusion_cam400364_320_64_right.mp4'
video_path = "/public/datasets/handdata/interhand26m/data/picked_videos/train/original_val_Capture0_ROM04_RT_Occlusion_cam400364_704_64_right.mp4"
import imageio.v3 as iio
# import cv2

rgb_img = iio.imread(video_path)
for i in range(len(rgb_img)):
    breakpoint()
    print(f'Processing frame {i}')