import numpy as np
import torch
import multiprocessing as mp
import os
import shutil
from tqdm import tqdm
from egoallo.data.hand_data import HandHdf5Dataset
from src.egoallo.data.dataclass import collate_dataclass
from torchvision.transforms import v2
import h5py


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def work_on_batch(buffer_frames, dinov3_vitb16, transform, layer_indices=[11]):
    batch_t = torch.cat(buffer_frames, dim=0).to("cuda", non_blocking=True)  # B, T, H, W, 3
    # T,H,W,3 to T,3,H,W
    batch_t = batch_t.permute(0, 3, 1, 2)
    batch_t = transform(batch_t)
    with torch.no_grad(),torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        batch_feats = dinov3_vitb16.get_intermediate_layers(
                    batch_t,
                    n=layer_indices,        
                    reshape=True,             
                    return_class_token=True,  # 👈 核心关键点：设为 True
                    norm=True                 
                )

    del batch_t 
    return batch_feats

def work_on_dataset(split, dataset_name, img_feat_root, dinov3_vitb16, transform, buffer_size, layer_indices=[11]):
    hdf5_name = os.path.join(img_feat_root, f'{dataset_name}_{split}_dino_fpn.hdf5')
    if os.path.exists(hdf5_name):
        print(f"{hdf5_name} already exists. Skipping {dataset_name} {split}.")
        return
    
    # local_tmp_hdf5 = f'/tmp/{dataset_name}_{split}_dino_fpn_temp.hdf5'
    # if os.path.exists(local_tmp_hdf5):
    #     os.remove(local_tmp_hdf5)

    dataset = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True, subseq_len=-1, min_len=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=collate_dataclass)
    mapping = dataset.get_mapping()
    # create a hdf5 file in this_root to store the features
    buffer_meta = []
    buffer_frames = []
    size_now=0
    with h5py.File(hdf5_name, 'w') as f:
        for idx, this_batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name} {split}")):
            _, group_name,_,seq_len = mapping[idx]
            size_now += seq_len
            if size_now >= buffer_size:
                batch_feats = work_on_batch(buffer_frames, dinov3_vitb16, transform,layer_indices)
                # write feats into hdf5
                cnt_len = 0
                for b_group_name, b_seq_len in buffer_meta:
                    group = f.create_group(b_group_name)
                    for i, l in enumerate(layer_indices):
                        group.create_dataset(f'layer_{l}', data=batch_feats[i][0][cnt_len:cnt_len+b_seq_len].half().cpu().numpy())
                    group.create_dataset(f'cls_token', data=batch_feats[-1][1][cnt_len:cnt_len+b_seq_len].half().cpu().numpy())
                    cnt_len += b_seq_len

                del batch_feats
                buffer_meta.clear()
                buffer_frames.clear()
                size_now = seq_len
            buffer_frames.append(this_batch.rgb_frames.squeeze(0)) # B, T, H, W, 3
            buffer_meta.append((group_name, seq_len))
        if size_now > 0:
            batch_feats = work_on_batch(buffer_frames, dinov3_vitb16, transform,layer_indices)
            # write feats into hdf5
            cnt_len = 0
            for b_group_name, b_seq_len in buffer_meta:
                group = f.create_group(b_group_name)
                for i, l in enumerate(layer_indices):
                    group.create_dataset(f'layer_{l}', data=batch_feats[i][0][cnt_len:cnt_len+b_seq_len].half().cpu().numpy())
                group.create_dataset(f'cls_token', data=batch_feats[-1][1][cnt_len:cnt_len+b_seq_len].half().cpu().numpy())
                cnt_len += b_seq_len   
            del batch_feats
            buffer_meta.clear()
            buffer_frames.clear() 
    # print(f"Finished locally processing {dataset_name} {split}. Now copying to {hdf5_name}...")
    # shutil.move(local_tmp_hdf5, hdf5_name)
    # print(f"Finished processing {dataset_name} {split}. Saved to {hdf5_name}.")

if __name__ == '__main__':  
    splits = ['train', 'test', 'val']
    dataset_names = ['dexycb', 'interhand26m', 'arctic', 'ho3d']
    img_feat_root = "/public/home/annie/preprocessed/dino_feats/"
    os.makedirs(img_feat_root, exist_ok=True)

    dinov3_vitb16 = torch.hub.load('/public/home/annie/code/dinov3', 'dinov3_vitb16', source='local', weights='/public/home/annie/data/dino_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    # dinov3_vitb16 = torch.nn.DataParallel(dinov3_vitb16)
    dinov3_vitb16 = dinov3_vitb16.eval().cuda()
    transform = make_transform().to("cuda")

    buffer_size = 3000

    for split in splits:
        for ds_name in dataset_names:
            work_on_dataset(split, ds_name, img_feat_root, dinov3_vitb16, transform, buffer_size)
