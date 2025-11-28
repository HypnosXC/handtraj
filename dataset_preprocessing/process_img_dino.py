import numpy as np
import torch
import multiprocessing as mp
import os
from tqdm import tqdm
from egoallo.data.hand_data import HandHdf5Dataset
split = 'test'
dataset_name = 'all'  # 'dexycb' 'interhand26m' 'arctic'
img_feat_root = {
    'dexycb':'/public/datasets/handdata/dexycb/img_feats_dino',
    'interhand26m':'/public/datasets/handdata/interhand26m/data/img_feats_dino',
    'arctic':'/public/datasets/handdata/arctic/img_feats_dino',
    'ho3d':'/public/datasets/handdata/ho3d/img_feats_dino'
}

# this_root = img_feat_root[dataset_name]
# os.makedirs(os.path.join(this_root,split), exist_ok=True)
for ds_name in img_feat_root:
    this_root = img_feat_root[ds_name]
    os.makedirs(os.path.join(this_root,split), exist_ok=True)
# dataset = HandHdf5Dataset(split=split , dataset_name=dataset_name,vis=True)
task_n = 6

import torchvision
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def worker(idx,q,split, dataset_name, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    # if device is not None and torch.cuda.is_available():
    #     torch.cuda.set_device(torch.device(device))
    import torch
    # from hamer_helper import HamerHelper
    import numpy as np
    dataset = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True, subseq_len=-1, min_len=1)
    mapping = dataset.get_mapping()
    batch_size = 1024
    dinov3_vitb16 = torch.hub.load('/public/home/group_ucb/yunqili/code/dinov3', 'dinov3_vitb16', source='local', weights='/public/home/group_ucb/yunqili/data/dino_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth').to("cuda")
    # hamer_helper = HamerHelper()
    transform = make_transform()
    for sample_id in range(idx, len(dataset),task_n):
        this_dataset_name, group_name,_,_ = mapping[sample_id]
        this_root = img_feat_root[this_dataset_name]
        if os.path.exists(os.path.join(this_root, split, f'imgfeat_{group_name}.pt')):
            q.put((idx, 1))
            continue
        sample=dataset.__getitem__(sample_id,resize=None)
        # mano_side=sample.mano_side
        img_features = []

        # video to batches of frames
        # T,H,W,3 to T,3,H,W
        frames = sample.rgb_frames.permute(0,3,1,2)  # T,3,H,W
        
        num_frames = frames.shape[0]
        preprocessed_feats = []
        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(start_idx + batch_size, num_frames)
            batch_frames = frames[start_idx:end_idx]  # B, 3, H, W
            # Apply transformations
            batch_frames_transformed = transform(batch_frames)  # B, 3, resize_size, resize_size
            with torch.no_grad():
                batch_feats = dinov3_vitb16(batch_frames_transformed.to("cuda"))  # B, feat_dim, h, w
            preprocessed_feats.append(batch_feats.cpu())
        img_features = torch.cat(preprocessed_feats, dim=0)  # T, feat_dim, h, w

        # for frame_id in range(sample.rgb_frames.shape[0]):
        #     image=sample.rgb_frames[frame_id].numpy().astype(np.uint8)
        #     img_feat = ...
        #     # img_feat=hamer_helper.get_img_feats(image, mano_side) # Int[np.ndarray, "batch height width 3"]
        #     img_features.append(img_feat.cpu())
        # img_features = torch.stack(img_features, dim=0) # T,1280
        # print("Shape of rgb_frames:", sample.rgb_frames.shape)

        # img_features = hamer_helper.get_img_feats(sample.rgb_frames.numpy().astype(np.uint8), mano_side)
        torch.save(img_features, os.path.join(this_root, split, f'imgfeat_{group_name}.pt'))
        q.put((idx, 1))
    q.put((idx, 0))

if __name__ == '__main__':  
    mp.set_start_method('spawn', force=True)
    _tmp_ds = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True, subseq_len=-1, min_len=1)
    task_list = [i for i in range(task_n)]
    totals = []
    for i in range(task_n):
        totals.append(len(range(i, len(_tmp_ds), task_n)))
    del _tmp_ds
    q = mp.Queue(maxsize=10000)
    num_gpus = torch.cuda.device_count()
    # devices = [f'cuda:{w % num_gpus}' for w in range(task_n)]
    devices = [w % num_gpus for w in range(task_n)]

    procs = [mp.Process(target=worker, args=(w, q, split, dataset_name, devices[w])) for w in range(task_n)]
    for p in procs: p.start()

    # 建立多条进度条（position 不同）
    bars = [tqdm(total=totals[w], position=w, desc=f"{dataset_name} {split} W{w}", leave=True) for w in range(task_n)]
    finished = set()
    while len(finished) < task_n:
        wid, inc = q.get()
        if inc > 0:
            bars[wid].update(inc)
        else:
            finished.add(wid)

    for b in bars: b.close()
    for p in procs: p.join()
