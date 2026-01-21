import numpy as np
import torch
import multiprocessing as mp
import os
from tqdm import tqdm
from egoallo.data.hand_data import HandHdf5Dataset
split = 'test'
dataset_name = 'dexycb'  # 'dexycb' 'interhand26m' 'arctic'
img_feat_root = {
    'dexycb':'/public/datasets/handdata/dexycb/img_feats',
    'interhand26m':'/public/datasets/handdata/interhand26m/data/img_feats',
    'arctic':'/public/datasets/handdata/arctic/img_feats'
}
this_root = img_feat_root[dataset_name]
os.makedirs(os.path.join(this_root,split), exist_ok=True)
# dataset = HandHdf5Dataset(split=split , dataset_name=dataset_name,vis=True)
task_n = 4

def worker(idx,q,split, dataset_name, this_root, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    # if device is not None and torch.cuda.is_available():
    #     torch.cuda.set_device(torch.device(device))
    import torch
    from hamer_helper import HamerHelper
    import numpy as np
    dataset = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True)
    hamer_helper = HamerHelper()
    for sample_id in range(idx, len(dataset),task_n):
        if os.path.exists(os.path.join(this_root, split, f'imgfeat_{sample_id}.pt')):
            q.put((idx, 1))
            continue
        sample=dataset.__getitem__(sample_id,resize=None)
        mano_side=sample.mano_side
        img_features = []
        for frame_id in range(sample.rgb_frames.shape[0]):
            image=sample.rgb_frames[frame_id].numpy().astype(np.uint8)
            img_feat=hamer_helper.get_img_feats(image, mano_side) # Int[np.ndarray, "batch height width 3"]
            img_features.append(img_feat.cpu())
        img_features = torch.stack(img_features, dim=0) # T,1280
        # print("Shape of rgb_frames:", sample.rgb_frames.shape)

        # img_features = hamer_helper.get_img_feats(sample.rgb_frames.numpy().astype(np.uint8), mano_side)
        torch.save(img_features, os.path.join(this_root, split, f'imgfeat_{sample_id}.pt'))
        q.put((idx, 1))
    q.put((idx, 0))

if __name__ == '__main__':  
    mp.set_start_method('spawn', force=True)
    _tmp_ds = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True)
    task_list = [i for i in range(task_n)]
    totals = []
    for i in range(task_n):
        totals.append(len(range(i, len(_tmp_ds), task_n)))
    del _tmp_ds
    q = mp.Queue(maxsize=10000)
    num_gpus = torch.cuda.device_count()
    # devices = [f'cuda:{w % num_gpus}' for w in range(task_n)]
    devices = [w % num_gpus for w in range(task_n)]

    procs = [mp.Process(target=worker, args=(w, q, split, dataset_name, this_root, devices[w])) for w in range(task_n)]
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
