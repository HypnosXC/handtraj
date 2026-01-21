import numpy as np
import torch
import multiprocessing as mp
import os
from tqdm import tqdm
from egoallo.data.hand_data import HandHdf5Dataset
import random

def worker(idx,split, dataset_name):
    # if device is not None and torch.cuda.is_available():
    #     torch.cuda.set_device(torch.device(device))
    # import torch
    from hamer_helper import HamerHelper
    import numpy as np
    dataset = HandHdf5Dataset(split=split, dataset_name=dataset_name, vis=True)
    hamer_helper = HamerHelper()
    # for sample_id in range(idx, len(dataset),task_n):
    sample_id = 5962
    frame_id =51
    # for _ in tqdm(range(1000)):
        # sample_id = random.randint(0, len(dataset)-1)
    sample=dataset.__getitem__(sample_id,resize=None)
    mano_side=sample.mano_side
    # img_features = []
        # for frame_id in range(sample.rgb_frames.shape[0]):
    image=sample.rgb_frames[frame_id].numpy().astype(np.uint8)
    img_feat=hamer_helper.get_img_feats(image, mano_side) # Int[np.ndarray, "batch height width 3"]
            # img_features.append(img_feat.cpu())
        # img_features = torch.stack(img_features, dim=0) # T,1280

if __name__ == '__main__':  
    split = 'train'
    dataset_name = 'dexycb'  # 'dexycb' 'inter
    worker(0, split, dataset_name)