from egoallo.data.dex_ycb import DexYCBHdf5Dataset
from hamer_helper import HamerHelper
import h5py
import numpy as np
import torch
import tqdm
import os
import pickle

def preprocess_hamer_all():
    train_dataset = DexYCBHdf5Dataset("/public/datasets/handdata/dexycb_s_all_unseen.hdf5", split='train')
    val_dataset = DexYCBHdf5Dataset("/public/datasets/handdata/dexycb_s_all_unseen.hdf5", split='val')
    test_dataset = DexYCBHdf5Dataset("/public/datasets/handdata/dexycb_s_all_unseen.hdf5", split='test')
    os.makedirs("/public/datasets/handdata/hamer_dexycb", exist_ok=True)
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    shapes = {
        "mano_joint_3d": (64,1,21, 3),
        "mano_betas": (10,),
        "mano_pose": (64, 48),
        "mano_hand_global_orient": (64, 3),
        "mano_side": (1,),
    }

    dtypes = {
        "mano_joint_3d":"f4",
        "mano_betas":"f4",
        "mano_pose":"f4",
        "mano_hand_global_orient":"f4",
        "mano_side": h5py.string_dtype('utf-8', None)
    }
    # compression = "gzip"
    hamer_helper = HamerHelper()
    # with h5py.File(hamer_hdf5_path, 'w') as f:
    for split in ['train', 'val', 'test']:
        # print(f"Saving split '{split}' to {hamer_hdf5_path}")
        # N = len(datasets[split])
        # grp = f.create_group(split)              

        # for key in shapes.keys():
        #     shape = (N,) + shapes[key]
        #     chunks = (1,) + shapes[key]
        #     ds = grp.create_dataset(
        #         name=key,
        #         shape=shape,
        #         dtype=dtypes[key],
        #         compression=compression,
        #         chunks=chunks
        #     )
        print("Start processing split:", split)
        os.makedirs(f"/public/datasets/handdata/hamer_dexycb/{split}", exist_ok=True)
        for idx, sample in enumerate(tqdm.tqdm(datasets[split])):
            hamer_pkl_path = f"/public/datasets/handdata/hamer_dexycb/{split}/{idx:05d}.pkl"
            if os.path.exists(hamer_pkl_path):
                print(f"Skipping {hamer_pkl_path}, already exists.")
                continue
            hamer_output_seq = []
            for i in range(64):
                hamer_out_frame = {}            
                hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
                    sample.rgb_frames[i].numpy().astype(np.uint8),
                    focal_length=sample.intrinsics[0].item(),
                )

                if hamer_out_left is None:
                    hamer_out_frame["left"] = None
                else:
                    hamer_out_frame["left"] = {
                        # "verts": hamer_out_left["verts"],
                        "keypoints_3d": hamer_out_left["keypoints_3d"],
                        "mano_hand_pose": hamer_out_left["mano_hand_pose"],
                        "mano_hand_betas": hamer_out_left["mano_hand_betas"],
                        "mano_hand_global_orient": hamer_out_left["mano_hand_global_orient"],
                    }

                if hamer_out_right is None:
                    hamer_out_frame["right"] = None
                else:
                    hamer_out_frame["right"] = {
                        # "verts": hamer_out_right["verts"],
                        "keypoints_3d": hamer_out_right["keypoints_3d"],
                        "mano_hand_pose": hamer_out_right["mano_hand_pose"],
                        "mano_hand_betas": hamer_out_right["mano_hand_betas"],
                        "mano_hand_global_orient": hamer_out_right["mano_hand_global_orient"],
                    }
                hamer_output_seq.append(hamer_out_frame)
            # save to pkl
            with open(hamer_pkl_path, 'wb') as f:
                pickle.dump(hamer_output_seq, f)

    return

if __name__ == "__main__":
    preprocess_hamer_all()