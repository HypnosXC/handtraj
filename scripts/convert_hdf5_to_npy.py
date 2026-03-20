"""Convert HDF5 visual features to per-group .npy files in float16.

This eliminates HDF5 lock contention between DataLoader workers and halves
disk I/O by using float16.

Usage:
    python scripts/convert_hdf5_to_npy.py \
        --feat_root /public/home/annie/preprocessed/dino_feats \
        --out_root /public/home/annie/preprocessed/dino_feats_npy \
        --datasets dexycb interhand26m arctic ho3d \
        --splits train test

Output structure:
    out_root/
        dexycb_train/
            <group_name>.npy     # float16, shape (T, 768, 16, 16)
        interhand26m_train/
            ...
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def convert_one(hdf5_path: str, out_dir: str, key: str = "layer_11") -> None:
    """Convert one HDF5 feature file to per-group .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        group_names = list(f.keys())
        print(f"  {hdf5_path}: {len(group_names)} groups")
        for gn in tqdm(group_names, desc=f"  {os.path.basename(out_dir)}"):
            out_path = os.path.join(out_dir, f"{gn}.npy")
            if os.path.exists(out_path):
                continue
            data = f[gn][key][:]  # (T, 768, 16, 16) float32
            np.save(out_path, data.astype(np.float16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_root", type=str,
                        default="/public/home/annie/preprocessed/dino_feats")
    parser.add_argument("--out_root", type=str,
                        default="/public/home/annie/preprocessed/dino_feats_npy")
    parser.add_argument("--datasets", nargs="+",
                        default=["dexycb", "interhand26m", "arctic", "ho3d"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--key", type=str, default="layer_11",
                        help="HDF5 dataset key to convert")
    args = parser.parse_args()

    for ds in args.datasets:
        for split in args.splits:
            hdf5_name = f"{ds}_{split}_dino_fpn.hdf5"
            hdf5_path = os.path.join(args.feat_root, hdf5_name)
            if not os.path.exists(hdf5_path):
                print(f"Skipping {hdf5_path} (not found)")
                continue
            out_dir = os.path.join(args.out_root, f"{ds}_{split}")
            print(f"Converting {hdf5_path} -> {out_dir}/")
            convert_one(hdf5_path, out_dir, key=args.key)

    print("Done.")


if __name__ == "__main__":
    main()
