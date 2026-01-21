"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import dataclasses
# import shutil
from pathlib import Path
from typing import Literal

# import tensorboardX
import torch.optim.lr_scheduler
import torch.utils.data
import tyro
# import yaml
# from accelerate import Accelerator, DataLoaderConfiguration
# from accelerate.utils import ProjectConfiguration
# from loguru import logger

from egoallo import network, training_loss, training_utils
from egoallo.data.amass import EgoAmassHdf5Dataset

# from egoallo.data.dex_ycb import DexYCBDataset
# from egoallo.data.dataclass import collate_dataclass


@dataclasses.dataclass(frozen=True)
class EgoAlloTrainConfig:
    experiment_name: str
    dataset_hdf5_path: Path
    dataset_files_path: Path

    model: network.EgoDenoiserConfig = network.EgoDenoiserConfig()
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

    # Dataset arguments.
    batch_size: int = 256
    """Effective batch size."""
    num_workers: int = 2
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic", "random_uniform_len", "random_variable_len"
    ] = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    train_splits: tuple[Literal["train", "val", "test", "just_humaneva"], ...] = (
        "train",
        "val",
    )

    # Optimizer options.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

import yaml

def run_training(
    config_file: str,
    # restore_checkpoint_dir: Path | None = None,
) -> None:
    config = yaml.load(
        Path(config_file).read_text(),
        Loader=yaml.Loader
    )
    assert isinstance(config, EgoAlloTrainConfig)
    dataset=EgoAmassHdf5Dataset(
        config.dataset_hdf5_path,
        config.dataset_files_path,
        splits=config.train_splits,
        subseq_len=config.subseq_len,
        cache_files=True,
        slice_strategy=config.dataset_slice_strategy,
        random_variable_len_proportion=config.dataset_slice_random_variable_len_proportion,
    )
    breakpoint()  # Debugging breakpoint to inspect the dataset
    print(f"Dataset length: {len(dataset)}")


if __name__ == "__main__":
    tyro.cli(run_training)
