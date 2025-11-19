import dataclasses
import shutil
from pathlib import Path
from typing import Literal

import tensorboardX
import torch
import torch.optim.lr_scheduler
import torch.utils.data
import tyro
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed
from loguru import logger

from src.egoallo import hand_network, network, training_loss, training_utils
from src.egoallo.data.amass import EgoAmassHdf5Dataset
from src.egoallo.data.dataclass import collate_dataclass
from src.egoallo.data.hand_data import HandHdf5Dataset

@dataclasses.dataclass(frozen=True)
class HandTrainConfig:
    experiment_name: str
    # dataset_hdf5_path: Path
    # dataset_files_path: Path

    model: hand_network.HandDenoiserConfig = hand_network.HandDenoiserConfig()
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

    # Dataset arguments.
    batch_size: int = 256
    """Effective batch size per GPU."""
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
    seed: int = 42
    """Random seed for reproducibility."""

def get_experiment_dir(experiment_name: str, version: int = 0) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version
    number. Similar to PyTorch lightning."""
    experiment_dir = (
        Path(__file__).absolute().parent
        / "experiments"
        / experiment_name
        / f"v{version}"
    )
    if experiment_dir.exists():
        return get_experiment_dir(experiment_name, version + 1)
    else:
        return experiment_dir

def run_training(
    config: HandTrainConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    torch.multiprocessing.set_start_method('spawn')
    # Initialize Accelerator for multi-GPU support
    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=str(get_experiment_dir(config.experiment_name)),
            logging_dir=str(get_experiment_dir(config.experiment_name) / "logs")
        ),
        dataloader_config=DataLoaderConfiguration(
            split_batches=True,
            use_seedable_sampler=True
        ),
        mixed_precision="fp16",  # Enable mixed precision for better performance
    )
    
    # Set random seed for reproducibility across GPUs
    set_seed(config.seed)

    # Setup experiment directory
    experiment_dir = get_experiment_dir(config.experiment_name)
    if accelerator.is_main_process:
        experiment_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize TensorBoard writer only on main process
    writer = (
        tensorboardX.SummaryWriter(logdir=str(experiment_dir), flush_secs=10)
        if accelerator.is_main_process
        else None
    )
    device = accelerator.device

    # Initialize experiment
    if accelerator.is_main_process:
        training_utils.pdb_safety_net()
        
        # Save experiment metadata
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        (experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

        # Add hyperparameters to TensorBoard
        assert writer is not None
        writer.add_hparams(
            hparam_dict=training_utils.flattened_hparam_dict_from_dataclass(config),
            metric_dict={},
            name=".",  # Hack to avoid timestamped subdirectory
        )

        # Write logs to file
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")

    # Setup model and data loader
    model = hand_network.HandDenoiser(config.model)
    model_config = model.config
    train_dataset = HandHdf5Dataset(
        # config.dataset_hdf5_path,
        # config.dataset_files_path,
        # splits=config.train_splits,
        # subseq_len=config.subseq_len,
        # cache_files=True,
        # slice_strategy=config.dataset_slice_strategy,
        # dataset_name='dexycb',
        # vis=True
        # min_len=32,
        # clip_stride=16,
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=collate_dataclass,
        drop_last=True,
        # Use DistributedSampler for multi-GPU
        sampler=torch.utils.data.distributed.DistributedSampler(train_dataset) if accelerator.num_processes > 1 else None,
    )

    # Setup optimizer and scheduler
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    )

    # Prepare objects for distributed training
    model, train_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, optim, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)

    #Restore checkpoint if provided
    restore_checkpoint_dir = (Path(__file__).absolute().parent
        / "experiments"
        / config.experiment_name
        / "v1"
        / "checkpoints_315000")
    if restore_checkpoint_dir is not None:
        accelerator.load_state(str(restore_checkpoint_dir))

    # Get initial step count
    if restore_checkpoint_dir is not None and restore_checkpoint_dir.name.startswith("checkpoint_"):
        step = int(restore_checkpoint_dir.name.partition("_")[2])
    else:
        step = int(scheduler.state_dict()["last_epoch"])
        assert step == 0 or restore_checkpoint_dir is not None, step

    # Save initial checkpoint
    if accelerator.is_main_process:
        accelerator.save_state(str(experiment_dir / f"checkpoints_{step}"))

    # Initialize loss helper and metrics
    loss_helper = training_loss.TrainingLossComputer(config.loss, device=device)
    loop_metrics_gen = training_utils.loop_metric_generator(counter_init=step)
    prev_checkpoint_path: Path | None = None
    error_cnt = 0

    # Training loop
    while True:
        for train_batch in train_loader:
            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter
            
            # Synchronize gradients across GPUs
            with accelerator.accumulate(model):
                loss, log_outputs = loss_helper.compute_hand_denoising_loss(
                    model,
                    unwrapped_model=accelerator.unwrap_model(model),
                    train_batch=train_batch,
                    using_mat=model_config.using_mat,
                    using_img_feat=model_config.using_img_feat,
                )

                if torch.isnan(loss).any():
                    if accelerator.is_main_process:
                        print("Encountered NaN, problematic data are saved!")
                        if error_cnt < 10:
                            save_path = experiment_dir / f"error_batch_{step}.pt"
                            torch.save(train_batch, save_path)
                        error_cnt += 1
                    continue

                log_outputs["learning_rate"] = scheduler.get_last_lr()[0]
                accelerator.log(log_outputs, step=step)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            # Logging and checkpointing on main process only
            if accelerator.is_main_process:
                if step % 10 == 0:
                    assert writer is not None
                    for k, v in log_outputs.items():
                        writer.add_scalar(k, v, step)

                if step % 20 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    logger.info(
                        f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                        f" mem: {(mem_total - mem_free) / 1024**3:.2f}/{mem_total / 1024**3:.2f}G"
                        f" lr: {scheduler.get_last_lr()[0]:.7f}"
                        f" loss: {loss.item():.6f}"
                        f" rank: {accelerator.process_index}/{accelerator.num_processes}"
                    )

                if step % 5000 == 0:
                    checkpoint_path = experiment_dir / f"checkpoints_{step}"
                    accelerator.save_state(str(checkpoint_path))
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    if prev_checkpoint_path is not None:
                        shutil.rmtree(prev_checkpoint_path)
                    prev_checkpoint_path = None if step % 100_000 == 0 else checkpoint_path
                    del checkpoint_path

            # Synchronize processes to ensure consistent state
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    tyro.cli(run_training)