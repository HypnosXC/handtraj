import argparse
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
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs

from accelerate.utils import ProjectConfiguration, set_seed
from loguru import logger

from src.egoallo import hand_network, network, training_loss, training_utils
from src.egoallo.data.amass import EgoAmassHdf5Dataset
from src.egoallo.data.dataclass import collate_dataclass, HandTrainingData
from src.egoallo.data.hand_data import HandHdf5Dataset, FeatureOnlyDataset

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
    num_workers: int = 8
    prefetch_factor: int = 4
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic", "random_uniform_len", "random_variable_len"
    ] = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    dataset_name: str = "all"
    use_feature: str = "visual_token"
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

def load_config_from_yaml(yaml_path: str) -> HandTrainConfig:
    """Load HandTrainConfig from a YAML file."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    # Build model config
    model_kwargs = raw.pop("model", {})
    model_config = hand_network.HandDenoiserConfig(**model_kwargs)

    # Build loss config
    loss_kwargs = raw.pop("loss", {})
    # Convert beta_coeff_weights list to tuple if present
    if "beta_coeff_weights" in loss_kwargs:
        loss_kwargs["beta_coeff_weights"] = tuple(loss_kwargs["beta_coeff_weights"])
    loss_config = training_loss.TrainingLossConfig(**loss_kwargs)

    # Build top-level config
    # Convert train_splits list to tuple if present
    if "train_splits" in raw:
        raw["train_splits"] = tuple(raw["train_splits"])

    return HandTrainConfig(model=model_config, loss=loss_config, **raw)


def run_training(
    config: HandTrainConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    torch.multiprocessing.set_start_method('spawn')

    # Initialize Accelerator for multi-GPU support
    # Increase NCCL timeout to allow for long dataset preloading across nodes.
    import datetime
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=True,
    )
    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=str(get_experiment_dir(config.experiment_name)),
            logging_dir=str(get_experiment_dir(config.experiment_name) / "logs")
        ),
        dataloader_config=DataLoaderConfiguration(
            split_batches=True,
            use_seedable_sampler=True
        ),
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
    )
    # Set a longer timeout for NCCL operations (default 600s is too short for preloading).
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        store = torch.distributed.distributed_c10d._get_default_store()
        store.set_timeout(datetime.timedelta(seconds=7200))
    
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
        subseq_len=config.subseq_len,
        dataset_name=config.dataset_name,
        use_feature=config.use_feature,
    )
    print("process at dataset", config.dataset_name)

    feat_dataset = FeatureOnlyDataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=feat_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        drop_last=True,
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

    # Preload all small fields to GPU AFTER accelerator.prepare() so NCCL is
    # already initialized and won't timeout during the long preloading phase.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Preloading small fields to GPU...")
    gpu_cache = train_dataset.build_gpu_cache(device)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cache_mb = sum(v.nbytes for v in gpu_cache.values()) / 1e6
        print(f"GPU cache: {cache_mb:.0f} MB, {len(train_dataset)} samples")
    accelerator.register_for_checkpointing(scheduler)

    #Restore checkpoint if provided
    # restore_checkpoint_dir = (Path(__file__).absolute().parent
    #     / "experiments"
    #     / config.experiment_name
    #     / "v1"
    #     / "checkpoints_315000")
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
        for batch_indices, batch_img_feat in train_loader:
            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter

            # Build HandTrainingData: small fields from GPU cache, features from DataLoader.
            idx = batch_indices.to(device)
            train_batch = HandTrainingData(
                mano_betas=gpu_cache["mano_betas"][idx],
                mano_pose=gpu_cache["mano_pose"][idx],
                mano_joint_3d=gpu_cache["mano_joint_3d"][idx],
                joint_2d=gpu_cache["joint_2d"][idx],
                intrinsics=gpu_cache["intrinsics"][idx],
                mask=gpu_cache["mask"][idx],
                mano_side=gpu_cache["mano_side"][idx],
                extrinsics=gpu_cache["extrinsics"][idx],
                img_feature=batch_img_feat.to(device),
                img_shape=gpu_cache["img_shape"][idx],
                rgb_frames=torch.zeros(idx.shape[0], config.subseq_len, dtype=torch.uint8, device=device),
            )

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
    # Check if --config is provided; if so, load from YAML, otherwise fall back to tyro CLI
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--restore_checkpoint_dir", type=str, default=None, help="Path to checkpoint dir to restore from")
    args, remaining = parser.parse_known_args()

    if args.config is not None:
        config = load_config_from_yaml(args.config)
        restore_dir = Path(args.restore_checkpoint_dir) if args.restore_checkpoint_dir else None
        run_training(config, restore_checkpoint_dir=restore_dir)
    else:
        tyro.cli(run_training)