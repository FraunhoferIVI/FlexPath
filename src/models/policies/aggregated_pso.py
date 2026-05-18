from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from torch.amp import GradScaler

from src.buffers.replay_buffer import ReplayBufferLoader, CUDAPrefetcher
from src.pso.pso_objective import PSOObjective

from src.pso.pso_registry import REWARD_COMPONENTS

from src.models.policies.pso_training.training.nonreasoning import training, evaluation

from src.models.policies.pso_training.schedule import get_pixel_penalty_schedule_linear

from torch.utils.tensorboard import SummaryWriter

from src.utils.checkpoint import CheckpointManager



# ------------------------------------------------------------------------------------
# AggregatedPSO CLASS
# ------------------------------------------------------------------------------------

class AggregatedPSO(nn.Module):
    """
    AggregatedPSO: Differentiable Reward Policy Gradient trainer.

    This class manages the complete training pipeline for a differentiable
    reinforcement-learning-like policy. It handles:

    - Dataset loading with CUDA prefetching
    - Mixed precision configuration (fp16/bf16)
    - Reward function construction (optionally with pixel-penalty scheduling)
    - Logging (TensorBoard)
    - Visualization callbacks
    - Checkpoint management
    - High-level training & evaluation loops

    Core Attributes
    ---------------
    policy : torch.nn.Module
        Policy network wrapper used to generate actions.

    optimizer : torch.optim.Optimizer
        Optimizer that updates the policy's parameters.

    trainloader, evalloader : CUDAPrefetcher
        Prefetching dataloaders for training and evaluation.

    reward_f : PSOObjective
        Fully differentiable reward function used for both train & eval.

    enable_amp : bool
        Whether AMP (automatic mixed precision) is enabled.

    amp_dtype : torch.dtype
        Precision type used inside autocast.

    scaler : torch.cuda.amp.GradScaler
        Gradient scaler used for fp16 AMP training.

    writer : Optional[SummaryWriter]
        TensorBoard writer (None if logging disabled).

    checkpoint_manager : CheckpointManager
        Utility for automatic checkpoint saving / pruning.

    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        exp_dir: str,
        exp_name: str,
        reward_func: str,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_schedule,
        train_dataset,
        eval_dataset,
        total_epochs: int,
        adaptive_pixel_penalty_scaling: bool = False,
        pixel_penalty_scaling_start_step: int = 0,
        pixel_penalty_scaling_start_val: float = 0.0,
        pixel_penalty_scaling_stop_val: float = 0.0,
        obstacle_penalty_scaling: float = 1.0,
        batch_size: int = 64,
        eval_freq: int = 1000,
        checkpoint_save_freq: int = 1,
        keep_last_n_checkpoints: int = 1,
        device: str = "cuda",
        tensorboard_log: Optional[str] = None,
        use_mixed_precision: bool = False,
        precision: str = "bf16",
    ):
        """
        Initialize the DRPG trainer and configure the full training environment.

        Parameters
        ----------
        exp_dir : str
            Output directory for checkpoints.
        exp_name : str
            Experiment label for logging and visualization.
        reward_func : str
            Name of the reward function to use.
        policy : nn.Module
            Policy network (action generator).
        optimizer : torch.optim.Optimizer
            Optimizer applied to the policy network.
        train_dataset, eval_dataset : Dataset
            Training and evaluation datasets.
        total_epochs : int
            Number of training epochs.
        adaptive_pixel_penalty_scaling : bool
            Whether to schedule pixel penalty scaling.
        pixel_penalty_scaling_start_step : int
            Step at which pixel penalty begins.
        pixel_penalty_scaling_start_val, pixel_penalty_scaling_stop_val : float
            Start/end values for linear penalty scheduling.
        obstacle_penalty_scaling : float
            Scale for the obstacle penalty.
        batch_size : int
            Batch size.
        eval_freq : int
            Number of replay-buffer batches per epoch.
        checkpoint_save_freq : int
            Epoch interval for checkpoint saves.
        keep_last_n_checkpoints : int
            Max number of saved checkpoints.
        device : str
            Training device ("cuda" or "cpu").
        tensorboard_log : Optional[str]
            Directory for tensorboard logs (None disables).
        use_mixed_precision : bool
            Enable PyTorch AMP.
        precision : str
            Precision for AMP ("fp32" / "fp16" / "bf16").
        """

        super(AggregatedPSO, self).__init__()

        # ----------------------------------------------------------------------
        # Store policy and optimizer
        # ----------------------------------------------------------------------
        self.policy = policy
        self.optimizer = optimizer

        self.trainloader = CUDAPrefetcher(  # prefetcher loads data to gpu ahead of time
            loader=ReplayBufferLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_batches=eval_freq
            ),
            device=device
        )

        self.evalloader = CUDAPrefetcher(  # prefetcher loads data to gpu ahead of time
            loader=ReplayBufferLoader(
                dataset=eval_dataset,
                batch_size=batch_size,
                num_batches=len(eval_dataset) // batch_size
            ),
            device=device
        )

        self.device = device

        # ----------------------------------------------------------------------
        # Automatic Mixed Precision (AMP) Setup
        # ----------------------------------------------------------------------
        if precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of: 'fp32', 'fp16', 'bf16'"
            )

        self.enable_amp: bool = (
            use_mixed_precision and device == "cuda" and precision in ["fp16", "bf16"]
        )

        self.amp_dtype: torch.dtype = (
            torch.bfloat16 if precision == "bf16" else torch.float16
        )

        # Gradient scaling is needed only for fp16
        self.enable_grad_scaling: bool = self.enable_amp and precision == "fp16"
        self.scaler: GradScaler = GradScaler(enabled=self.enable_grad_scaling)

        # ----------------------------------------------------------------------
        # Reward function setup (possibly with scheduling)
        # ----------------------------------------------------------------------
        if adaptive_pixel_penalty_scaling:
            pixel_sum_penalty_schedule = get_pixel_penalty_schedule_linear(
                start_step=pixel_penalty_scaling_start_step,
                total_steps=(total_epochs * eval_freq) + 1,
                start_val=pixel_penalty_scaling_start_val,
                stop_val=pixel_penalty_scaling_stop_val,
            )
        else:
            pixel_sum_penalty_schedule = None

        self.reward_f = PSOObjective(
            reward_f=reward_func,
            pixel_sum_penalty_schedule=pixel_sum_penalty_schedule,
            obstacle_penalty_scaling=obstacle_penalty_scaling,
            compile_reward_f=True,
        )

        self.reward_components_map: List[str] = REWARD_COMPONENTS[reward_func]

        # ----------------------------------------------------------------------
        # TensorBoard writer
        # ----------------------------------------------------------------------
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir=tensorboard_log) if tensorboard_log else None
        )

        # ----------------------------------------------------------------------
        # Checkpoint management
        # ----------------------------------------------------------------------
        self.checkpoint_manager = CheckpointManager(
            exp_dir=exp_dir,
            keep_last_n=keep_last_n_checkpoints,
        )
        self.checkpoint_save_freq = checkpoint_save_freq

        self.epochs = total_epochs

        self.lr_schedule = lr_schedule

    # ------------------------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------------------------

    def learn(self) -> None:

        """

        Execute the full training pipeline.

        Steps:
        ------
        1. Initial evaluation before training.
        2. For each epoch:
            - Train for one full pass over the replay buffer
            - Evaluate model performance
            - Optionally visualize outputs
            - Save periodic checkpoints

        Returns
        -------
        None

        """

        batches_in_epoch_train: int = len(self.trainloader)

        # Initial evaluation before any training
        evaluation(
            policy=self.policy,
            reward_f=self.reward_f,
            evalloader=self.evalloader,
            writer=self.writer,
            device=self.device,
            reward_components_map=self.reward_components_map,
            step=0,
            amp_dtype=self.amp_dtype,
            enable_amp=self.enable_amp,
        )

        for epoch in range(self.epochs):

            # -------------------------
            # Train for one epoch
            # -------------------------
            training(
                policy=self.policy,
                reward_f=self.reward_f,
                optimizer=self.optimizer,
                trainloader=self.trainloader,
                writer=self.writer,
                device=self.device,
                reward_components_map=self.reward_components_map,
                step=(epoch + 1) * batches_in_epoch_train,
                amp_dtype=self.amp_dtype,
                enable_amp=self.enable_amp,
                scaler=self.scaler,
            )

            self.lr_schedule.step(epoch)

            # -------------------------
            # Evaluate
            # -------------------------
            mean_reward = evaluation(
                policy=self.policy,
                reward_f=self.reward_f,
                evalloader=self.evalloader,
                writer=self.writer,
                device=self.device,
                reward_components_map=self.reward_components_map,
                step=(epoch + 1) * batches_in_epoch_train,
                amp_dtype=self.amp_dtype,
                enable_amp=self.enable_amp,
            )

            # -------------------------
            # Checkpoint saving
            # -------------------------
            if (epoch - 1) % self.checkpoint_save_freq == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.policy.actor_model,
                    step=epoch,
                    metric_value=mean_reward,
                )

        # save last checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.policy.actor_model,
            step=epoch,
            metric_value=mean_reward,
        )
