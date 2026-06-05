from typing import Callable, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from torch.amp import autocast, GradScaler

from src.models.policies.drpg.logging import log_to_tensorboard








# ------------------------------------------------------------------------------------
# TRAINING & EVAL HELPERS
# ------------------------------------------------------------------------------------

def training(
    policy: nn.Module,
    reward_f: Callable,
    optimizer: torch.optim.Optimizer,
    trainloader: DataLoader,
    writer: Optional[SummaryWriter],
    device: str,
    reward_components_map: List[str],
    step: int,
    amp_dtype: torch.dtype,
    enable_amp: bool,
    scaler: GradScaler,
) -> None:
    
    """

    Train the policy for one pass over the training dataloader.

    Parameters
    ----------
    policy : nn.Module
        Policy network.
    reward_f : Callable
        Reward function.
    optimizer : torch.optim.Optimizer
        Policy optimizer.
    trainloader : DataLoader
        Training data iterator.
    writer : Optional[SummaryWriter]
        TensorBoard writer.
    device : str
        Compute device.
    reward_components_map : List[str]
        Names of reward components for logging.
    step : int
        Global logging step.
    amp_dtype : torch.dtype
        AMP precision type.
    enable_amp : bool
        Whether AMP is enabled.
    scaler : GradScaler
        AMP gradient scaler.

    Returns
    -------
    None

    """

    policy.train()

    # Track averaged rewards across the epoch
    mean_reward = torch.tensor(
        data=[0.0], 
        dtype=torch.float32,
        device=device, 
        requires_grad=False
    )
    mean_reward_components = torch.tensor(
        data=[0.0 for _ in range(len(reward_components_map))],
        dtype=torch.float32, 
        device=device, 
        requires_grad=False
    )

    batches_in_epoch_train = len(trainloader)

    for batch in tqdm(trainloader):
        rewards, reward_components = train_step(
            X=torch.concat([batch["images"], batch["labels"].unsqueeze(1)], dim=1),
            weights=batch["weights"],
            policy=policy,
            policy_optimizer=optimizer,
            reward_model=reward_f,
            scaler=scaler,
            amp_dtype=amp_dtype,
            enable_amp=enable_amp,
            device=device
        )
        
        trainloader.loader.feedback(batch["idx"], rewards)

        mean_reward += torch.mean(rewards.detach().float())
        mean_reward_components += torch.mean(reward_components.detach().float(), dim=1)


    mean_reward = mean_reward.item() / batches_in_epoch_train
    mean_reward_components = (mean_reward_components / batches_in_epoch_train).tolist()

    # log metrics
    log_to_tensorboard(
        writer=writer,
        values=[*mean_reward_components, mean_reward],
        names=[*reward_components_map, "mean_reward"],
        step=step,
        split="train"
    )


def evaluation(
        policy: torch.nn.Module, 
        reward_f: Callable, 
        evalloader: DataLoader, 
        writer: Optional[SummaryWriter], 
        device: str, 
        reward_components_map: List[str],
        step: int,
        amp_dtype: torch.dtype,
        enable_amp: bool,
        visualization: Optional[Callable] = None,
    ):

    """
    Evaluate the policy over one epoch.

    Computes mean reward and per-component metrics over the evaluation loader,
    logs them, and optionally triggers visualization callbacks.

    Returns
    -------
    float
        Mean evaluation reward.

    """

    policy.eval()
 
    mean_reward_eval = torch.tensor(
        data=[0.0], 
        dtype=torch.float32,
        device=device, 
        requires_grad=False
    )
    mean_reward_eval_components = torch.tensor(
        data=[0.0 for _ in range(len(reward_components_map))],
        dtype=torch.float32, 
        device=device, 
        requires_grad=False
    )

    batches_in_epoch_val = len(evalloader)

    valid_paths = 0
    
    for batch in tqdm(evalloader):
        reward_tuple, valids = eval_step(
            X=torch.concat([batch["images"], batch["labels"].unsqueeze(1)], dim=1),
            policy=policy,
            reward_model=reward_f,
            amp_dtype=amp_dtype,
            enable_amp=enable_amp,
            device=device
        )

        rewards, reward_components = reward_tuple
        valid_paths += valids

        mean_reward_eval += torch.mean(rewards.detach().float())
        mean_reward_eval_components += torch.mean(reward_components.detach().float(), dim=1)

    mean_reward_eval = mean_reward_eval.item() / batches_in_epoch_val
    mean_reward_eval_components = (mean_reward_eval_components / batches_in_epoch_val).tolist()

    mean_validity = valid_paths / batches_in_epoch_val  # normalize by number of samples

    # log metrics
    log_to_tensorboard(
        writer=writer,
        values=[*mean_reward_eval_components, mean_reward_eval, mean_validity],
        names=[*reward_components_map, "mean_reward", "mean_astar_validity"],
        step=step,
        split="eval"
    )

    if visualization is not None:
        visualization(step)

    return mean_reward_eval


# @torch.compile(dynamic=False)
def train_step(
    X: torch.Tensor,
    weights: torch.Tensor,
    policy: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    reward_model: nn.Module,
    scaler: GradScaler,
    amp_dtype: torch.dtype,
    enable_amp: bool,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """

    Perform a single training step: forward → reward → backward → optimizer step.

    Parameters
    ----------
    X : torch.Tensor
        Input batch tensor containing state) + target path channels (3+1 channels).
    policy : nn.Module
        Policy model producing predicted action maps.
    policy_optimizer : torch.optim.Optimizer
        Optimizer updating policy parameters.
    reward_model : nn.Module
        Differentiable reward function.
    scaler : GradScaler
        Gradient scaler used for mixed precision fp16.
    amp_dtype : torch.dtype
        AMP precision (fp16 or bf16).
    enable_amp : bool
        Enable AMP autocasting.
    device : str
        Compute device.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        rewards : Tensor of shape [B]
        reward_components : Tensor of shape [num_components, B]

    """

    with autocast(dtype=amp_dtype, enabled=enable_amp, device_type=device):
        # Action by the current actor for the sampled state
        actions, _ = policy(X)
        
        with autocast(enabled=False, device_type=device):
            thresholded_actions_pi = (actions + 1) / 2

        rewards, reward_components = reward_model(
            state=X[:, :3],
            predicted_path=thresholded_actions_pi.reshape(-1, 1, *X.shape[2:]),
            target_path=X[:, 3:4]  # keep dim
        )

        actor_loss = -(rewards * weights).mean()

    # Optimize the actor
    policy_optimizer.zero_grad()

    scaler.scale(actor_loss).backward()

    scaler.unscale_(policy_optimizer)

    torch.nn.utils.clip_grad_norm_(policy.actor_model.actor_model.parameters(), max_norm=1.0)

    # Optimization step
    scaler.step(policy_optimizer)
    scaler.update()
                
    return rewards, reward_components


def eval_step(
    X: torch.Tensor,
    policy: nn.Module,
    reward_model: nn.Module,
    amp_dtype: torch.dtype,
    enable_amp: bool,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """

    Perform a forward-only evaluation step.

    Parameters
    ----------
    X : torch.Tensor
        Input batch.
    policy : nn.Module
        Policy network (evaluated in deterministic mode).
    reward_model : nn.Module
        Reward function, evaluated without gradient tracking.
    amp_dtype : torch.dtype
        AMP precision.
    enable_amp : bool
        Whether AMP autocast is enabled.
    device : str
        Compute device.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        rewards and reward components.

    """

    # Action by the current actor for the sampled state
    with autocast(dtype=amp_dtype, enabled=enable_amp, device_type=device):
        actions, _ = policy(X, deterministic=True)
        actions = actions.reshape(-1, 1, *X.shape[2:])
        
        with autocast(enabled=False, device_type=device):
            thresholded_actions_pi = (actions + 1) / 2

        state = X[:, :3]

        rewards = reward_model(
            state=state,
            predicted_path=thresholded_actions_pi,
            target_path=X[:, 3:4],  # keep dim
            eval=True
        )

    indices = torch.nonzero(
        (state[:, 0] == 1.0) &
        (state[:, 1] == 76 / 255) &
        (state[:, 2] == 76 / 255)
    )  # [B, 3] → (batch, row, col)

    starts = indices[:, 1:]  # [B, 2]

    indices = torch.nonzero(
        (state[:, 0] == 76 / 255) &
        (state[:, 1] == 1.0) &
        (state[:, 2] == 76 / 255)
    )  # [B, 3] → (batch, row, col)

    ends = indices[:, 1:]  # [B, 2]

    # calculate validity
    binarized_actions = (actions >= 0.0).float()  # [B, 1, H, W]

    validities = is_connected(
        path_pred=binarized_actions,
        starts=starts,
        ends=ends
    )

    valids = torch.sum((validities >= 0.99)).item() / X.size(0)  # normalize by batch size
                
    return rewards, valids


def is_connected(
    path_pred: torch.Tensor,
    starts: torch.Tensor, 
    ends: torch.Tensor, 
    steps: int = 500,  # 125, 
):

    B, _, H, W = path_pred.shape

    # Convert flattened indices back to 2D coordinates (x, y)
    start_x = starts[:, 1]  # [B] x-coordinate
    start_y = starts[:, 0]  # [B] y-coordinate
    end_x = ends[:, 1]      # [B]
    end_y = ends[:, 0]    # [B]

    # Initialize seeds with zeros
    seeds = torch.zeros(
        size=path_pred.shape,  # [B,1,H,W]
        device=path_pred.device,
        dtype=torch.float32
    )

    # Include the end location in the path copy so end can be "reached" even if the predicted
    # path stops adjacent to the end pixel.
    path_pred_with_end_included = torch.clone(path_pred)  # [B,1,H,W]
    path_pred_with_end_included[torch.arange(B), 0, end_y, end_x] = 1.0  # broadcast index by batch

    # Place a seed at the start point (use the batch indices and the coordinates)
    seeds[torch.arange(B), 0, start_y, start_x] = 1.0  # seeds now contain 1 at start locations

    reachable = seeds.clone()  # [B,1,H,W]

    for _ in range(steps):
        # Dilate reachable area using 3x3 max-pool (differentiable morphological dilation)
        reachable = torch.nn.functional.max_pool2d(
            input=reachable, 
            kernel_size=3, 
            stride=1,
            padding=1
        )
        # Only keep propagation where path_pred_with_end_included allows (i.e., along the predicted path)
        reachable = reachable * path_pred_with_end_included  # [B,1,H,W]

    # Read connectivity at the end point (extract the value at the end coordinates)
    return reachable[torch.arange(B), 0, end_y, end_x]  # [B]