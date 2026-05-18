"""
Generic actor wrapper for VSAC.

This module provides a generic wrapper that adds VSAC-specific functionality
(STD calculation, action masking) to ANY pretrained actor model.

Previously, PathFindingNetworkUNET2withSTD hardcoded the actor architecture.
Now, PSOActorWrapper accepts any actor model, enabling architecture ablation studies.

Usage:
    from src.utils.checkpoint_loader import load_pretrained_actor
    from src.models.policies.vsac.actor_wrapper import PSOActorWrapper

    # Load ANY actor model
    actor_model = load_pretrained_actor(
        model_config={"type": "actor_rnt_m", "params": {"num_labels": 1}},
        checkpoint_path="checkpoints/actor_rnt_m.pth.tar",
        device="cuda"
    )

    # Wrap it (works with all actor architectures!)
    wrapper = PSOActorWrapper(actor_model, num_actions=1764)

    # Use in VSAC
    mean, log_std = wrapper(state, std_decay)
"""

import logging
import torch
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

# Constants from original implementation
LOG_STD_MAX = 0
LOG_STD_MIN = -20



def safe_clone(x):
    if type(x) in {list, tuple}:
        return tuple(el.clone() for el in x)
    else:
        return x.clone()

        
class PSOActorWrapper(nn.Module):
    """
    Generic wrapper that adds VSAC-specific functionality to any actor model.

    This wrapper:
    1. Accepts ANY pretrained actor model (ActorUNetBasic, ActorRNT_M, etc.)
    2. Adds state-dependent STD calculation
    3. Adds state-dependent action masking
    4. Returns (mean, log_std) for SAC's Gaussian policy

    The wrapper is architecture-agnostic - it works with any actor model that
    returns logits via the `.logits` attribute when called with `forward()`.

    Attributes:
        actor_model: The pretrained actor model (any architecture)
        num_actions: Number of actions (e.g., 42x42 = 1764 for grid)
    """

    def __init__(self, actor_model: nn.Module, exploration_strategy: str = "v0", compile_model: bool = False):
        """
        Initialize the VSAC actor wrapper.

        Args:
            actor_model: Pretrained actor model (any architecture from registry)
                        Must have .forward() that returns object with .logits attribute
            num_actions: Number of actions in flattened action space
                        Default: 1764 (42x42 grid)

        Example:
            >>> from src.models.actor.actor_unet_basic import ActorUNetBasic
            >>> actor = ActorUNetBasic(num_labels=1)
            >>> wrapper = PSOActorWrapper(actor, num_actions=1764)
        """
        super().__init__()
        self.actor_model = actor_model

        if compile_model:
            self.actor_model = torch.compile(
                model=self.actor_model, 
                backend="inductor", 
                dynamic=True
        )

        # Set exploration strategy
        self.exploration_strategy = exploration_strategy
        
        # Define obstacle color for state dependant log_std masking
        self.obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device="cuda")  # BLUE

        # initalize default params for exploration strategy v2
        self.exploration_region_radius = 3,
        self.base_log_std = -20,
        self.exploration_log_std = 1.1

        self.add_pixel_ratio = 0.5
        self.amount_of_pixels_added_removed = 2
        self.decay = False

    def set_exploration_strategy(self, new_exploration_strategy: str):
        """
        Sets the exploration strategy to use. Valid options: {'v0', 'v1', 'v2', 'v3'}

        """
        
        self.exploration_strategy = new_exploration_strategy

    def modify_exploration_strategy_params(
        self,
        exploration_region_radius: int = None, 
        base_log_std: float = None, 
        exploration_log_std: float = None,
        add_pixel_ratio: float = None,
        amount_of_pixels_added_removed: int = None,
        decay: bool = None
    ):
        """
        Updates the given parameters. If None, the value remains unchanged.

        """
        
        if exploration_region_radius is not None:
            self.exploration_region_radius = exploration_region_radius
        if base_log_std is not None:
            self.base_log_std = base_log_std
        if exploration_log_std is not None:
            self.exploration_log_std = exploration_log_std
        if add_pixel_ratio is not None:
            self.add_pixel_ratio = add_pixel_ratio
        if amount_of_pixels_added_removed is not None:
            self.amount_of_pixels_added_removed = amount_of_pixels_added_removed
        if decay is not None:
            self.decay = decay

    def forward_nonreasoning(self, state: torch.Tensor, std_decay: float = 1.0) -> tuple:
        """
        Forward pass through wrapped actor with VSAC-specific logic.

        Process:
        1. Get logits from actor model
        2. Flatten logits to action space
        3. Calculate state-dependent log_std
        4. Apply state-dependent action masking
        5. Clamp log_std to valid range

        Args:
            state: Observation tensor [B, C, H, W] (e.g., [B, 3, 42, 42])
            std_decay: STD decay value (decreases over training)
                Not currently used in calc_log_std1, but kept for compatibility
            t: Optional timestep tensor used by reasoning actors (shape [B, 1]).

        Returns:
            Tuple of (mean, log_std):
                - mean: Flattened logits [B, num_actions]
                - log_std: State-dependent log standard deviation [B, num_actions]

        Example:
            >>> state = torch.randn(32, 3, 42, 42)  # Batch of 32 observations
            >>> mean, log_std = wrapper(state, std_decay=0.5)
            >>> mean.shape
            torch.Size([32, 1764])
            >>> log_std.shape
            torch.Size([32, 1764])
        """

        if state.shape[1] > 3:
            # extract target (4. obs dim)
            raw_obs = state[:, :3, :, :]
            target_path = state[:, 3, :, :]
        else:
            raw_obs = state
            target_path = None

        # Get logits from actor model (any architecture)
        x = self.actor_model(raw_obs).logits.clone()  # [B, 1, H, W]  # MOD: exclude target path from obs

        # Flatten to action space
        flattened_logits = x.view(x.size(0), -1)  # [B, num_actions]

        if self.exploration_strategy == "v0":
            log_std = self.exploration_strategy_v0(
                flattened_logits=flattened_logits,
            )

        elif self.exploration_strategy == "v1":
            log_std = self.exploration_strategy_v1(
                flattened_logits=flattened_logits,
                state=raw_obs,
                std_decay=std_decay,
            )

        elif self.exploration_strategy == "v2":
            log_std = self.exploration_strategy_v2(
                logits=x,
                state=raw_obs, 
                target_path=target_path,
                std_decay=std_decay,
            )  
            # log_std: [B, H, W] -> flatten to [B, H*W]
            log_std = log_std.view(log_std.size(0), -1)

        else:
            raise RuntimeError("No valid exploration strategy set in PSOActorWrapper.")

        return flattened_logits, log_std
    
    
    def forward(
        self,
        state: torch.Tensor,
        std_decay: float = 1.0,
    ) -> tuple:
        return self.forward_nonreasoning(state=state, std_decay=std_decay)
        
        
    def exploration_strategy_v0(self, flattened_logits: torch.Tensor):
        return torch.full(size=flattened_logits.shape, device=flattened_logits.device, fill_value=-torch.inf)

    def exploration_strategy_v1(self, flattened_logits: torch.Tensor, state: torch.Tensor, std_decay: float) -> torch.Tensor:
        logger.debug(f"std_decay: {std_decay}")

        # Calculate state-dependent log_std
        logger.debug(f"max logits: {torch.max(flattened_logits)}")
        log_std = self.calc_log_std1(flattened_logits)

        # Apply state-dependent action masking
        log_std = self.state_dependent_action_masking(state, log_std)

        logger.debug(f"max logits after masking: {torch.max(flattened_logits)}")

        # Clamp to valid range
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        return log_std
    
    
    def exploration_strategy_v2(
        self, 
        logits: torch.Tensor,  
        state: torch.Tensor, 
        target_path: torch.Tensor,
        std_decay,
    ):
        """
        Params:
        - state: [B, 3, H, W]
        - logits: [B, H, W]
        """

        # Identify obstacles in the unnormalized observation
        obstacle_map = torch.all(state == self.obstacle_color[None, :, None, None], axis=1)

        log_stds = set_neighborhood(
            predicted_path=target_path,
            radius=self.exploration_region_radius,
            inside_value=self.exploration_log_std * std_decay,  # std_decay starts from 1 
            outside_value=self.base_log_std * std_decay
        )

        # preds = F.tanh(logits) >= 0.0
        # collisions = torch.logical_and(preds, obstacle_map)

        log_stds[obstacle_map] = self.base_log_std * std_decay  # mask out obstacles again
        # log_stds[collisions] = self.exploration_log_std * std_decay  # mask in collisions

        # also give logits with value >= -1 an exploration bonus
        # log_stds[logits >= -1] = self.exploration_log_std * std_decay

        return log_stds

    def calc_log_std(self, action_output: torch.Tensor, std_decay: float) -> torch.Tensor:
        """
        Calculate log_std with decay (original version - currently unused).

        This is the original calc_log_std from PathFindingNetworkUNET2withSTD.
        Currently not used (calc_log_std1 is used instead), but kept for reference.

        Args:
            action_output: Actor output logits [B, num_actions]
            std_decay: Decay factor for exploration (0 to 1)

        Returns:
            log_std tensor [B, num_actions]

        Note:
            This method is NOT currently called. calc_log_std1() is used instead.
            Kept for backward compatibility and potential future use.
        """
        log_std = 2 + 0.1 * abs(action_output)

        # Set log_std for values >= -7, others to -20
        log_std = torch.where(
            (action_output >= -7),
            torch.tensor(std_decay - log_std),
            torch.tensor(-20.0)
        )

        return log_std.expand_as(action_output)

    def calc_log_std1(self, action_output: torch.Tensor) -> torch.Tensor:
        """
        Calculate state-dependent log_std (actively used version).

        This method creates exploration noise proportional to action magnitude:
        1. Clips extreme values to [-7, 4] range
        2. For clipped values [-4, 4]: log_std = 1 + 0.25 * |value|
        3. For extreme values: log_std = -20 (effectively zero exploration)

        Logic:
        - High magnitude actions (>4 or <-7): No exploration (log_std = -20)
        - Medium magnitude actions [-4, 4]: Proportional exploration
        - Encourages exploration where actor is less confident

        Args:
            action_output: Actor output logits [B, num_actions]

        Returns:
            log_std tensor [B, num_actions]

        Example:
            >>> logits = torch.tensor([[5.0, 2.0, -5.0, -8.0]])
            >>> log_std = wrapper.calc_log_std1(logits)
            >>> # logits[5.0 > 4] clipped to 4.0 → log_std = 1 + 0.25*4 = 2.0
            >>> # logits[2.0] → log_std = 1 + 0.25*2 = 1.5
            >>> # logits[-5.0] clipped to -4.0 → log_std = 1 + 0.25*4 = 2.0
            >>> # logits[-8.0 < -7] → log_std = -20.0 (no exploration)
        """
        # Clone to avoid in-place modification issues
        action_output = action_output.clone()

        # 1. Clip extreme positive values (>4) to 4
        mask_positive = action_output > 4
        action_output[mask_positive] = 4

        # 2. Clip extreme negative values (-7 < x < -4) to -4
        mask_negative = (action_output > -7) & (action_output < -4)
        action_output[mask_negative] = -4

        # 3. Calculate log_std: proportional to |action_output| for values in [-4, 4]
        #    Otherwise set to -20 (no exploration for extreme values)
        log_std = torch.where(
            (action_output >= -4) & (action_output <= 4),
            1 + 0.25 * abs(action_output),  # Proportional exploration
            torch.tensor(-20.0)  # No exploration for extreme values
        )

        return log_std.expand_as(action_output)

    def state_dependent_action_masking(
        self,
        state: torch.Tensor,
        log_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask actions based on state (e.g., obstacles, start, end positions).

        This method prevents the agent from selecting invalid actions:
        - Actions at obstacle positions → log_std = -20 (no probability)
        - Actions at start position → log_std = -20 (no probability)
        - Actions at end position → log_std = -20 (no probability)

        Implementation:
        1. Flatten state to [B, C, H*W]
        2. Find positions where any channel is non-zero
        3. Set log_std = -20 for those positions (mask them out)

        Args:
            state: Observation tensor [B, C, H, W]
            log_std: Current log_std tensor [B, num_actions]

        Returns:
            Masked log_std tensor [B, num_actions]

        Example:
            >>> state = torch.zeros(1, 3, 42, 42)
            >>> state[0, 0, 10, 10] = 1  # Obstacle at position (10, 10)
            >>> log_std = torch.ones(1, 1764)
            >>> masked_log_std = wrapper.state_dependent_action_masking(state, log_std)
            >>> # masked_log_std[0, 10*42 + 10] == -20.0 (action at obstacle masked)
        """
        # Flatten state: [B, C, H, W] → [B, C, H*W]
        flattened_states = state.view(state.size(0), state.size(1), -1)

        logger.debug(f"flattened states shape: {flattened_states.shape}")

        # Find positions where any channel is non-zero
        # mask[b, i] = 1 if any channel at position i is non-zero
        mask = (flattened_states != 0).any(dim=1)  # [B, H*W]
        mask = mask.float()

        # Set log_std = -20 where mask == 1 (invalid actions)
        log_std[mask == 1] = -20

        return log_std


def set_neighborhood(
    predicted_path: torch.Tensor,
    radius: int,
    inside_value: float,
    outside_value: float
) -> torch.Tensor:
    
    """
    Returns a tensor of shape predicted_path where every pixel with a distance of at most radius (chevyshev) is set to inside_value. Everything else is set to outside_value.

    Parameter:
    - predicted_path: [B, H, W], binary map where 1: path, 0: no path
    - radius: Desired radius
    - value: Desired value
    
    Returns:
    - Described output, shape: [B, H, W] (same as predicted_path)

    """

    # Use custom kernel such that each pixel with at least one path element gets the <value> >= <value> -> threshold back to <value> later
    kernel_hw = 2 * radius + 1  # for each center pixel include <radius> pixels in either direction, including diagonal
    kernel = torch.full(
        size=(1, 1, kernel_hw, kernel_hw), 
        fill_value=inside_value,
        device=predicted_path.device
    )

    # Apply 2d convolution with previously described kernel
    # proximity_map = map of 0: too far away from path, multiple of <value>: close enough
    proximity_map = F.conv2d(
        input=predicted_path.unsqueeze(1),  # pytorch requires a channel dim
        weight=kernel,
        padding=radius,  # such that input shape = result shape
        bias=None
    )

    # clip map to get map of 0: too far away from path, <value>: close enough   
    if inside_value > 0:
        proximity_map_clipped = torch.where(proximity_map >= inside_value, inside_value, outside_value)
    else:
        proximity_map_clipped = torch.where(proximity_map <= inside_value, inside_value, outside_value)

    return proximity_map_clipped.squeeze(1)
