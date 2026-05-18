import torch

from src.pso.pso_registry import AVAILABLE_REWARD_FUNCTIONS

from typing import Union, Tuple, Set, Callable

import logging

logger = logging.getLogger(__name__)


class PSOObjective():

    """

    Wrapper for selecting and calling a differentiable reward function.

    Attributes
    - f: Callable
        Selected reward function (either from AVAILABLE_REWARD_FUNCTIONS or a user-provided callable).
    - current_step: int
        Counter tracking how many times forward() has been called (used for schedules).
    - pixel_sum_penalty_schedule: Optional[torch.Tensor]
        Per-step schedule used to scale pixel-sum related penalties.
    - obstacle_penalty_scaling: float
        Scaling factor applied to obstacle-related penalties.
    - (internal) compile_reward_f: bool
        Indicates whether the selected reward function was compiled via torch.compile.

    Purpose
    - Encapsulates selection, optional JIT compilation, and consistent calling semantics
      for differentiable reward functions used during training/evaluation.

    """
    
    def __init__(
        self, 
        reward_f: Union[str, Callable], 
        pixel_sum_penalty_schedule: torch.Tensor = None,
        obstacle_penalty_scaling: float = 1.0,
        compile_reward_f: bool = True
    ):
        
        """

        Initialize the PSOObjective.

        Parameters
        - reward_f: Union[str, Callable]
          Either the name of a reward function registered in AVAILABLE_REWARD_FUNCTIONS
          (e.g. 'obstacle', 'mindist') or a callable with the reward signature expected
          by the codebase.
        - pixel_sum_penalty_schedule: Optional[torch.Tensor]
          A 1D tensor containing per-step scaling factors for any pixel-sum penalty terms.
          If None, a default behavior is used (no schedule-based scaling).
        - obstacle_penalty_scaling: float
          Scalar multiplier applied to obstacle penalties passed into the underlying reward function.
        - compile_reward_f: bool
          If True, attempt to compile the selected reward function via torch.compile
          for potential speed improvements.

        Behavior
        - Resolves reward_f into self.f (callable). Logs errors for unknown string keys.
        - If compile_reward_f is True, wraps the function with torch.compile.
        - Initializes internal state such as current_step and stored schedules.

        """
        
        super().__init__()

        self.current_step: int = 0
        self.pixel_sum_penalty_schedule = pixel_sum_penalty_schedule
        self.obstacle_penalty_scaling = obstacle_penalty_scaling

        if isinstance(reward_f, str):
            if reward_f in AVAILABLE_REWARD_FUNCTIONS.keys():
                self.f = AVAILABLE_REWARD_FUNCTIONS[reward_f]
                logger.info(f"Successfully initalized differentiable reward function: {reward_f}.")

            else:
                logger.error(f"Error, unknown reward function: {reward_f}.")

        elif callable(reward_f):
            self.f = reward_f
            logger.info(f"Successfully initalized custom reward function: {str(reward_f)}.")

        else:
            logger.error(f"Error, unsupported type for agument reward_f: {type(reward_f)}. Only available strings and functions are supported.")
            
        # compile on demand
        if compile_reward_f:
            self.f = torch.compile(self.f)
            logger.info(f"Reward function is marked for compilation.")
    
    def get_available_reward_functions(self) -> Set:

        """

        Returns
        - Set[str]: the keys of AVAILABLE_REWARD_FUNCTIONS (names of builtin reward functions).

        Purpose
        - Convenience helper to expose which registered reward functions can be selected
          by passing a string to the constructor.

        """

        
        return AVAILABLE_REWARD_FUNCTIONS.keys()

    def forward(
        self,
        state: torch.Tensor, 
        predicted_path: torch.Tensor, 
        target_path: torch.Tensor,
        eval: bool = False, 
        eps: float = 1e-8,
        t: int = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        """

        Call the configured reward function and return its scalar reward and components.

        Parameters
        - state: torch.Tensor
          Environment/state tensor expected by the underlying reward function.
        - predicted_path: torch.Tensor
          Model-predicted path heatmap tensor (shape convention used elsewhere in the codebase).
        - target_path: torch.Tensor
          Ground-truth or target path tensor used by the reward function.
        - eval: bool
          If True, the internal current_step counter will not be incremented.
        - eps: float
          Small epsilon forwarded to the underlying reward function to avoid numerical issues.

        Returns
        - (rewards, components): Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
          The primary reward tensor followed by a tuple of auxiliary reward components
          as produced by the underlying reward function.

        Behavior / details
        - If a pixel_sum_penalty_schedule was provided at construction, this method selects
          the per-step scale according to self.current_step and passes it as
          pixel_sum_penalty_scale to the underlying reward function.
        - If no schedule is present, the base reward function is called with obstacle scaling
          and eps only.
        - After a non-eval call, increments self.current_step to advance schedules.

        """
        
        # use default scaling if no pixel sum penalty schedule given
        if self.pixel_sum_penalty_schedule is None:
            rewards = self.f(
                state, 
                predicted_path,
                target_path,
                obstacle_penalty_scaling=self.obstacle_penalty_scaling,
                eps=eps
            )
        else:
            pixel_sum_penalty_scale = self.pixel_sum_penalty_schedule[self.current_step]
            rewards =  self.f(
                state, 
                predicted_path,
                target_path,
                pixel_sum_penalty_scale=pixel_sum_penalty_scale,
                obstacle_penalty_scaling=self.obstacle_penalty_scaling,
                eps=eps
            )
        
        if not eval:
            self.current_step += 1

        return rewards[0], rewards[1].detach()
    
    def __call__(
        self,
        state: torch.Tensor, 
        predicted_path: torch.Tensor, 
        target_path: torch.Tensor,
        eval: bool = False, 
        eps: float = 1e-8,
        t: int = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Shortcut to forward(...); timestep parameter is ignored for uniform API."""

        return self.forward(
            state=state,
            predicted_path=predicted_path,
            target_path=target_path,
            eval=eval,
            eps=eps
        )
    