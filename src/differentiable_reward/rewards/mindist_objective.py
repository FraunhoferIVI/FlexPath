import torch

from src.differentiable_reward.rewards.util import is_collision, compute_path_cost_approximation, soft_optimal_connectivity

import math


def reward_mindist(
    state: torch.Tensor, 
    predicted_path: torch.Tensor, 
    target_path: torch.Tensor,
    pixel_sum_penalty_scale: float = 1.0,
    obstacle_penalty_scaling: float = None,  # kept for signature compatibility with other reward functions
    collision_scale: float = 1.0,
    eps: float = 1e-8
):
    
    """

    Compute a differentiable reward that encourages a predicted path to be connected between a start
    and end marker and to avoid obstacles, while also penalizing deviations in path length (pixel sum)
    relative to a target path.
    The function uses color-based soft masks to locate start, end and obstacle pixels in an RGB
    state image, performs soft/differentiable checks for connectivity and collision (via helper
    functions is_connected and is_collision), and adds a pixel-sum penalty to discourage overly long
    or short predicted paths.
    Parameters
    ----------
    state : torch.Tensor
        RGB state image tensor with shape [B, 3, H, W]. Expected to contain colored start/end
        markers and obstacle pixels encoded as specific RGB colors. Device and dtype follow `state`.
    predicted_path : torch.Tensor
        Predicted path occupancy/probability map with shape [B, 1, H, W]. Values are expected to be
        in a continuous range (e.g. [0, 1]) so gradients can flow.
    target_path : torch.Tensor
        Ground-truth/target path occupancy map with shape [B, 1, H, W]. Used to compute a pixel-sum
        penalty that encourages matching path length.
    pixel_sum_penalty_scale : float, optional (default=1.0)
        Scaling factor applied to the pixel-sum penalty term. Larger values increase the importance
        of matching the target path pixel count.
    obstacle_penalty_scaling : float or None, optional
        Reserved/unused in the current implementation (kept for API compatibility). If used, would
        scale obstacle-related penalties.
    eps : float, optional (default=1e-8)
        Small constant to avoid division by zero when computing the pixel-sum penalty.
    Behavior and implementation notes
    ---------------------------------
    - Start, end and obstacle pixels are located by comparing `state` to predefined RGB colors:
        start ~ [1.0, 76/255, 76/255], end ~ [76/255, 1.0, 76/255], obstacle ~ [76/255, 76/255, 1.0].
        For each color the code builds a "soft" distance using sigmoid(abs(state - color)).sum(1, True),
        then thresholds that sum to produce a binary mask for downstream checks.
    - is_connected(predicted_path, start, end) is expected to return a per-batch connectivity score
        (soft/continuous in [0,1] if the helper is differentiable). It checks whether the predicted path
        connects the start and end masks.
    - is_collision(predicted_path, obstacle_map) is expected to return a per-batch collision score
        (soft/continuous in [0,1]), where larger values indicate more collision with obstacles.
    - Pixel-sum penalty: let P = sum(predicted_path) and T = sum(target_path) (summed over channel/XY).
        The implemented penalty is
            pixel_penalty = -clamp(1 - (T + eps) / (P + eps), 0, 1)
        so:
        - If P >= T, the inside term ∈ [0, 1] and pixel_penalty ∈ [-1, 0].
        - If P < T, the clamp yields 0 and pixel_penalty == 0 (no negative penalty).
        This encourages the predicted path not to be substantially longer than the target.
    - Returned reward combines connectivity and collision as
            connectivity * (1 - collision)
        and then adds the scaled pixel-sum penalty:
            reward = connectivity * (1 - collision) + pixel_sum_penalty_scale * pixel_penalty
        Typical ranges:
        - connectivity * (1 - collision) ∈ [0, 1] (soft)
        - pixel_penalty ∈ [-1, 0]
        - reward ∈ [-pixel_sum_penalty_scale, 1] (approximate)
    Returns
    -------
    tuple
        A pair (reward, diagnostics) where:
        - reward : torch.Tensor of shape [B] (or [B, ...] depending on helper outputs) containing the
            scalar reward per batch element computed as described above.
        - diagnostics : torch.Tensor with shape [3, B] (stacked per-batch diagnostics) containing:
            diagnostics[0, :] = connectivity scores (is_connected) flattened per batch
            diagnostics[1, :] = collision scores (is_collision) flattened per batch
            diagnostics[2, :] = pixel_penalty values flattened per batch
    Notes
    -----
    - The helper functions is_connected and is_collision are expected to operate on the provided
        masks and path tensor and to return batch-wise scalar scores. Their exact output shape must be
        compatible with the broadcast operations used here (they are flattened for diagnostics).
    - The thresholding step used to convert soft color distances to binary masks is not strictly
        differentiable; if full differentiability is required end-to-end, consider keeping the soft
        masks without hard thresholding or use a smooth approximation of the comparison.
    - Colors and thresholds are hard-coded; if your environment uses different markers, update the
        color constants accordingly.

    """
    
    # Use a soft color-based mask to find start and end points differentiably
    end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=state.device).view(3, 1, 1)
    start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=state.device).view(3, 1, 1)

    start = torch.sigmoid((state - start_color).abs()).sum(1, keepdim=True)  # Soft mask for start
    end = torch.sigmoid((state - end_color).abs()).sum(1, keepdim=True)  # Soft mask for end

    start = (start < 1.51).float()  # start will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies
    end = (end < 1.51).float()  # end will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

    # Obstacle map soft detection (differentiable obstacle detection)
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=state.device)  # BLUE
    obstacle_map = torch.sigmoid((state - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    obstacle_map = (obstacle_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

    # Using soft penalties for differentiable collision and connectivity checks
    _is_collision = is_collision(predicted_path, obstacle_map)
    _is_connected = (soft_optimal_connectivity(predicted_path, start=start, end=end, obstacles=obstacle_map) + soft_optimal_connectivity(predicted_path, start=end, end=start, obstacles=obstacle_map)) / 2.0

    # Compute pixel sum penalty
    pred_cost_map = compute_path_cost_approximation(
        path=predicted_path
    )

    target_cost_map = compute_path_cost_approximation(
        path=target_path
    )

    pred_cost = torch.sum(pred_cost_map, dim=(1, 2, 3))  # [B, 1]
    target_cost = torch.sum(target_cost_map, dim=(1, 2, 3))  # [B, 1]
    # target_cost = torch.zeros_like(pred_cost)  # for no supervision abl

    max_possible_cost = state.size(-1) * state.size(-2) * math.sqrt(2)
    pixel_penalty = -torch.abs(target_cost - pred_cost) / max_possible_cost
    return 0.005 * _is_connected + collision_scale * (1 - _is_collision) + pixel_sum_penalty_scale * pixel_penalty, torch.stack([_is_connected.view(-1), _is_collision.view(-1), pixel_penalty.view(-1)])


def reward_mindist_uniform(
    state: torch.Tensor, 
    predicted_path: torch.Tensor, 
    target_path: torch.Tensor,
    pixel_sum_penalty_scale: float = 1.0,
    obstacle_penalty_scaling: float = None,  # kept for signature compatibility with other reward functions
    collision_scale: float = 1.0,
    eps: float = 1e-8
):
    
    """

    Compute a differentiable reward that encourages a predicted path to be connected between a start
    and end marker and to avoid obstacles, while also penalizing deviations in path length (pixel sum)
    relative to a target path.
    The function uses color-based soft masks to locate start, end and obstacle pixels in an RGB
    state image, performs soft/differentiable checks for connectivity and collision (via helper
    functions is_connected and is_collision), and adds a pixel-sum penalty to discourage overly long
    or short predicted paths.
    Parameters
    ----------
    state : torch.Tensor
        RGB state image tensor with shape [B, 3, H, W]. Expected to contain colored start/end
        markers and obstacle pixels encoded as specific RGB colors. Device and dtype follow `state`.
    predicted_path : torch.Tensor
        Predicted path occupancy/probability map with shape [B, 1, H, W]. Values are expected to be
        in a continuous range (e.g. [0, 1]) so gradients can flow.
    target_path : torch.Tensor
        Ground-truth/target path occupancy map with shape [B, 1, H, W]. Used to compute a pixel-sum
        penalty that encourages matching path length.
    pixel_sum_penalty_scale : float, optional (default=1.0)
        Scaling factor applied to the pixel-sum penalty term. Larger values increase the importance
        of matching the target path pixel count.
    obstacle_penalty_scaling : float or None, optional
        Reserved/unused in the current implementation (kept for API compatibility). If used, would
        scale obstacle-related penalties.
    eps : float, optional (default=1e-8)
        Small constant to avoid division by zero when computing the pixel-sum penalty.
    Behavior and implementation notes
    ---------------------------------
    - Start, end and obstacle pixels are located by comparing `state` to predefined RGB colors:
        start ~ [1.0, 76/255, 76/255], end ~ [76/255, 1.0, 76/255], obstacle ~ [76/255, 76/255, 1.0].
        For each color the code builds a "soft" distance using sigmoid(abs(state - color)).sum(1, True),
        then thresholds that sum to produce a binary mask for downstream checks.
    - is_connected(predicted_path, start, end) is expected to return a per-batch connectivity score
        (soft/continuous in [0,1] if the helper is differentiable). It checks whether the predicted path
        connects the start and end masks.
    - is_collision(predicted_path, obstacle_map) is expected to return a per-batch collision score
        (soft/continuous in [0,1]), where larger values indicate more collision with obstacles.
    - Pixel-sum penalty: let P = sum(predicted_path) and T = sum(target_path) (summed over channel/XY).
        The implemented penalty is
            pixel_penalty = -clamp(1 - (T + eps) / (P + eps), 0, 1)
        so:
        - If P >= T, the inside term ∈ [0, 1] and pixel_penalty ∈ [-1, 0].
        - If P < T, the clamp yields 0 and pixel_penalty == 0 (no negative penalty).
        This encourages the predicted path not to be substantially longer than the target.
    - Returned reward combines connectivity and collision as
            connectivity * (1 - collision)
        and then adds the scaled pixel-sum penalty:
            reward = connectivity * (1 - collision) + pixel_sum_penalty_scale * pixel_penalty
        Typical ranges:
        - connectivity * (1 - collision) ∈ [0, 1] (soft)
        - pixel_penalty ∈ [-1, 0]
        - reward ∈ [-pixel_sum_penalty_scale, 1] (approximate)
    Returns
    -------
    tuple
        A pair (reward, diagnostics) where:
        - reward : torch.Tensor of shape [B] (or [B, ...] depending on helper outputs) containing the
            scalar reward per batch element computed as described above.
        - diagnostics : torch.Tensor with shape [3, B] (stacked per-batch diagnostics) containing:
            diagnostics[0, :] = connectivity scores (is_connected) flattened per batch
            diagnostics[1, :] = collision scores (is_collision) flattened per batch
            diagnostics[2, :] = pixel_penalty values flattened per batch
    Notes
    -----
    - The helper functions is_connected and is_collision are expected to operate on the provided
        masks and path tensor and to return batch-wise scalar scores. Their exact output shape must be
        compatible with the broadcast operations used here (they are flattened for diagnostics).
    - The thresholding step used to convert soft color distances to binary masks is not strictly
        differentiable; if full differentiability is required end-to-end, consider keeping the soft
        masks without hard thresholding or use a smooth approximation of the comparison.
    - Colors and thresholds are hard-coded; if your environment uses different markers, update the
        color constants accordingly.

    """
    
    # Use a soft color-based mask to find start and end points differentiably
    end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=state.device).view(3, 1, 1)
    start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=state.device).view(3, 1, 1)

    start = torch.sigmoid((state - start_color).abs()).sum(1, keepdim=True)  # Soft mask for start
    end = torch.sigmoid((state - end_color).abs()).sum(1, keepdim=True)  # Soft mask for end

    start = (start < 1.51).float()  # start will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies
    end = (end < 1.51).float()  # end will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

    # Obstacle map soft detection (differentiable obstacle detection)
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=state.device)  # BLUE
    obstacle_map = torch.sigmoid((state - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    obstacle_map = (obstacle_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

    # Using soft penalties for differentiable collision and connectivity checks
    _is_collision = is_collision(predicted_path, obstacle_map)
    _is_connected = (soft_optimal_connectivity(predicted_path, start=start, end=end, obstacles=obstacle_map, use_uniform_step_cost=True) + soft_optimal_connectivity(predicted_path, start=end, end=start, obstacles=obstacle_map, use_uniform_step_cost=True)) / 2.0

    # Compute pixel sum penalty
    pred_cost_map = compute_path_cost_approximation(
        path=predicted_path,
        use_uniform_step_cost=True
    )

    target_cost_map = compute_path_cost_approximation(
        path=target_path,
        use_uniform_step_cost=True
    )

    pred_cost = torch.sum(pred_cost_map, dim=(1, 2, 3))  # [B, 1]
    target_cost = torch.sum(target_cost_map, dim=(1, 2, 3))  # [B, 1]
    # target_cost = torch.zeros_like(pred_cost)  # for no supervision abl

    max_possible_cost = state.size(-1) * state.size(-2) * math.sqrt(2)
    pixel_penalty = -torch.abs(target_cost - pred_cost) / max_possible_cost
    return 0.02 * _is_connected + collision_scale * (1 - _is_collision) + pixel_sum_penalty_scale * pixel_penalty, torch.stack([_is_connected.view(-1), _is_collision.view(-1), pixel_penalty.view(-1)])
