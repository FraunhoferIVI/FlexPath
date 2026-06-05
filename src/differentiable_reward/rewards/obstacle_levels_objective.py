import torch

from src.differentiable_reward.rewards.util import is_connected, is_collision, compute_soft_obstacle_distances, compute_path_cost_approximation

import math


def reward_obstacle_levels_with_cost_penalty(
        state: torch.Tensor, 
        predicted_path: torch.Tensor, 
        target_path: torch.Tensor,
        allowed_distance: float = 3.0,
        desired_obstacle_min_dist: float = 2.0,
        distance_dif_clip: float = 10.0,
        pixel_threshold: float = 1.2,
        pixel_sum_penalty_scale: float = 0.0,
        obstacle_penalty_scaling: float = 1.0,
        eps: float = 1e-8
):  
    
    """

    Compute a composite, differentiable reward for a predicted path that penalizes collisions,
    insufficient connectivity between start and goal, proximity to obstacles, and cost deviation
    relative to a target path.

    This function uses soft (differentiable) color-based masks to detect start, end and obstacle
    regions in the input state image, then combines several signal terms computed by helper
    functions into a single scalar reward per batch element plus a diagnostic vector.

    Parameters
    ----------
    state : torch.Tensor
        Batched observation image tensor containing start, end and obstacle markers by color.
        Expected shape: (B, 3, H, W). Values are on the same scale as the color constants
        used in the function (typically [0, 1] float). Computations follow the device of this
        tensor.

    predicted_path : torch.Tensor
        Batched predicted path representation. Heatmap of values between 0 and 1. Shape: [B, 1, H, W]

    target_path : torch.Tensor
        Batched reference/target path in the same representation as predicted_path. Used to
        compute cost similarity/penalty.

    allowed_distance : float, optional (default=3.0)
        Threshold parameter used by path deviation routines (kept for compatibility; the
        current implementation may not use this directly).

    desired_obstacle_min_dist : float, optional (default=2.0)
        Desired minimum soft distance from obstacles passed to compute_soft_obstacle_distances.
        Larger values encourage paths that keep further away from obstacle regions.

    distance_dif_clip : float, optional (default=10.0)
        Clipping value for distance difference penalties (kept for compatibility; may be used
        by path deviation helper functions).

    pixel_threshold : float, optional (default=1.2)
        Threshold used in color/soft-mask logic (kept for compatibility; masks are computed
        using sigmoids and a numeric margin around 1.5).

    pixel_sum_penalty_scale : float, optional (default=0.0)
        Scale factor applied to the pixel-sum / cost-deviation penalty term. If zero,
        the pixel-sum penalty is ignored.

    obstacle_penalty_scaling : float, optional (default=1.0)
        Scale factor for the soft obstacle proximity penalty. Obstacle penalties returned by
        compute_soft_obstacle_distances are expected in the negative range (e.g. [-1, 0]),
        so increasing this scale increases the negative contribution (stronger penalty).

    eps : float, optional (default=1e-8)
        Small numerical epsilon used to stabilize relative difference computations.

    Returns
    -------
    reward : torch.Tensor
        Batched scalar reward combining:
            - connectivity score (soft) multiplied by (1 - soft collision indicator),
            - scaled soft obstacle proximity penalty,
            - scaled pixel/cost-similarity penalty.
        Reward shape: matches the internal connectivity/collision outputs (commonly (B, 1) or (B,)).
        Typical ranges for sub-terms:
            - connectivity term: [0, 1]
            - collision term: [0, 1] (used as a negation factor)
            - obstacle_penalty: [-1, 0]
            - pixel_sum_penalty: [-1, 0]
        The aggregated reward value will therefore normally lie approximately within a range
        determined by these components and their scaling factors.

    diagnostics : torch.Tensor
        A stacked diagnostic vector with shape (B, 6) (flattened per-batch entries), containing
        the following elements in order for each batch element:
            0) is_connected       - soft connectivity score between start and goal (float)
            1) is_collision       - soft collision indicator (float; close to 1 indicates collision)
            2) pixel_sum_penalty  - negative clamped relative cost difference between predicted and
                                    target path costs (in [-1, 0])
            3) obstacle_penalty   - averaged soft obstacle proximity penalty (in [-1, 0])
            4) obstacle_penalty_mean - mean proximity penalty from compute_soft_obstacle_distances
            5) obstacle_penalty_max  - max proximity penalty from compute_soft_obstacle_distances

    Notes
    -----
    - Start, end and obstacles are detected with soft color-matching masks derived from
        sigmoid(distance-to-color) maps. This makes the detection differentiable but slightly
        tolerant to color noise.
    - Connectivity and collision checks are delegated to is_connected and is_collision helper
        functions and are expected to be differentiable (soft) implementations.
    - Soft obstacle distance penalties are computed by compute_soft_obstacle_distances and are
        expected to yield negative penalties when the path is too close to obstacles.
    - Path cost approximations (used to compute the pixel_sum_penalty) come from
        compute_path_cost_approximation. The pixel_sum_penalty is a negative clamped relative
        absolute error between predicted and target path costs; it is scaled by
        pixel_sum_penalty_scale before adding to the final reward.
    - All operations preserve device placement of the input state tensor; ensure other inputs
        and helper functions operate on the same device and with matching batch sizes.

    Examples
    --------
    - Ensure predicted_path and target_path are in the representation expected by the
        helper utilities. Call as:
        reward, diag = reward_obstacle_with_cost_penalty(state, predicted_path, target_path)
        where reward and diag are tensors batched over the leading dimension.

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

    obstacle_color_lv4 = torch.tensor([100 / 255, 100 / 255, 255 / 255], device=state.device)  # BLUE
    obstacle_map_lv4 = torch.sigmoid((state - obstacle_color_lv4.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    obstacle_map_lv4 = (obstacle_map_lv4 < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

    # Using soft penalties for differentiable collision and connectivity checks
    _is_collision = (is_collision(predicted_path, obstacle_map) + is_collision(predicted_path, obstacle_map_lv4)) / 2
    _is_connected = (is_connected(predicted_path, start=start, end=end) + is_connected(predicted_path, start=end, end=start)) / 2  # avg connectivity from both sides seperately

    # Compute obstacle proximity penalties
    obstacle_penalty_mean, obstacle_penalty_max = compute_soft_obstacle_distances(
        path_pred=predicted_path, 
        obstacle_grid=obstacle_map,
        desired_min_dist=desired_obstacle_min_dist
    )
    obstacle_penalty_base = (obstacle_penalty_mean + obstacle_penalty_max) / 2  # [-1, 0]

    obstacle_penalty_mean_lv4, obstacle_penalty_max_lv4 = compute_soft_obstacle_distances(
        path_pred=predicted_path, 
        obstacle_grid=obstacle_map_lv4,
        desired_min_dist=4
    )
    obstacle_penalty_lv4 = (obstacle_penalty_mean_lv4 + obstacle_penalty_max_lv4) / 2  # [-1, 0]

    obstacle_penalty = (obstacle_penalty_base + obstacle_penalty_lv4) / 2

    pred_cost_map = compute_path_cost_approximation(
        path=predicted_path
    )

    target_cost_map = compute_path_cost_approximation(
        path=target_path
    )

    pred_cost = torch.sum(pred_cost_map, dim=(1, 2, 3))  # [B, 1]
    target_cost = torch.sum(target_cost_map, dim=(1, 2, 3))  # [B, 1]

    max_possible_cost = state.size(-1) * state.size(-2) * math.sqrt(2)
    pixel_sum_penalty = -torch.abs(target_cost - pred_cost) / max_possible_cost

    final_reward = _is_connected * (1 - _is_collision.view(-1)) + obstacle_penalty_scaling * obstacle_penalty.view(-1) + pixel_sum_penalty_scale * pixel_sum_penalty

    return final_reward, torch.stack(
        [
            _is_connected.view(-1),
            _is_collision.view(-1),
            pixel_sum_penalty.view(-1),
            obstacle_penalty.view(-1),
            obstacle_penalty_mean.view(-1),
            obstacle_penalty_max.view(-1),
            obstacle_penalty_base.view(-1),
            obstacle_penalty_lv4.view(-1)
        ]
    )  # flatten view as some tensors have [B, 1] shape and some [B, ], make all -> [B, ]
