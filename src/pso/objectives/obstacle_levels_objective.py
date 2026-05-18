import torch

from src.pso.objectives.util import is_connected, is_collision, compute_soft_obstacle_distances, compute_path_cost_approximation

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
