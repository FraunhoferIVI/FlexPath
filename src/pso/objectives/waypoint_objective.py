import torch

from src.pso.objectives.util import is_collision, compute_path_cost_approximation, soft_optimal_connectivity

import math


def reward_waypoint_mindist(
    state: torch.Tensor, 
    predicted_path: torch.Tensor, 
    target_path: torch.Tensor,
    pixel_sum_penalty_scale: float = 1.0,
    obstacle_penalty_scaling: float = None,  # kept for signature compatibility with other reward functions
    collision_scale: float = 1.0,
    desired_obstacle_min_dist: float = 2,
    connectivity_scaling: float = 0.005,
    eps: float = 1e-8
):
    
    # Use a soft color-based mask to find start and end points differentiably
    end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=state.device).view(3, 1, 1)
    start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=state.device).view(3, 1, 1)
    waypoint_color = torch.tensor([255 / 255, 255 / 255, 76 / 255], device=state.device).view(3, 1, 1)

    start = torch.sigmoid((state - start_color).abs()).sum(1, keepdim=True)  # Soft mask for start
    end = torch.sigmoid((state - end_color).abs()).sum(1, keepdim=True)  # Soft mask for end
    waypoint = torch.sigmoid((state - waypoint_color).abs()).sum(1, keepdim=True)  # Soft mask for end

    start = (start < 1.51).float()  # start will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies
    end = (end < 1.51).float()  # end will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies
    waypoint = (waypoint < 1.51).float()

    # Obstacle map soft detection (differentiable obstacle detection)
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=state.device)  # BLUE
    obstacle_map = torch.sigmoid((state - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    obstacle_map = (obstacle_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

    # Using soft penalties for differentiable collision and connectivity checks
    _is_collision = is_collision(predicted_path, obstacle_map)
    _is_connected = (soft_optimal_connectivity(predicted_path, start=start, end=waypoint, obstacles=obstacle_map) + soft_optimal_connectivity(predicted_path, start=waypoint, end=start, obstacles=obstacle_map) + soft_optimal_connectivity(predicted_path, start=waypoint, end=end, obstacles=obstacle_map) + soft_optimal_connectivity(predicted_path, start=end, end=waypoint, obstacles=obstacle_map)) / 4.0

    pred_cost_map = compute_path_cost_approximation(
        path=predicted_path
    )

    target_cost_map = compute_path_cost_approximation(
        path=target_path
    )

    pred_cost = torch.sum(pred_cost_map, dim=(1, 2, 3))  # [B, 1]
    target_cost = torch.sum(target_cost_map, dim=(1, 2, 3))  # [B, 1]

    max_possible_cost = state.size(-1) * state.size(-2) * math.sqrt(2)
    pixel_penalty = -torch.abs(target_cost - pred_cost) / max_possible_cost

    return connectivity_scaling * _is_connected + collision_scale * (1 - _is_collision) + pixel_sum_penalty_scale * pixel_penalty, torch.stack([_is_connected.view(-1), _is_collision.view(-1), pixel_penalty.view(-1)])
