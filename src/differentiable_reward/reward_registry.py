from src.differentiable_reward.rewards.mindist_objective import reward_mindist, reward_mindist_uniform
from src.differentiable_reward.rewards.obstacle_objective import reward_obstacle_with_cost_penalty
from src.differentiable_reward.rewards.obstacle_levels_objective import reward_obstacle_levels_with_cost_penalty
from src.differentiable_reward.rewards.waypoint_objective import reward_waypoint, reward_waypoint_mindist


"""Centralized section for mapping the names of all available differentiable reward functions to its method. """


AVAILABLE_REWARD_FUNCTIONS = {
    "obstacle_v2": reward_obstacle_with_cost_penalty,
    "mindist": reward_mindist,
    "mindist_uniform": reward_mindist_uniform,
    "obstacle_levels": reward_obstacle_levels_with_cost_penalty,
    "reward_waypoint": reward_waypoint,
    "reward_waypoint_mindist": reward_waypoint_mindist,
}


"""Centralized section for mapping the returned reward components of a differentiable reward function to its names. """


REWARD_COMPONENTS = {
    "obstacle_v2": [
        "_is_connected",
        "_is_collision",
        "pixel_cost_penalty",
        "obstacle_penalty",
        "obstacle_penalty_mean",
        "obstacle_penalty_max"
    ],
    "mindist": [
        "_is_connected", 
        "_is_collision", 
        "pixel_penalty"
    ],
    "mindist_uniform": [
        "_is_connected", 
        "_is_collision", 
        "pixel_penalty"
    ],
    "reward_waypoint_mindist": [
        "_is_connected", 
        "_is_collision", 
        "pixel_penalty"
    ],
    "obstacle_levels": [
        "_is_connected",
        "_is_collision",
        "pixel_cost_penalty",
        "obstacle_penalty",
        "obstacle_penalty_mean",
        "obstacle_penalty_max",
        "obstacle_penalty_base",
        "obstacle_penalty_lv4"
    ],
    "reward_waypoint": [
        "_is_connected", 
        "_is_collision", 
        "pixel_penalty",
        "obstacle_penalty",
        "obstacle_penalty_mean",
        "obstacle_penalty_max"
    ]
}