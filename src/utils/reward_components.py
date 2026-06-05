import numpy as np


REWARD_COMPONENTS_MAP = [
    "is_valid_path", 
    "pixel_penalty"
]
# REWARD_COMPONENTS_MAP = [
#     "is_valid_path", 
#     "obstacle_penalty_mean", 
#     "sum_pixels_in_proximity", 
#     "sum_pixels_outside_proximity", 
#     "obstacle_penalty_max", 
#     "pixel_sum_penalty",
#     "pixel_sum_dif",
#     "pred_pixel_sum",
#     "deviation_penalty_mean",  
#     "deviation_penalty_max", 
#     "sum_pixels_in_trust_region", 
#     "sum_pixels_outside_trust_region",
#     "old_reward",
#     "pixels_connected",
#     "pixels_unconnected",
#     "connectivity_penalty"
# ]

# REWARD_COMPONENTS_MAP = [
#     "is_valid_path", 
#     "deviation_penalty_mean", 
#     "deviation_penalty_max", 
#     "sum_pixels_in_trust_region", 
#     "sum_pixels_outside_trust_region", 
#     "deviation_penalty_mean2", 
#     "deviation_penalty_max2", 
#     "sum_pixels_in_trust_region2", 
#     "sum_pixels_outside_trust_region2", 
#     "deviation_penalty_mean1",  
#     "deviation_penalty_max1", 
#     "sum_pixels_in_trust_region1", 
#     "sum_pixels_outside_trust_region1", 
#     "obstacle_penalty_mean", 
#     "obstacle_penalty_max", 
#     "sum_pixels_in_proximity", 
#     "sum_pixels_outside_proximity", 
#     "obstacle_penalty_mean3", 
#     "obstacle_penalty_max3", 
#     "sum_pixels_in_proximity3", 
#     "sum_pixels_outside_proximity3", 
#     "obstacle_penalty_mean1", 
#     "obstacle_penalty_max1", 
#     "sum_pixels_in_proximity1", 
#     "sum_pixels_outside_proximity1"
# ]

NUM_REWARD_COMPONENTS = len(REWARD_COMPONENTS_MAP)

def get_empty_list() -> list:
    return [[] for _ in range(NUM_REWARD_COMPONENTS)]

def add_reward_components_to_tracker_list(tracker_list: list, components: tuple) -> list:
    for i in range(NUM_REWARD_COMPONENTS):
        tracker_list[i].append(components[i])

    return tracker_list

def write_to_tensorboard(logger, episode_reward_components, eval=True):
    if eval:
        for i in range(NUM_REWARD_COMPONENTS):
            logger.record(f"eval/{REWARD_COMPONENTS_MAP[i]}", np.mean(episode_reward_components[i]))

    else:
        for i in range(NUM_REWARD_COMPONENTS):
            logger.record(f"train/{REWARD_COMPONENTS_MAP[i]}", np.mean(episode_reward_components[i]))