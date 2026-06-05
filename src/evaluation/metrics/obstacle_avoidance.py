import numpy as np

from scipy.spatial.distance import cdist


def compute_obstacle_avoidance(
    predicted_path: np.ndarray,
    obstacle_grid: np.ndarray,
    desired_obstacle_distance: float
):
    
    """

    Parameters:
    - predicted path shape=[H, W], values=[0, 1], not thresholded!
    - obstacle grid shape=[H, W], values={0, 1}
    - desired obstacle distance
    Returns:
    - avg. closest distance to obstacles
    - full obsstacle avoidance (if no obstacle is closer than wished)
    
    """

    predicted_path = None


DISTANCE_MAP = {}  # cache for map of distances from each HxW to each HxW point


def compute_distance_map(
    shape: tuple,
):
    """
    
    Computes the distances from each HxW to each HxW point

    Parameters:
    - shape: 2d shape to calculate distances for

    Returns:
    - distance tensor: [shape[0], shape[1], shape[0], shape[1]] 

    """

    assert len(shape) == 2, "compute_distance_map() only supports 2d shapes"

    H, W = shape

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.column_stack([ys.ravel(), xs.ravel()]).astype(np.float32)

    dist = cdist(coords, coords, metric="euclidean")

    return dist.reshape(H, W, H, W)

    
def compute_obstacle_avoidance_metrics_from_thresholded_path(
    predicted_path: np.ndarray,
    obstacle_grid: np.ndarray,
    desired_obstacle_distance: float
):
    """

    Parameters:
    - predicted path shape=[H, W], values={0, 1}, thresholded!
    - obstacle grid shape=[H, W], values={0, 1}
    - desired obstacle distance
    Returns:
    - avg. closest distance to obstacles
    - desired distance avoidance rate (ratio of predictions that are at least desired_obstacle_distance away from obstacles)
    - full obsstacle avoidance (if no obstacle is closer than wished)
    
    """

    shape = predicted_path.shape

    if shape not in DISTANCE_MAP.keys():
        distance_map = compute_distance_map(shape)
        DISTANCE_MAP[shape] = distance_map
    else:
        distance_map = DISTANCE_MAP[shape]

    obstacle_coords = np.argwhere(obstacle_grid)  # [N, 2]
    path_coords = np.argwhere(predicted_path)  # [N, 2]

    n_path_coords = len(path_coords)

    py = path_coords[:, 0][:, None]        # (P, 1)
    px = path_coords[:, 1][:, None]        # (P, 1)

    oy = obstacle_coords[:, 0][None, :]    # (1, O)
    ox = obstacle_coords[:, 1][None, :]    # (1, O)

    distances = distance_map[py, px, oy, ox]   # (P, O)
    closest_distances = distances.min(axis=1)  # (P,)

    avg_obst_dist = np.mean(closest_distances)
    avoided_obstacles = (closest_distances >= desired_obstacle_distance).sum()
    avg_obstacle_avoidance_ratio = avoided_obstacles / n_path_coords

    is_full_obstacle_avoidance = avoided_obstacles == n_path_coords

    return (
        avg_obst_dist,
        avg_obstacle_avoidance_ratio,
        is_full_obstacle_avoidance
    )
