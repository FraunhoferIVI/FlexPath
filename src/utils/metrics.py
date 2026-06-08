import numpy as np

from cstar.pathfinding import run_astar_2D


def compute_path_optimality(predicted_path_occupancy_map: np.ndarray, state: np.ndarray, diagonal_movements_at_obstacle: bool = False) -> tuple:
    """
    Takes the (thresholded) path prediction map and computes: 
    1. collision (whether a path prediction cuts an obstacle)
    2. path validity (connects start and goal position without holes and without hitting obstacles)
    3. path optimality (predicted path length equals length of an optimal path)
    4. path optimality factor: length(predicted_path) / length(optimal_path)
    5. Expansion ratio: visited_nodes(predicted) / visited_nodes(label)
    
    Params:
    - predicted_path_occupancy_map: [H, W] map of zeros (no path) and ones (path)
    - state: [H, W, 3] The current state of the map (without paths)
    - diagonal_movements_at_obstacle: Whether to allow diagonal movements next to an obstacle

    Returns:
    1. collision: bool
    2. path validity: bool
    3. path optimality: bool
    4. path optimality factor: length(predicted_path) / length(optimal_path): float
    5. Expansion ratio: visited_nodes(predicted) / visited_nodes(label)
    """

    # 1. check for collisions
    # count collions
    obstacle_occupancy_map = (state == np.array([76, 76, 255])[None, None, :]).all(axis=-1)
    collisions = np.sum(np.logical_and(predicted_path_occupancy_map, obstacle_occupancy_map))

    if collisions > 0:
        # obstacles hit -> invalid path -> non-optimal path
        return True, False, False, None, None

    # run a* (heuristic: octile distance) on prediction

    # get start and goal position coordinates
    start_coordinates = np.argwhere((state == np.array([255, 76, 76])[None, None, :]).all(axis=-1))
    goal_coordinates = np.argwhere((state == np.array([76, 255, 76])[None, None, :]).all(axis=-1))

    assert start_coordinates.shape[0] == 1 and goal_coordinates.shape[0] == 1, "Error: More/Less than one starting point/goal found"

    # pick first
    start_coordinates = start_coordinates[0]
    goal_coordinates = goal_coordinates[0]

    # make start and goal position walkable
    predicted_path_occupancy_map[tuple(start_coordinates)] = True
    predicted_path_occupancy_map[tuple(goal_coordinates)] = True

    # run a* on top of prediction
    pred_waypoint_coordinates, pred_cost, pred_runs, _ = run_astar_2D(predicted_path_occupancy_map, start_coordinates[0], start_coordinates[1], goal_coordinates[0], goal_coordinates[1])
    
    # run a* on whole map
    label_waypoint_coordinates, label_cost, label_runs, _ = run_astar_2D(~obstacle_occupancy_map, start_coordinates[0], start_coordinates[1], goal_coordinates[0], goal_coordinates[1])

    path_found = len(pred_waypoint_coordinates) > 0  # if no path was found then waypoints = []

    if  not path_found:
        # invalid path
        return False, False, False, None, None
    else:
        # calc expansion ratio
        expansion_ratio = pred_runs / label_runs

        # valid path
        if pred_cost == label_cost:
            # optimal path
            return False, True, True, 1.0, expansion_ratio
        else:
            # suboptimal path
            optimality = pred_cost / label_cost
            return False, True, False, optimality, expansion_ratio


def compute_bins(var: np.ndarray, bin_width: float) -> tuple:
    """
    Takes a tanh-squashed input (range: [-1, 1]) and computes bins with the corresponding width.

    Params:
    - var: np.ndarray 
    - bin_width: Width of the bins: float

    Returns:
    1. List of number of elements in corresponding bins
    2. np.ndarray: Linspace of bin bounds
    """

    num_bin_elements = list()  # list that stores number of elements for each respective bin
    num_bins = int(2 // bin_width)  # number of total bins

    bin_bounds = np.linspace(-1, 1, num=num_bins)

    for i in range(len(bin_bounds) - 1):
        thresholded_map = np.logical_and(var > bin_bounds[i], var <= bin_bounds[i+1])
        num_bin_elements.append(np.sum(thresholded_map))

    return np.array(num_bin_elements), np.array(bin_bounds)


def compute_bins_clipped(var: np.ndarray, bin_width: float, lower_bound: int, upper_bound: int) -> tuple:
    """
    Takes an unbounded variable, clips it into the desired range and computes bins.

    Params:
    - var: np.ndarray 
    - bin_width: Width of the bins: float

    Returns:
    1. List of number of elements in corresponding bins
    2. np.ndarray: Linspace of bin bounds
    """

    num_bin_elements = list()  # list that stores number of elements for each respective bin
    num_bins = int((upper_bound - lower_bound) // bin_width)  # number of total bins

    bin_bounds = np.linspace(lower_bound, upper_bound, num=num_bins)

    for i in range(len(bin_bounds) - 1):
        thresholded_map = np.logical_and(var > bin_bounds[i], var <= bin_bounds[i+1])
        num_bin_elements.append(np.sum(thresholded_map))

    return np.array(num_bin_elements), np.array(bin_bounds)
