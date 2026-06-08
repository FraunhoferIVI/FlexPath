import numpy as np
import os

from scipy.ndimage import distance_transform_edt
from cstar.pathfinding import run_astar_2D
from tqdm import tqdm


standard_obstacle = [76, 76, 255]
waypoint = [255, 255, 76]  # unused placeholder; retained for compatibility
badobstacle = [100, 100, 255]

START_COLOR = [255, 76, 76]
END_COLOR = [76, 255, 76]

additional_obstacles: list = [
    (4, 0.1, (100, 100, 255))
]  # desired obstacle distance, edge_prob, color_encoding


def sample_obstacles(path_to_ds: str, save_path: str, proximity: int = 10, num_obstacles: int = 10):
    """
    For each image in the dataset, place up to `num_obstacles` new obstacle pixels near the optimal path.

    Args:
        path_to_ds: input dataset path (.npz) containing keys: image, label, rgb_label, path_label, path_rgb_label
        save_path: output path for the modified dataset (.npz). Directories will be created if needed.
        proximity: how far (8-connected steps) away from the path an obstacle may be placed.
        num_obstacles: maximum number of obstacle pixels to add per image (no replacement).
    """

    arr = np.load(path_to_ds, allow_pickle=False)
    # Expect keys: image, label, rgb_label, path_label, path_rgb_label
    if 'image' not in arr:
        raise KeyError("Input dataset must contain key 'image'.")
    if 'path_label' not in arr:
        raise KeyError("Input dataset must contain key 'path_label'.")

    images = arr['image']
    path_labels = arr['path_label']

    # Validate shapes
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected images with shape (N, H, W, 3).")
    if not (path_labels.ndim == 3 and path_labels.shape == images.shape[:3]):
        raise ValueError("path_label shape must align with images (N, H, W).")

    obstacle_color = np.array(standard_obstacle, dtype=images.dtype)
    badobstacle_color = np.array(badobstacle, dtype=images.dtype)

    def octile_distance(mask: np.ndarray) -> np.ndarray:
        """Compute octile distance (8-connected, costs 1/√2) using EDT nearest indices (vectorized)."""
        inv = ~mask
        dist_euclid, nearest = distance_transform_edt(inv, return_indices=True)
        coords = np.indices(mask.shape)
        dy = np.abs(coords[0] - nearest[0])
        dx = np.abs(coords[1] - nearest[1])
        mx = np.maximum(dx, dy)
        mn = np.minimum(dx, dy)
        return mx + (np.sqrt(2.0) - 1.0) * mn

    modified_images = images.copy()
    rng = np.random.default_rng()

    for idx in tqdm(range(modified_images.shape[0])):
        img = modified_images[idx]
        path_mask = path_labels[idx].astype(bool)

        # Current obstacle mask
        obstacle_mask = (
            (img[:, :, 0] == obstacle_color[0]) &
            (img[:, :, 1] == obstacle_color[1]) &
            (img[:, :, 2] == obstacle_color[2])
        )

        # Candidates: non-obstacle and non-path pixels within proximity (octile distance) of path
        dist_map = octile_distance(path_mask)
        proximity_mask = dist_map <= float(proximity)
        candidate_mask = proximity_mask & (~path_mask) & (~obstacle_mask)

        candidate_indices = np.argwhere(candidate_mask)
        if candidate_indices.size == 0:
            # No available spot; skip this image
            continue

        # Pick up to num_obstacles unique candidates
        k = min(num_obstacles, candidate_indices.shape[0])
        choice_idx = rng.choice(candidate_indices.shape[0], size=k, replace=False)
        chosen = candidate_indices[choice_idx]

        a = k // 2
        ys, xs = chosen[:a, 0], chosen[:a, 1]
        img[ys, xs] = badobstacle_color

        ys, xs = chosen[a:, 0], chosen[a:, 1]
        img[ys, xs] = obstacle_color

        modified_images[idx] = img

    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save new copy of the dataset with updated images, other arrays unchanged
    np.savez_compressed(
        save_path,
        image=modified_images,
        label=arr['label'] if 'label' in arr else None,
        rgb_label=arr['rgb_label'] if 'rgb_label' in arr else None,
        path_label=arr['path_label'] if 'path_label' in arr else None,
        path_rgb_label=arr['path_rgb_label'] if 'path_rgb_label' in arr else None,
    )


def sample_waypoints(path_to_ds: str, save_path: str, proximity: int = 20, timeout: int = 10):
    """
    For each image in the dataset, place a single waypoint pixel near the optimal path.

    Args:
        path_to_ds: input dataset path (.npz) containing keys: image, label, rgb_label, path_label, path_rgb_label
        save_path: output path for the modified dataset (.npz). Directories will be created if needed.
        proximity: how far (8-connected steps) away from the path a waypoint may be placed.
    """

    arr = np.load(path_to_ds, allow_pickle=False)
    # Expect keys: image, label, rgb_label, path_label, path_rgb_label
    if 'image' not in arr:
        raise KeyError("Input dataset must contain key 'image'.")
    if 'path_label' not in arr:
        raise KeyError("Input dataset must contain key 'path_label'.")

    images = arr['image']
    path_labels = arr['path_label']

    # Validate shapes
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected images with shape (N, H, W, 3).")
    if not (path_labels.ndim == 3 and path_labels.shape == images.shape[:3]):
        raise ValueError("path_label shape must align with images (N, H, W).")

    obstacle_color = np.array(standard_obstacle, dtype=images.dtype)
    waypoint_color = np.array(waypoint, dtype=images.dtype)
    start_color = np.array(START_COLOR, dtype=images.dtype)
    end_color = np.array(END_COLOR, dtype=images.dtype)

    def octile_distance(mask: np.ndarray) -> np.ndarray:
        """Compute octile distance (8-connected, costs 1/√2) using EDT nearest indices (vectorized)."""
        inv = ~mask
        dist_euclid, nearest = distance_transform_edt(inv, return_indices=True)
        coords = np.indices(mask.shape)
        dy = np.abs(coords[0] - nearest[0])
        dx = np.abs(coords[1] - nearest[1])
        mx = np.maximum(dx, dy)
        mn = np.minimum(dx, dy)
        return mx + (np.sqrt(2.0) - 1.0) * mn

    modified_images = images.copy()
    rng = np.random.default_rng()

    for idx in tqdm(range(modified_images.shape[0])):
        img = modified_images[idx]
        path_mask = path_labels[idx].astype(bool)

        # Current obstacle mask
        obstacle_mask = (
            (img[:, :, 0] == obstacle_color[0]) &
            (img[:, :, 1] == obstacle_color[1]) &
            (img[:, :, 2] == obstacle_color[2])
        )

        start = np.argwhere(
            (img[:, :, 0] == start_color[0]) &
            (img[:, :, 1] == start_color[1]) &
            (img[:, :, 2] == start_color[2])
        )[0]

        # Candidates: non-obstacle and non-path pixels within proximity (octile distance) of path
        dist_map = octile_distance(path_mask)

        proximity_mask = dist_map <= float(proximity)
        candidate_mask = proximity_mask & (~path_mask) & (~obstacle_mask)

        candidate_indices = np.argwhere(candidate_mask)
        if candidate_indices.size == 0:
            # No available spot; skip this image
            continue

        # Pick one random candidate
        is_valid = False
        tries = 0
        while not is_valid and tries < timeout:
            y, x = candidate_indices[rng.integers(0, candidate_indices.shape[0])]
            astar_waypoints, _, _ = run_astar_2D(~obstacle_mask, start[0], start[1], y, x)  # differnt x and y convention

            if len(astar_waypoints) > 0:
                is_valid = True

            tries += 1

        if tries >= timeout:
            continue

        img[y, x] = waypoint_color

        modified_images[idx] = img

    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save new copy of the dataset with updated images, other arrays unchanged
    np.savez_compressed(
        save_path,
        image=modified_images,
        label=arr['label'] if 'label' in arr else None,
        rgb_label=arr['rgb_label'] if 'rgb_label' in arr else None,
        path_label=arr['path_label'] if 'path_label' in arr else None,
        path_rgb_label=arr['path_rgb_label'] if 'path_rgb_label' in arr else None,
    )
