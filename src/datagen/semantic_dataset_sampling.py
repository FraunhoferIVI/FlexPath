import numpy as np
import os
import shutil
import zarr

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


def _iter_slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def _open_zarr_io(path_to_ds: str, save_path: str, skip_arrays=()):
    src = zarr.open_group(path_to_ds, mode="r")

    if "image" not in src:
        raise KeyError("Input dataset must contain key 'image'.")
    if "path_label" not in src:
        raise KeyError("Input dataset must contain key 'path_label'.")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    dst = zarr.open_group(save_path, mode="w")
    dst.attrs.update(src.attrs.asdict())

    for name, arr in src.arrays():
        if name in skip_arrays:
            continue
        out = dst.create_dataset(
            name,
            shape=arr.shape,
            chunks=arr.chunks,
            dtype=arr.dtype,
            compressor=arr.compressor,
            fill_value=arr.fill_value,
            filters=arr.filters,
            order=arr.order,
        )
        out.attrs.update(arr.attrs.asdict())
        batch_size = arr.chunks[0] if arr.chunks else arr.shape[0]
        for batch_slice in _iter_slices(arr.shape[0], batch_size):
            out[batch_slice] = arr[batch_slice]

    return src, dst


def _create_image_output(src, dst):
    images = src["image"]
    out = dst.create_dataset(
        "image",
        shape=images.shape,
        chunks=images.chunks,
        dtype=images.dtype,
        compressor=images.compressor,
        fill_value=images.fill_value,
        filters=images.filters,
        order=images.order,
    )
    out.attrs.update(images.attrs.asdict())
    return out


def _validate_shapes(images, path_labels):
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected images with shape (N, H, W, 3).")
    if not (path_labels.ndim == 3 and path_labels.shape == images.shape[:3]):
        raise ValueError("path_label shape must align with images (N, H, W).")


def _octile_distance(mask: np.ndarray) -> np.ndarray:
    inv = ~mask
    _, nearest = distance_transform_edt(inv, return_indices=True)
    coords = np.indices(mask.shape)
    dy = np.abs(coords[0] - nearest[0])
    dx = np.abs(coords[1] - nearest[1])
    mx = np.maximum(dx, dy)
    mn = np.minimum(dx, dy)
    return mx + (np.sqrt(2.0) - 1.0) * mn


def sample_obstacles(path_to_ds: str, save_path: str, proximity: int = 10, num_obstacles: int = 10):
    """
    For each image in the dataset, place up to `num_obstacles` new obstacle pixels near the optimal path.

    Args:
        path_to_ds: input zarr group containing keys: image, label, rgb_label, path_label, path_rgb_label
        save_path: output zarr group for the modified dataset. Directories will be created if needed.
        proximity: how far (8-connected steps) away from the path an obstacle may be placed.
        num_obstacles: maximum number of obstacle pixels to add per image (no replacement).
    """

    src, dst = _open_zarr_io(path_to_ds, save_path, skip_arrays=("image",))
    images = src["image"]
    path_labels = src["path_label"]
    _validate_shapes(images, path_labels)
    output_images = _create_image_output(src, dst)

    obstacle_color = np.array(standard_obstacle, dtype=images.dtype)
    badobstacle_color = np.array(badobstacle, dtype=images.dtype)

    rng = np.random.default_rng()
    batch_size = images.chunks[0] if images.chunks else images.shape[0]

    with tqdm(total=images.shape[0]) as pbar:
        for batch_slice in _iter_slices(images.shape[0], batch_size):
            image_batch = np.asarray(images[batch_slice]).copy()
            path_batch = np.asarray(path_labels[batch_slice])

            for offset in range(image_batch.shape[0]):
                img = image_batch[offset]
                path_mask = path_batch[offset].astype(bool)

                obstacle_mask = (
                    (img[:, :, 0] == obstacle_color[0])
                    & (img[:, :, 1] == obstacle_color[1])
                    & (img[:, :, 2] == obstacle_color[2])
                )

                dist_map = _octile_distance(path_mask)
                proximity_mask = dist_map <= float(proximity)
                candidate_mask = proximity_mask & (~path_mask) & (~obstacle_mask)

                candidate_indices = np.argwhere(candidate_mask)
                if candidate_indices.size == 0:
                    continue

                k = min(num_obstacles, candidate_indices.shape[0])
                choice_idx = rng.choice(candidate_indices.shape[0], size=k, replace=False)
                chosen = candidate_indices[choice_idx]

                a = k // 2
                ys, xs = chosen[:a, 0], chosen[:a, 1]
                img[ys, xs] = badobstacle_color

                ys, xs = chosen[a:, 0], chosen[a:, 1]
                img[ys, xs] = obstacle_color

            output_images[batch_slice] = image_batch
            pbar.update(image_batch.shape[0])


def sample_waypoints(path_to_ds: str, save_path: str, proximity: int = 20, timeout: int = 10):
    """
    For each image in the dataset, place a single waypoint pixel near the optimal path.

    Args:
        path_to_ds: input zarr group containing keys: image, label, rgb_label, path_label, path_rgb_label
        save_path: output zarr group for the modified dataset. Directories will be created if needed.
        proximity: how far (8-connected steps) away from the path a waypoint may be placed.
    """

    src, dst = _open_zarr_io(path_to_ds, save_path, skip_arrays=("image",))
    images = src["image"]
    path_labels = src["path_label"]
    _validate_shapes(images, path_labels)
    output_images = _create_image_output(src, dst)

    obstacle_color = np.array(standard_obstacle, dtype=images.dtype)
    waypoint_color = np.array(waypoint, dtype=images.dtype)
    start_color = np.array(START_COLOR, dtype=images.dtype)

    rng = np.random.default_rng()
    batch_size = images.chunks[0] if images.chunks else images.shape[0]

    with tqdm(total=images.shape[0]) as pbar:
        for batch_slice in _iter_slices(images.shape[0], batch_size):
            image_batch = np.asarray(images[batch_slice]).copy()
            path_batch = np.asarray(path_labels[batch_slice])

            for offset in range(image_batch.shape[0]):
                img = image_batch[offset]
                path_mask = path_batch[offset].astype(bool)

                obstacle_mask = (
                    (img[:, :, 0] == obstacle_color[0])
                    & (img[:, :, 1] == obstacle_color[1])
                    & (img[:, :, 2] == obstacle_color[2])
                )

                start = np.argwhere(
                    (img[:, :, 0] == start_color[0])
                    & (img[:, :, 1] == start_color[1])
                    & (img[:, :, 2] == start_color[2])
                )[0]

                dist_map = _octile_distance(path_mask)
                proximity_mask = dist_map <= float(proximity)
                candidate_mask = proximity_mask & (~path_mask) & (~obstacle_mask)

                candidate_indices = np.argwhere(candidate_mask)
                if candidate_indices.size == 0:
                    continue

                is_valid = False
                tries = 0
                while not is_valid and tries < timeout:
                    y, x = candidate_indices[rng.integers(0, candidate_indices.shape[0])]
                    astar_waypoints = run_astar_2D(~obstacle_mask, start[0], start[1], y, x)[0]

                    if len(astar_waypoints) > 0:
                        is_valid = True

                    tries += 1

                if tries >= timeout:
                    continue

                img[y, x] = waypoint_color

            output_images[batch_slice] = image_batch
            pbar.update(image_batch.shape[0])
