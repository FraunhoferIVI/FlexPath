import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from cstar.pathfinding import run_astar_2D, run_focal_search_2D, run_astar_2D_heuristic_grid
from src.evaluation.metrics.general import binary_obstacle_from_image, octile_distance


torch.set_float32_matmul_precision('high')


def octile_distance_map(shape, goal):
    """
    Compute octile distance from every cell in a 2D grid to a goal.

    Parameters
    ----------
    shape : tuple (H, W)
        Shape of the 2D grid.
    goal : tuple (gy, gx)
        Goal coordinates (row, col).

    Returns
    -------
    dist : np.ndarray of shape (H, W)
        Octile distance to goal for every cell.
    """
    H, W = shape
    gy, gx = goal

    # Create coordinate grid
    y_indices, x_indices = np.indices((H, W))

    dx = np.abs(x_indices - gx)
    dy = np.abs(y_indices - gy)

    # Octile distance
    dist = np.maximum(dx, dy) + (np.sqrt(2) - 1) * np.minimum(dx, dy)

    return dist

def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None, None
    return float(np.mean(arr)), float(np.std(arr))


def _cuda_sync_if_needed(device: Optional[str]):
    if device is None:
        return
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_inference_time(
    model: torch.nn.Module,
    dataset,  # HF DatasetDict or Dataset
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    min_hardness: Optional[float] = 1.05,
    astar: bool = False,
    wastar: bool = False,
    restrict_search_space: bool = False,
):
    """Measure model inference and planner runtime over dataset samples.

    This measures:
    - forward inference time
    - additional planning time (focal search by default, or A* / weighted A*
      if ``astar=True`` or ``wastar=True``)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(dtype=torch.bfloat16)
    model.eval()

    combined_inside_model = False
    model = torch.compile(model)
    
    dl = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=128,
    )

    inference_times = []
    planning_times = []
    focal_no_nn_times = []
    combined_times = []
    hardness_list = []
    total_samples = 0
    timed_samples = 0

    print("Warmup...")
    i = 0
    for idx, sample in tqdm(enumerate(dl)):

        i += 1

        if i > 10:
            break

        img_tensor = sample["images"]  # [1, C, H, W]
        start = sample["starts"][0].cpu().numpy().astype(np.int64)
        end = sample["ends"][0].cpu().numpy().astype(np.int64)

        img_np = torch.clamp(img_tensor * 255, min=0, max=255).cpu().numpy()
        img_np_uint8 = img_np.astype(np.uint8).transpose(0, 2, 3, 1)  # [1, H, W, C]
        obstacle_grid = binary_obstacle_from_image(img_np_uint8)[0]
        full_free = ~obstacle_grid

        if min_hardness is not None:
            _, a_star_grid_cost, _, _ = run_astar_2D(full_free, start[0], start[1], end[0], end[1])
            heuristic = max(float(octile_distance(start[0], start[1], end[0], end[1])), 1e-8)
            hardness_i = float(a_star_grid_cost) / heuristic
            if hardness_i < min_hardness:
                continue
            hardness_list.append(hardness_i)

        with torch.no_grad():
            img_tensor = img_tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True)

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()
            out = model(img_tensor)
            _cuda_sync_if_needed(device)
            inference_time_s = time.perf_counter() - t0

        planning_time_s = 0.0
        focal_no_nn_time_s = 0.0
        if not combined_inside_model:
            if hasattr(out, "logits"):
                logits = out.logits.to(dtype=torch.float32)
            elif isinstance(out, dict) and "logits" in out:
                logits = out["logits"].to(dtype=torch.float32)
            else:
                logits = out.to(dtype=torch.float32)

            logits = torch.nn.functional.tanh(logits)

            preds_activated = (logits + 1.0) / 2.0
            path_pred = preds_activated.detach().cpu().numpy()[0]
            if path_pred.ndim == 3:
                path_pred = path_pred[0]

            path_mask = path_pred >= 0.5
            path_mask[start[0], start[1]] = True
            path_mask[end[0], end[1]] = True

            collision_map = np.logical_and(path_mask, obstacle_grid)
            path_mask[collision_map] = False

            planning_space = path_mask if restrict_search_space else full_free

            if astar:
                t1 = time.perf_counter()
                run_astar_2D(planning_space, start[0], start[1], end[0], end[1])
            else:
                t1 = time.perf_counter()
                run_focal_search_2D(
                    planning_space,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    w=2.0,
                    path_proximity_map=(-np.copy(path_pred) + 1.0),
                )
            planning_time_s = time.perf_counter() - t1

            t2 = time.perf_counter()
            run_focal_search_2D(
                planning_space,
                start[0],
                start[1],
                end[0],
                end[1],
                w=2.0,
                path_proximity_map=np.zeros_like(path_pred, dtype=np.float32),
            )
            focal_no_nn_time_s = time.perf_counter() - t2

    print("Starting time measurements...")
    for _ in range(3):
        for idx, sample in tqdm(enumerate(dl)):
            if max_samples is not None and idx >= max_samples:
                break

            total_samples += 1

            img_tensor = sample["images"]  # [1, C, H, W]
            start = sample["starts"][0].cpu().numpy().astype(np.int64)
            end = sample["ends"][0].cpu().numpy().astype(np.int64)

            img_np = torch.clamp(img_tensor * 255, min=0, max=255).cpu().numpy()
            img_np_uint8 = img_np.astype(np.uint8).transpose(0, 2, 3, 1)  # [1, H, W, C]
            obstacle_grid = binary_obstacle_from_image(img_np_uint8)[0]
            full_free = ~obstacle_grid

            if min_hardness is not None:
                _, a_star_grid_cost, _, _ = run_astar_2D(full_free, start[0], start[1], end[0], end[1])
                heuristic = max(float(octile_distance(start[0], start[1], end[0], end[1])), 1e-8)
                hardness_i = float(a_star_grid_cost) / heuristic
                if hardness_i < min_hardness:
                    continue
                hardness_list.append(hardness_i)

            with torch.no_grad():
                img_tensor = img_tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True)

                _cuda_sync_if_needed(device)
                t0 = time.perf_counter()
                out = model(img_tensor)
                _cuda_sync_if_needed(device)
                inference_time_s = time.perf_counter() - t0

            planning_time_s = 0.0
            focal_no_nn_time_s = 0.0
            if not combined_inside_model:
                if hasattr(out, "logits"):
                    logits = out.logits.to(dtype=torch.float32)
                elif isinstance(out, dict) and "logits" in out:
                    logits = out["logits"].to(dtype=torch.float32)
                else:
                    logits = out.to(dtype=torch.float32)

                logits = torch.nn.functional.tanh(logits)

                preds_activated = (logits + 1.0) / 2.0
                path_pred = preds_activated.detach().cpu().numpy()[0]
                if path_pred.ndim == 3:
                    path_pred = path_pred[0]

                path_mask = path_pred >= 0.5
                path_mask[start[0], start[1]] = True
                path_mask[end[0], end[1]] = True

                collision_map = np.logical_and(path_mask, obstacle_grid)
                path_mask[collision_map] = False

                planning_space = path_mask if restrict_search_space else full_free

                if wastar:
                    octile = octile_distance_map(path_pred.shape, (end[0], end[1]))
                    h = octile + 25 * (1 - path_pred)

                    t1 = time.perf_counter()
                    run_astar_2D_heuristic_grid(planning_space, h, start[0], start[1], end[0], end[1], w=2.0)
                    planning_time_s = time.perf_counter() - t1

                    t2 = time.perf_counter()
                    octile = octile_distance_map(path_pred.shape, (end[0], end[1]))
                    run_astar_2D_heuristic_grid(planning_space, octile, start[0], start[1], end[0], end[1], w=2.0)
                    focal_no_nn_time_s = time.perf_counter() - t2

                else:
                    if astar:
                        t1 = time.perf_counter()
                        run_astar_2D(planning_space, start[0], start[1], end[0], end[1])
                    else:
                        h = -np.copy(path_pred) + 1.0
                        t1 = time.perf_counter()
                        run_focal_search_2D(
                            planning_space,
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            w=2.0,
                            path_proximity_map=h,
                        )
                    planning_time_s = time.perf_counter() - t1

                    octile = octile_distance_map(path_pred.shape, (end[0], end[1]))

                    t2 = time.perf_counter()
                    run_focal_search_2D(
                        planning_space,
                        start[0],
                        start[1],
                        end[0],
                        end[1],
                        w=2.0,
                        path_proximity_map=octile,
                    )
                    focal_no_nn_time_s = time.perf_counter() - t2

            timed_samples += 1
            inference_times.append(float(inference_time_s))
            planning_times.append(float(planning_time_s))
            focal_no_nn_times.append(float(focal_no_nn_time_s))
            combined_times.append(float(inference_time_s + planning_time_s))

    inference_mean, inference_std = _mean_std(inference_times)
    planning_mean, planning_std = _mean_std(planning_times)
    focal_no_nn_mean, focal_no_nn_std = _mean_std(focal_no_nn_times)
    combined_mean, combined_std = _mean_std(combined_times)
    hardness_mean, hardness_std = _mean_std(hardness_list)

    return {
        "total_samples": int(total_samples),
        "timed_samples": int(timed_samples),
        "combined_inside_model": bool(combined_inside_model),
        "inference_times_s": inference_times,
        "planning_times_s": planning_times,
        "focal_no_nn_times_s": focal_no_nn_times,
        "combined_times_s": combined_times,
        "hardness_list": hardness_list,
        "metrics": {
            "inference_time_s": {
                "mean": inference_mean,
                "std": inference_std,
                "count": int(len(inference_times)),
            },
            "planning_time_s": {
                "mean": planning_mean,
                "std": planning_std,
                "count": int(len(planning_times)),
            },
            "focal_no_nn_time_s": {
                "mean": focal_no_nn_mean,
                "std": focal_no_nn_std,
                "count": int(len(focal_no_nn_times)),
            },
            "combined_time_s": {
                "mean": combined_mean,
                "std": combined_std,
                "count": int(len(combined_times)),
            },
            "hardness": {
                "mean": hardness_mean,
                "std": hardness_std,
                "count": int(len(hardness_list)),
            },
        },
    }
