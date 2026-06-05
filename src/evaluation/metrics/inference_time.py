import time
from types import SimpleNamespace
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


def _rgb_preprocess_maps(images: torch.Tensor):
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=images.device)
    obstacle_map = torch.sigmoid((images - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    obstacle_map = (~(obstacle_map < 1.51)).float()

    start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=images.device)
    start_map = torch.sigmoid((images - start_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    start_map = (start_map < 1.51).float()

    end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)
    end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
    end_map = (end_map < 1.51).float()

    return obstacle_map, start_map, end_map


def _extract_model_logits(model_output):
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    return model_output


def _measure_combined_model_times(model, img_tensor: torch.Tensor, use_rgb: bool, ours: bool, device: Optional[str]):
    obstacle_map, start_map, end_map = _rgb_preprocess_maps(img_tensor)

    astar_outputs = None

    with torch.no_grad():
        if hasattr(model, "perform_astar"):
            _cuda_sync_if_needed(device)
            if use_rgb:
                t0 = time.perf_counter()
                encoder_output = model.encoder(img_tensor)
                _cuda_sync_if_needed(device)
                inference_time_s = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                encoder_output = model.encoder(torch.cat((obstacle_map, start_map + end_map), dim=1))  # neural A*
                # encoder_output = model.encoder(torch.cat((obstacle_map, start_map, end_map), dim=1))  # iA*
                _cuda_sync_if_needed(device)
                inference_time_s = time.perf_counter() - t0

            # print("\n\n", encoder_output.shape, encoder_output.min(), encoder_output.max(), encoder_output.mean())

            logits = _extract_model_logits(encoder_output).to(dtype=torch.float32)  # squeeze here only for iA*
            cost_maps = torch.sigmoid(logits.unsqueeze(1))

            if ours:
                cost_maps = -cost_maps + 1  # we have a different conversion here, high is good for our model for neural A* and iA* its oposite
                cost_maps_scaling = 10.0
            else:
                cost_maps_scaling = 4.0
            
            # cost_maps_scaling differs by model to keep similar optimality/search effort tradeoffs as reported in the paper
            # (Cost Factor / Expansion Factor): 
            # Neural Astar: 
            cost_maps = cost_maps * cost_maps_scaling

            obstacles_maps = obstacle_map

            _cuda_sync_if_needed(device)
            # print("\n\n", cost_maps.shape, cost_maps.min(), cost_maps.max(), cost_maps.mean())
            t1 = time.perf_counter()
            astar_outputs = model.perform_astar(cost_maps, start_map, end_map, obstacles_maps, False)
            _cuda_sync_if_needed(device)
            planning_time_s = time.perf_counter() - t1

        elif hasattr(model, "dastar"):
            _cuda_sync_if_needed(device)
            if use_rgb:
                t0 = time.perf_counter()
                encoder_output = model.encoder(img_tensor)
                _cuda_sync_if_needed(device)
                inference_time_s = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                encoder_output = model.encoder(torch.cat((obstacle_map, start_map + end_map), dim=1))  # neural A*
                # encoder_output = model.encoder(torch.cat((obstacle_map, start_map, end_map), dim=1)).squeeze(1)  # iA*, squeeze here
                _cuda_sync_if_needed(device)
                inference_time_s = time.perf_counter() - t0
                
            # print("\n\n", encoder_output.shape, encoder_output.min(), encoder_output.max(), encoder_output.mean())

            logits = _extract_model_logits(encoder_output).to(dtype=torch.float32).squeeze(1)  # only for iA*
            if ours:
                logits = -logits + 1  # we have a different conversion here, high is good for our model for neural A* and iA* its oposite
            cost_maps = torch.sigmoid(logits.unsqueeze(1))

            if hasattr(model, "init_obstacles_maps"):
                obstacles_maps = model.init_obstacles_maps(obstacle_map)
            else:
                obstacles_maps = obstacle_map

            _cuda_sync_if_needed(device)
            # print("\n\n", cost_maps.shape, cost_maps.min(), cost_maps.max(), cost_maps.mean())
            t1 = time.perf_counter()
            astar_outputs = model.dastar(cost_maps, start_map, end_map, obstacles_maps)
            _cuda_sync_if_needed(device)
            planning_time_s = time.perf_counter() - t1

        else:
            raise ValueError("Combined-timing model does not expose a planner entry point.")

    path_map_np = None
    history_map_np = None
    if astar_outputs is not None:
        try:
            path_map = astar_outputs.paths[0, 0]
            history_map = astar_outputs.histories[0, 0]
            path_map_np = path_map.detach().to(dtype=torch.float32).cpu().numpy()
            history_map_np = history_map.detach().to(dtype=torch.float32).cpu().numpy()
        except Exception:
            path_map_np = None
            history_map_np = None

    return inference_time_s, planning_time_s, path_map_np, history_map_np


def _daastar_infer(
    model_module: torch.nn.Module,
    images: torch.Tensor,
    device: Optional[str],
) -> tuple[SimpleNamespace, float, float]:
    walkable_map, start_map, end_map = _rgb_preprocess_maps(images)

    if hasattr(model_module, "planner"):
        _cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        cost_maps, obstacles_maps, _ = model_module.planner.data_preprocessing(
            walkable_map,
            start_map,
            end_map,
            prob_maps=None,
        )
        _cuda_sync_if_needed(device)
        inference_time_s = time.perf_counter() - t0

        _cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        outputs = model_module.planner.perform_astar(
            cost_maps,
            start_map,
            end_map,
            obstacles_maps,
            store_intermediate_results=False,
            store_hist_coordinates=False,
            disable_heuristic=False,
            disable_compute_path_angle=False,
        )
        _cuda_sync_if_needed(device)
        planning_time_s = time.perf_counter() - t1
    else:
        _cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        outputs, _ = model_module.infer(
            walkable_map,
            start_map,
            end_map,
            prob_maps=None,
        )
        _cuda_sync_if_needed(device)
        inference_time_s = time.perf_counter() - t0
        planning_time_s = 0.0

    out = SimpleNamespace(
        logits=outputs.paths.squeeze(1),
        hidden_states=outputs.histories.squeeze(1),
    )
    return out, inference_time_s, planning_time_s


def _measure_daastar_times(model, img_tensor: torch.Tensor, device: Optional[str]):
    out, inference_time_s, planning_time_s = _daastar_infer(model, img_tensor, device)

    path_map_np = None
    history_map_np = None
    try:
        path_map = out.logits[0]
        history_map = out.hidden_states[0]
        path_map_np = path_map.detach().to(dtype=torch.float32).cpu().numpy()
        history_map_np = history_map.detach().to(dtype=torch.float32).cpu().numpy()
    except Exception:
        path_map_np = None
        history_map_np = None

    return inference_time_s, planning_time_s, path_map_np, history_map_np


def measure_inference_time(
    model: torch.nn.Module,
    dataset,  # HF DatasetDict or Dataset
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    min_hardness: Optional[float] = 1.05,
    transPath: bool = False,
    daastar: bool = False,
    iastar: bool = False,
    astar: bool = False,
    wastar: bool = False,
    restrict_search_space: bool = False,
    use_ours_with_diff_astar: bool = False,
    use_rgb: bool = True,
    ours: bool = False, 
):
    """Measure model inference and planner runtime over dataset samples.

    For non-A* actor heads (e.g. TransPath / single-shot actor), this measures:
    - forward inference time
    - additional planning time (focal search by default or A* if ``astar=True``)

    For iA* / NeuralA* models, planning is already part of forward inference and
    therefore planner time remains 0 and combined time equals inference time.
    """

    assert not (transPath and iastar), "Only one out of transPath and iastar flags can be set."
    assert not (daastar and iastar), "Only one out of daastar and iastar flags can be set."
    if use_ours_with_diff_astar and iastar:
        raise ValueError("use_ours_with_diff_astar cannot be combined with transPath or iastar.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(dtype=torch.float32)
    model.eval()

    # iA* / NeuralA* expose a planner inside the model.
    combined_inside_model = (
        daastar
        or hasattr(model, "perform_astar")
        or (hasattr(model, "dastar") and hasattr(model, "encode"))
    )
    
    if not combined_inside_model:
        # iastar and neuralastar are partially compiled by default
        model = torch.compile(model)
    elif hasattr(model, "encoder"):
        model.encoder.forward = torch.compile(model.encoder)
    
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
    # path/cost metrics
    pred_path_costs = []
    optimal_path_costs = []
    pred_path_expansions = []
    optimal_path_expansions = []
    cost_factors = []
    expansion_ratios = []
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
            img_tensor = img_tensor.to(device=device, dtype=torch.float32, non_blocking=True)

            pred_path_waypoints = None
            a_star_path_cost = None
            a_star_path_expansions = None

            if combined_inside_model:
                if daastar:
                    inference_time_s, planning_time_s, combined_path_map, combined_history_map = _measure_daastar_times(model, img_tensor, device)
                else:
                    inference_time_s, planning_time_s, combined_path_map, combined_history_map = _measure_combined_model_times(model, img_tensor, use_rgb, ours, device)
                focal_no_nn_time_s = 0.0

                if combined_path_map is not None:
                    path_mask = combined_path_map >= 0.5
                    path_mask[start[0], start[1]] = True
                    path_mask[end[0], end[1]] = True

                    collision_map = np.logical_and(path_mask, obstacle_grid)
                    path_mask[collision_map] = False

                    pred_path_waypoints, a_star_path_cost, a_star_path_expansions_from_mask, _ = run_astar_2D(
                        path_mask, start[0], start[1], end[0], end[1]
                    )

                    if combined_history_map is not None:
                        a_star_path_expansions = float(np.sum(combined_history_map))
                    else:
                        a_star_path_expansions = float(a_star_path_expansions_from_mask)
            else:
                _cuda_sync_if_needed(device)
                t0 = time.perf_counter()
                out = model(img_tensor)
                _cuda_sync_if_needed(device)
                inference_time_s = time.perf_counter() - t0

                if hasattr(out, "logits"):
                    logits = out.logits.to(dtype=torch.float32)
                elif isinstance(out, dict) and "logits" in out:
                    logits = out["logits"].to(dtype=torch.float32)
                else:
                    logits = out.to(dtype=torch.float32)

                if not transPath:
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
                    t0 = time.perf_counter()
                    octile = octile_distance_map(path_pred.shape, (end[0], end[1]))
                    heur_t = time.perf_counter() - t0
                    h = octile + 25 * (1 - path_pred)

                    t2 = time.perf_counter()
                    _, _, _, exp_map = run_astar_2D_heuristic_grid(planning_space, octile, start[0], start[1], end[0], end[1], w=1.0)
                    focal_no_nn_time_s = time.perf_counter() - t2#  + heur_t

                    t1 = time.perf_counter()
                    pred_path_waypoints, a_star_path_cost, a_star_path_expansions, exp_map = run_astar_2D_heuristic_grid(planning_space, h, start[0], start[1], end[0], end[1], w=2.0)
                    planning_time_s = time.perf_counter() - t1#  + heur_t

                else:
                    if astar:
                        t1 = time.perf_counter()
                        pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_astar_2D(planning_space, start[0], start[1], end[0], end[1])
                        planning_time_s = time.perf_counter() - t1
                    else:
                        
                        octile = octile_distance_map(path_pred.shape, (end[0], end[1]))
                    
                        t2 = time.perf_counter()
                        _, _, _, _ = run_focal_search_2D(
                            planning_space,
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            w=2.0,
                            path_proximity_map=np.zeros_like(path_pred),
                        )
                        focal_no_nn_time_s = time.perf_counter() - t2

                        h = -np.copy(path_pred) + 1.0
                        t1 = time.perf_counter()
                        pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_focal_search_2D(
                            planning_space,
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            w=2.0,
                            path_proximity_map=h,
                        )
                        planning_time_s = time.perf_counter() - t1

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
                img_tensor = img_tensor.to(device=device, dtype=torch.float32, non_blocking=True)

                pred_path_waypoints = None
                a_star_path_cost = None
                a_star_path_expansions = None

                if combined_inside_model:
                    if daastar:
                        inference_time_s, planning_time_s, combined_path_map, combined_history_map = _measure_daastar_times(model, img_tensor, device)
                    else:
                        inference_time_s, planning_time_s, combined_path_map, combined_history_map = _measure_combined_model_times(model, img_tensor, use_rgb, ours, device)
                    focal_no_nn_time_s = 0.0

                    if combined_path_map is not None:
                        path_mask = combined_path_map >= 0.5
                        path_mask[start[0], start[1]] = True
                        path_mask[end[0], end[1]] = True

                        collision_map = np.logical_and(path_mask, obstacle_grid)
                        path_mask[collision_map] = False

                        pred_path_waypoints, a_star_path_cost, a_star_path_expansions_from_mask, _ = run_astar_2D(
                            path_mask, start[0], start[1], end[0], end[1]
                        )

                        if combined_history_map is not None:
                            a_star_path_expansions = float(np.sum(combined_history_map))
                        else:
                            a_star_path_expansions = float(a_star_path_expansions_from_mask)
                else:
                    _cuda_sync_if_needed(device)
                    t0 = time.perf_counter()
                    out = model(img_tensor)
                    _cuda_sync_if_needed(device)
                    inference_time_s = time.perf_counter() - t0

                    if hasattr(out, "logits"):
                        logits = out.logits.to(dtype=torch.float32)
                    elif isinstance(out, dict) and "logits" in out:
                        logits = out["logits"].to(dtype=torch.float32)
                    else:
                        logits = out.to(dtype=torch.float32)

                    if not transPath:
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
                        t0 = time.perf_counter()
                        octile = octile_distance_map(path_pred.shape, (end[0], end[1]))
                        heur_t = time.perf_counter() - t0
                        h = octile + 25 * (1 - path_pred)

                        t2 = time.perf_counter()
                        _, _, _, exp_map = run_astar_2D_heuristic_grid(planning_space, octile, start[0], start[1], end[0], end[1], w=1.0)
                        focal_no_nn_time_s = time.perf_counter() - t2#  + heur_t

                        t1 = time.perf_counter()
                        pred_path_waypoints, a_star_path_cost, a_star_path_expansions, exp_map = run_astar_2D_heuristic_grid(planning_space, h, start[0], start[1], end[0], end[1], w=2.0)
                        planning_time_s = time.perf_counter() - t1#  + heur_t

                    else:
                        if astar:
                            t2 = time.perf_counter()
                            pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_astar_2D(full_free, start[0], start[1], end[0], end[1])
                            focal_no_nn_time_s = time.perf_counter() - t2

                            t1 = time.perf_counter()
                            pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_astar_2D(planning_space, start[0], start[1], end[0], end[1])
                            planning_time_s = time.perf_counter() - t1
                        else:
                            
                            octile = octile_distance_map(path_pred.shape, (end[0], end[1]))
                        
                            t2 = time.perf_counter()
                            _, _, _, _ = run_focal_search_2D(
                                planning_space,
                                start[0],
                                start[1],
                                end[0],
                                end[1],
                                w=2.0,
                                path_proximity_map=np.zeros_like(path_pred),
                            )
                            focal_no_nn_time_s = time.perf_counter() - t2

                            h = -np.copy(path_pred) + 1.0
                            t1 = time.perf_counter()
                            pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_focal_search_2D(
                                planning_space,
                                start[0],
                                start[1],
                                end[0],
                                end[1],
                                w=2.0,
                                path_proximity_map=h,
                            )
                            planning_time_s = time.perf_counter() - t1

            timed_samples += 1
            inference_times.append(float(inference_time_s))
            planning_times.append(float(planning_time_s))
            focal_no_nn_times.append(float(focal_no_nn_time_s))
            combined_times.append(float(inference_time_s + planning_time_s))
            # compute naive optimal a* cost/expansions on full free map
            try:
                a_star_waypoints, a_star_grid_cost, a_star_grid_expansions, _ = run_astar_2D(
                    full_free, start[0], start[1], end[0], end[1]
                )
            except Exception:
                a_star_grid_cost = None
                a_star_grid_expansions = None

            # record predicted vs optimal costs/expansions when a predicted path exists
            if pred_path_waypoints is not None and len(pred_path_waypoints) > 0 and a_star_path_cost is not None and a_star_path_expansions is not None:
                # only record when optimal A* produced a finite cost
                if a_star_grid_cost is not None and a_star_grid_cost > 0:
                    pred_path_costs.append(float(a_star_path_cost))
                    optimal_path_costs.append(float(a_star_grid_cost))
                    pred_path_expansions.append(float(a_star_path_expansions))
                    optimal_path_expansions.append(float(a_star_grid_expansions))
                    # ratios
                    cost_factors.append(float(a_star_path_cost) / max(float(a_star_grid_cost), 1e-8))
                    expansion_ratios.append(float(a_star_path_expansions) / max(float(a_star_grid_expansions), 1e-8))

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
        "cost_factors": cost_factors,
        "expansion_ratios": expansion_ratios,
        "pred_path_costs": pred_path_costs,
        "optimal_path_costs": optimal_path_costs,
        "pred_path_expansions": pred_path_expansions,
        "optimal_path_expansions": optimal_path_expansions,
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
