"""General evaluation helpers.

Utilities to load a pretrained actor from an experiment directory, load a
dataset (huggingface style), iterate over samples, run the model to obtain
predictions and extract a binary obstacle occupancy grid from images using
the project's normalized color encoding.

This module re-uses the existing loading helpers in
`src.utils.checkpoint_loader` and the dataset conventions used elsewhere in
the codebase.
"""

from pathlib import Path

from typing import Dict, Optional

import numpy as np

import torch

from torch.utils.data.dataloader import DataLoader

from src.utils import checkpoint_loader
from omegaconf import OmegaConf
from src.data.dataset_factory import DatasetFactory
import logging

from src.utils.model_registry import get_actor_from_config  # Updated to use unified registry

from tqdm import tqdm

from scipy.ndimage import distance_transform_edt
from cstar.pathfinding import run_astar_2D, run_focal_search_2D, run_mhastar_2D
from .obstacle_avoidance import compute_obstacle_avoidance_metrics_from_thresholded_path


logger = logging.getLogger(__name__)


def load_actor_from_experiment(exp_dir: str, device: Optional[str] = None, use_best: bool = True):

    """
    Load an actor model from an experiment directory.

    Args:
        exp_dir: Path to the experiment directory (contains config.yaml and checkpoints/)
        device: torch device string (auto-detected if None)
        use_best: whether to use the best.pth checkpoint (otherwise latest.pth)

    Returns:
        torch.nn.Module in eval() mode on the requested device

    """

    cfg = OmegaConf.load(str(Path.joinpath(Path(exp_dir), "config.yaml")))

    file = "best.pth" if use_best else "latest.pth"
    checkpoint_path = Path(exp_dir) / "checkpoints" / file

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if hasattr(cfg, "model"):
        if hasattr(cfg.model, "single_shot"):
            # -> config from finetuning run
            exp_dir = cfg.model.single_shot.actor.experiment_dir
        else:
            # -> config from pretraining run
            state_dict_key = "state_dict" if "state_dict" in checkpoint.keys() else "model_state_dict"

            state_dict = checkpoint[state_dict_key]  # {k.replace("_orig_mod.", ""): v for k, v in checkpoint[state_dict_key].items()}

            # Load actor model
            logger.info("Creating model...")
            model = get_actor_from_config(cfg)  # Updated to use unified registry

            model.load_state_dict(state_dict)

            model = model.to(device=device)
            model.eval()
            return model

    model = checkpoint_loader.load_pretrained_actor_from_experiment(exp_dir, use_best=use_best, device=device)

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
    
def load_dataset_via_factory(dataset_config_path: str, split: str = "validation"):

    """
    Load a dataset using the project's DatasetFactory.

    Args:
        dataset_config_path: path to a yaml config file compatible with DatasetFactory
        split: dataset split to create (e.g. 'train' or 'validation')

    Returns:
        dataset instance returned by DatasetFactory.create_actor_dataset

    """

    cfg_path = Path(dataset_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))
    
    if not hasattr(cfg, "data"):
        # -> direct dataset config
        cfg.data = cfg

    # Ensure evaluation normalization (simple) unless specified
    if not hasattr(cfg.data, "normalization"):
        cfg.data.normalization = {}
    cfg.data.normalization.use_simple_norm = True

    # Use DatasetFactory (actor dataset assumed)
    dataset = DatasetFactory.create_actor_dataset(cfg, split=split)
    return dataset


def binary_obstacle_from_image(img: np.ndarray) -> np.ndarray:

    """
    Extract a binary obstacle occupancy grid from an image.

    The repository uses a fixed color encoding for obstacles: RGB = [76, 76, 255].
    This helper accepts either a uint8 BxHxWx3 RGB image or a normalized
    float image in [0, 1]. It returns a boolean array of shape BxHxW where True
    indicates obstacle pixels.

    Args:
        img: BxHxWx3 uint8 or float32/float64 array. If float, values are expected
             in [0, 1].

    Returns:
        BxHxW boolean numpy array

    """

    if img.dtype == np.float32 or img.dtype == np.float64:
        # normalized floats in [0, 1]
        obs_color = np.array([76, 76, 255], dtype=np.float32) / 255.0
        # allow tiny numerical tolerance
        tol = 1.0 / 255.0 + 1e-6
        mask = np.all(np.isclose(img, obs_color[None, None, None, :], atol=tol), axis=-1)
        return mask
    else:
        # assume uint8
        obs_color = np.array([76, 76, 255], dtype=np.uint8)
        return np.all(img == obs_color[None, None, None, :], axis=-1)
    
def octile_distance(x, y, x2, y2):
    """
    Compute octile distance between points (x, y) and (x2, y2).

    Parameters
    ----------
    x, y, x2, y2 : array_like or scalar
        Coordinates. Must be broadcastable to the same shape.

    Returns
    -------
    dist : ndarray
        Octile distance(s).
    """
    dx = np.abs(np.asarray(x) - np.asarray(x2))
    dy = np.abs(np.asarray(y) - np.asarray(y2))

    return np.maximum(dx, dy) + (np.sqrt(2) - 1.0) * np.minimum(dx, dy)


def iterate_dataset_and_predict(
    model: torch.nn.Module,
    dataset,  # HF DatasetDict or Dataset
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    min_hardness: Optional[float] = 1.05,
    desired_obstacle_distance: float = 1.0,
    astar: bool = False,
    restrict_search_space: bool = False,
    mhastar: bool = False,
) -> Dict[str, object]:

    """
    Iterate over dataset samples, run model inference and accumulate metrics.

    Returns a dict with metric arrays and counters (validity list, confidences,
    cost factors, obstacle metrics, and sample counts) for downstream summaries.

    Notes:
    - The function tries to be robust with dataset element formats. It looks
      for a top-level 'image' field (typical HF dataset). If the image is a
      PIL.Image it will be converted to a numpy uint8 array.
        - The dataset factory already normalizes inputs for evaluation, so the
            evaluation loop assumes images are normalized to [0,1].

    """
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accept either a HF-style DatasetDict, a Dataset object, or a custom ActorDataset
    # DatasetFactory-created ActorDataset implements __len__ and __getitem__ returning a dict
    # if isinstance(dataset, dict) and split in dataset:
    # 	print("\n\ndict!\n\n")
    # 	ds = dataset[split]
    # else:
    # 	ds = dataset

    # collect per-sample metrics so downstream callers can compute mean/std
    validity_list = []  # for hard-enough samples: 1 if valid path found else 0
    path_confidences = []  
    voxel_confidences = []
    opt_planning_expansions = []
    cost_factors = []  # only for valid samples
    expansion_ratios = []  # only for valid samples
    pred_path_costs = []  # predicted path costs (only for valid samples)
    optimal_path_costs = []  # optimal path costs (only for valid samples)
    hardness_list = []  # hardness values for hard-enough samples
    collisions = 0  # total collisions across all samples
    total_samples = 0

    obstacle_metrics = {
        "categories": {
            "all": {"avg_distances": [], "avoidance_ratios": [], "full_avoidance": []},
            "possible": {"avg_distances": [], "avoidance_ratios": [], "full_avoidance": []},
            "impossible": {"avg_distances": [], "avoidance_ratios": [], "full_avoidance": []}
        },
        "counts": {"possible": 0, "impossible": 0},
        "valids": {"possible": 0, "impossible": 0}
    }

    dl = DataLoader(
        dataset,
        batch_size=32,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    for idx, sample in tqdm(enumerate(dl)):
        if max_samples is not None and idx >= max_samples:
            break

        img_tensor = sample["images"]
        starts = sample["starts"].cpu().numpy()
        ends = sample["ends"].cpu().numpy()

        img_np = torch.clamp(img_tensor * 255, min=0, max=255).cpu().numpy()

        # keep a uint8 copy for exact color checks
        img_np_uint8 = img_np.astype(np.uint8).transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            out = model(img_tensor)

        # obstacle grid from original uint8 image
        obstacle_masks = binary_obstacle_from_image(img_np_uint8)

        # model outputs vary across model implementations. Prefer the common
        # attribute 'logits' (SemanticSegmenterOutput), otherwise attempt to
        # use the raw return value.
        logits = out.logits
        logits = torch.nn.functional.tanh(logits)
        preds_activated = (logits + 1.0) / 2.0
        preds_np = preds_activated.cpu().detach().numpy()

        (
            batch_validity_list,
            batch_cost_factors,
            batch_expansion_ratios,
            batch_hardness_list,
            batch_collisions,
            batch_total_samples,
            batch_obstacle_metrics,
            batch_pred_path_costs,
            batch_optimal_path_costs,
            batch_path_confidences,
            batch_voxel_confidences,
            batch_opt_planning_expansions,
        ) = compute_path_metrics(
            obstacle_grids=obstacle_masks,
            path_preds=preds_np,
            starts=starts,
            ends=ends,
            desired_obstacle_distance=desired_obstacle_distance,
            min_hardness=min_hardness,
            astar=astar,
            restrict_search_space=restrict_search_space,
            mhastar=mhastar,
        )

        # accumulate
        validity_list.extend(batch_validity_list)
        cost_factors.extend(batch_cost_factors)
        expansion_ratios.extend(batch_expansion_ratios)
        pred_path_costs.extend(batch_pred_path_costs)
        optimal_path_costs.extend(batch_optimal_path_costs)
        hardness_list.extend(batch_hardness_list)
        path_confidences.extend(batch_path_confidences)
        voxel_confidences.extend(batch_voxel_confidences)
        opt_planning_expansions.extend(batch_opt_planning_expansions)
        collisions += batch_collisions
        total_samples += batch_total_samples

        for category, metrics in batch_obstacle_metrics["categories"].items():
            for metric_name, metric_values in metrics.items():
                obstacle_metrics["categories"][category][metric_name].extend(metric_values)

        obstacle_metrics["counts"]["possible"] += batch_obstacle_metrics["counts"]["possible"]
        obstacle_metrics["counts"]["impossible"] += batch_obstacle_metrics["counts"]["impossible"]

        
        obstacle_metrics["valids"]["possible"] += batch_obstacle_metrics["valids"]["possible"]
        obstacle_metrics["valids"]["impossible"] += batch_obstacle_metrics["valids"]["impossible"]

    # Return per-sample lists and counters. Caller will compute mean/std as desired.
    return {
        "validity_list": validity_list,
        "cost_factors": cost_factors,
        "pred_path_costs": pred_path_costs,
        "optimal_path_costs": optimal_path_costs,
        "expansion_ratios": expansion_ratios,
        "hardness_list": hardness_list,
        "collisions": collisions,
        "total_samples": total_samples,
        "obstacle_metrics": obstacle_metrics,
        "path_confidences": path_confidences,
        "voxel_confidences": voxel_confidences,
        "opt_planning_expansions": opt_planning_expansions,
    }


def compute_path_metrics(
    obstacle_grids: np.ndarray,
    path_preds: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    desired_obstacle_distance: float = 1.0,
    min_hardness: float = 1.05,
    astar: bool = False,
    restrict_search_space: bool = False,
    mhastar: bool = False,
):
    # collect per-sample metrics for this batch
    validity_list = []
    path_confidences = []
    voxel_confidences = []
    opt_planning_expansions = []
    cost_factors = []
    expansion_ratios = []
    pred_path_costs = []
    optimal_path_costs = []
    hardness_list = []
    collisions = 0
    total = 0

    def _make_obstacle_bucket():
        return {"avg_distances": [], "avoidance_ratios": [], "full_avoidance": []}

    obstacle_metrics = {
        "categories": {
            "all": _make_obstacle_bucket(),
            "possible": _make_obstacle_bucket(),
            "impossible": _make_obstacle_bucket(),
        },
        "counts": {"possible": 0, "impossible": 0},
        "valids": {"possible": 0, "impossible": 0}
    }

    for i, (obstacle_grid, path_pred, start, end) in enumerate(zip(obstacle_grids, path_preds, starts, ends)):
        total += 1

        # Binary predicted path mask
        path_mask = (path_pred >= 0.5)
        # Always ensure start/end are part of the allowed subset
        path_mask[start[0], start[1]] = True
        path_mask[end[0], end[1]] = True
        # count collisions: any predicted path pixel overlapping an obstacle
        collision_map = np.logical_and(path_mask, obstacle_grid)
        if np.any(collision_map):
            collisions += 1
            continue

        path_mask[collision_map] = 0.0  # mask out possible collisions for pathfinding'

        # Full traversable map (True=free, False=obstacle)
        full_free = ~obstacle_grid

        if astar:
            pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_astar_2D(
                path_mask,
                start[0],
                start[1],
                end[0],
                end[1],
            )
        else:
            if mhastar:
                pred_path_waypoints, a_star_path_cost, a_star_path_expansions, expansions_map = run_mhastar_2D(
                    full_free if not restrict_search_space else path_mask,
                    100 * (1-np.copy(path_pred)),
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    w1 = 3.5, #3.2,
                    w2 = 5.0  #5.0,
                )

            else:
                pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_focal_search_2D(
                    full_free if not restrict_search_space else path_mask,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    w=2.0,
                    path_proximity_map=-np.copy(path_pred) + 1,
                )

        a_star_waypoints, a_star_grid_cost, a_star_grid_expansions, _ = run_astar_2D(
            full_free, start[0], start[1], end[0], end[1]
        )

        hardness_i = a_star_grid_cost / octile_distance(start[0], start[1], end[0], end[1])

        # skip 'easy' samples
        if hardness_i < min_hardness: 
            continue

        # record hardness for this sample
        hardness_list.append(hardness_i)

        if np.any(obstacle_grid):
            dist_to_obstacles = distance_transform_edt(~obstacle_grid)
        else:
            dist_to_obstacles = np.full(obstacle_grid.shape, np.inf, dtype=np.float32)
        enlarged_obstacles = dist_to_obstacles < (desired_obstacle_distance - 1e-8)
        enlarged_free = ~enlarged_obstacles

        avoidance_possible = False
        if enlarged_free[start[0], start[1]] and enlarged_free[end[0], end[1]]:
            avoidance_path, _, _, _ = run_astar_2D(
                enlarged_free,
                start[0],
                start[1],
                end[0],
                end[1]
            )
            avoidance_possible = len(avoidance_path) > 0

        category_key = "possible" if avoidance_possible else "impossible"
        obstacle_metrics["counts"][category_key] += 1

        if len(pred_path_waypoints) > 0:
            # valid path
            # cost factor should be >= 1.0 with smaller is better: achieved / optimal
            cost_factor_i = a_star_path_cost / max(a_star_grid_cost, 1e-8)
            expansion_ratio_i = a_star_path_expansions / a_star_grid_expansions

            path_confidence_i = np.sum(path_pred[pred_path_waypoints[:, 0], pred_path_waypoints[:, 1]]) / pred_path_waypoints.shape[0]
            voxel_confidence_i = np.mean(path_pred)

            opt_planning_expansions_i = a_star_path_expansions / pred_path_waypoints.shape[0]

            obstacle_metrics["valids"][category_key] += 1

            validity_list.append(1)
            cost_factors.append(cost_factor_i)
            # record raw costs as well
            pred_path_costs.append(a_star_path_cost)
            optimal_path_costs.append(a_star_grid_cost)
            expansion_ratios.append(expansion_ratio_i)
            path_confidences.append(path_confidence_i)
            voxel_confidences.append(voxel_confidence_i)
            opt_planning_expansions.append(opt_planning_expansions_i)

            path_coords = np.asarray(pred_path_waypoints, dtype=np.intp)
            path_waypoint_mask = np.zeros_like(obstacle_grid, dtype=bool)
            path_waypoint_mask[path_coords[:, 0], path_coords[:, 1]] = True
            path_waypoint_mask[start[0], start[1]] = True
            path_waypoint_mask[end[0], end[1]] = True

            avg_obst_dist, avoidance_ratio, is_full_avoidance = compute_obstacle_avoidance_metrics_from_thresholded_path(
                path_waypoint_mask,
                obstacle_grid,
                desired_obstacle_distance,
            )

            for category in ("all", category_key):
                bucket = obstacle_metrics["categories"][category]
                bucket["avg_distances"].append(avg_obst_dist)
                bucket["avoidance_ratios"].append(avoidance_ratio)
                bucket["full_avoidance"].append(1.0 if is_full_avoidance else 0.0)

        else:
            # valid flag 0 for hard-enough sample without valid path
            validity_list.append(0)

    return (
        validity_list,
        cost_factors,
        expansion_ratios,
        hardness_list,
        collisions,
        total,
        obstacle_metrics,
        pred_path_costs,
        optimal_path_costs,
        path_confidences,
        voxel_confidences,
        opt_planning_expansions,
    )
