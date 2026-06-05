"""General evaluation helpers.

Utilities to load a pretrained actor from an experiment directory, load a
dataset (huggingface style), iterate over samples, run the model to obtain
predictions and extract a binary obstacle occupancy grid from images using
the project's normalized color encoding.

This module re-uses the existing loading helpers in
`src.utils.checkpoint_loader` and the dataset conventions used elsewhere in
the codebase.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import Iterator, Optional, Tuple
from types import SimpleNamespace

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
from cstar.pathfinding import run_astar_2D, run_focal_search_2D, run_astar_2D_heuristic_grid
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
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint[state_dict_key].items()}

            # Load actor model
            logger.info("Creating model...")
            model = get_actor_from_config(cfg)  # Updated to use unified registry

            model.load_state_dict(state_dict)

            return model.to(device=device)

    model = checkpoint_loader.load_pretrained_actor_from_experiment(exp_dir, use_best=use_best, device=device)
    
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
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
    transPath: bool = False,
    daastar: bool = False,
    iastar: bool = False,
    desired_obstacle_distance: float = 1.0,
    astar: bool = False,
    restrict_search_space: bool = False,
    mhastar: bool = False,
    output_logits: bool = True,
    waypoints: bool = False,
    semantic_obstacles: bool = False,
    use_uniform_step_cost: bool = False,
) -> Iterator[Tuple[int, torch.Tensor, np.ndarray, np.ndarray]]:

    """
    Iterate over dataset samples, run model inference and yield results.

    Yields tuples of (index, input_tensor, prediction_np, obstacle_mask_np).

    - input_tensor: the tensor passed to the model (on device if device provided)
    - prediction_np: numpy array prediction (H x W) of predicted class or binary mask
    - obstacle_mask_np: boolean numpy array (H x W) marking obstacles

    Notes:
    - The function tries to be robust with dataset element formats. It looks
      for a top-level 'image' field (typical HF dataset). If the image is a
      PIL.Image it will be converted to a numpy uint8 array.
    - By default the preprocess step converts uint8 HxWx3 -> float32 CxHxW
      normalized to [0,1]. If you need a different normalization (e.g. mean/std)
      pass a custom preprocess_fn.

    """

    assert not (transPath and iastar), "Only one out of transPath and iastar flags can be set."
    assert not (daastar and iastar), "Only one out of daastar and iastar flags can be set."

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
    semantic_full_avoidance_at4 = {
        "all": [],
        "possible": [],
        "impossible": [],
    }

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
        batch_size=256,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    def _daastar_infer(model_module: torch.nn.Module, images: torch.Tensor) -> SimpleNamespace:
        obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=images.device)
        obstacle_map = torch.sigmoid((images - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        walkable_map = (~(obstacle_map < 1.51)).float()

        start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=images.device)
        start_map = torch.sigmoid((images - start_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        start_map = (start_map < 1.51).float()

        end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)
        end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        end_map = (end_map < 1.51).float()

        outputs, _ = model_module.infer(
            walkable_map,
            start_map,
            end_map,
            prob_maps=None,
        )
        return SimpleNamespace(logits=outputs.paths.squeeze(1), hidden_states=outputs.histories.squeeze(1))  # still has C dim

    for idx, sample in tqdm(enumerate(dl)):
        if max_samples is not None and idx >= max_samples:
            break

        img_tensor = sample["images"]
        starts = sample["starts"].cpu().numpy()
        ends = sample["ends"].cpu().numpy()

        if waypoints:
            wp_color = torch.tensor([1.0, 1.0, 76 / 255], device=img_tensor.device)  # WP
            wp_map = torch.sigmoid((img_tensor - wp_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
            wp_map = (wp_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

            B, _, H, W = wp_map.shape

            # Flatten H,W dimensions
            flat_maps = wp_map.view(B, -1)  # [B, H*W]

            # Get linear indices of the '1's
            indices = flat_maps.argmax(dim=1)  # [B]

            # Convert linear index to 2D coordinates
            y = indices // W
            x = indices % W

            wps_numpy = torch.stack([y, x], dim=1).cpu().numpy()  # [B, 2] as (y,x)

        elif semantic_obstacles:
            semobst_color = torch.tensor([100 / 255, 100 / 255, 1.0], device=img_tensor.device)  # semantic obstacle
            semobst_map = torch.sigmoid((img_tensor - semobst_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
            semobst_map = semobst_map < 1.51  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        img_np = torch.clamp(img_tensor * 255, min=0, max=255).cpu().numpy()

        # keep a uint8 copy for exact color checks
        img_np_uint8 = img_np.astype(np.uint8).transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            if daastar:
                out = _daastar_infer(model, img_tensor)
            else:
                out = model(img_tensor)
            # out = model.inference(img_tensor, N=4)  # use inference API instead of forward for reasoning models

        # obstacle grid from original uint8 image
        obstacle_masks = binary_obstacle_from_image(img_np_uint8)

        if iastar or daastar:

            paths = out.logits.cpu().detach().numpy()
            expansion_maps = out.hidden_states.cpu().detach().numpy()

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
                batch_semantic_full_avoidance_at4,
            ) = compute_path_metrics_iastar(
                obstacle_grids=obstacle_masks,
                path_preds=paths,
                expansion_maps=expansion_maps,
                starts=starts,
                ends=ends,
                desired_obstacle_distance=desired_obstacle_distance,
                min_hardness=min_hardness,
                astar=astar,
                mhastar=mhastar,
                semantic_obstacles=semobst_map if semantic_obstacles else None,
                use_uniform_step_cost=use_uniform_step_cost,
            )

        else:
            # model outputs vary across model implementations. Prefer the common
            # attribute 'logits' (SemanticSegmenterOutput), otherwise attempt to
            # use the raw return value.
            logits = out.logits

            if output_logits:
                if not transPath:
                    # transPath squashes values inside of model
                    logits = torch.nn.functional.tanh(logits)
                
                preds_activated = (logits + 1.0) / 2.0
            else:
                preds_activated = logits

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
                batch_semantic_full_avoidance_at4,
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
                waypoints=wps_numpy if waypoints else None,
                semantic_obstacles=semobst_map if semantic_obstacles else None,
                use_uniform_step_cost=use_uniform_step_cost,
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
        for category in semantic_full_avoidance_at4.keys():
            semantic_full_avoidance_at4[category].extend(batch_semantic_full_avoidance_at4.get(category, []))
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
        "semantic_full_avoidance_at4": semantic_full_avoidance_at4,
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
    waypoints = None,
    semantic_obstacles = None,
    use_uniform_step_cost: bool = False,
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
    semantic_full_avoidance_at4 = {
        "all": [],
        "possible": [],
        "impossible": [],
    }
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
            # continue

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
                use_uniform_step_cost=use_uniform_step_cost
            )
        else:
            if mhastar:
                # pred_path_waypoints, a_star_path_cost, a_star_path_expansions, expansions_map = run_mhastar_2D(
                #     full_free if not restrict_search_space else path_mask,
                #     100 * (1-np.copy(path_pred)),
                #     start[0],
                #     start[1],
                #     end[0],
                #     end[1],
                #     w1 = 3.5,   #3.2,
                #     w2 = 5.0,   #5.0,
                #     use_uniform_step_cost=use_uniform_step_cost,
                # )
                # pred_path_waypoints, a_star_path_cost, a_star_path_expansions, expansions_map = run_mhastar_2D(
                # 	full_free if not restrict_search_space else path_mask,
                # 	0.5 * (-np.copy(path_pred) + path_pred.max()),
                # 	start[0],
                # 	start[1],
                # 	end[0],
                # 	end[1],
                # 	w1 = 1.2, #3.2,
                # 	w2 = 1.1  #5.0,
                # )

                pred_path_waypoints, a_star_path_cost, a_star_path_expansions, expansions_map = run_astar_2D_heuristic_grid(
                	full_free if not restrict_search_space else path_mask,
                	start[0],
                	start[1],
                	end[0],
                	end[1],
                	np.copy(path_pred),
                	w=15.0
                )

                # pred_path_waypoints, a_star_path_cost, a_star_path_expansions, expansions_map = run_astar_2D_heuristic_grid(
                # 	~np.astype(path_pred[1], np.bool),
                # 	np.copy(path_pred[0]),
                # 	start[0],
                # 	start[1],
                # 	end[0],
                # 	end[1],
                # 	w=15.0
                # )

                # path_mask = path_pred[0]
                # path_pred = path_mask

            else:
                if waypoints is not None:
                    wp = waypoints[i]

                    pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_focal_search_2D(
                        full_free if not restrict_search_space else path_mask,
                        start[0],
                        start[1],
                        wp[0],
                        wp[1],
                        w=2.0,
                        path_proximity_map=-np.copy(path_pred) + 1,
                        use_uniform_step_cost=use_uniform_step_cost,
                    )
                    if len(pred_path_waypoints) > 0:
                        pred_path_waypoints_2, a_star_path_cost_2, a_star_path_expansions_2, _ = run_focal_search_2D(
                            full_free if not restrict_search_space else path_mask,
                            wp[0],
                            wp[1],
                            end[0],
                            end[1],
                            w=2.0,
                            path_proximity_map=-np.copy(path_pred) + 1,
                            use_uniform_step_cost=use_uniform_step_cost,
                        )

                        if len(pred_path_waypoints_2) > 0:
                            pred_path_waypoints = np.concat((pred_path_waypoints, pred_path_waypoints_2), axis=0)
                            a_star_path_cost = a_star_path_cost + a_star_path_cost_2
                            a_star_path_expansions = a_star_path_expansions + a_star_path_expansions_2
                        else:
                            pred_path_waypoints = pred_path_waypoints_2

                else:
                    pred_path_waypoints, a_star_path_cost, a_star_path_expansions, _ = run_focal_search_2D(
                        full_free if not restrict_search_space else path_mask,
                        start[0],
                        start[1],
                        end[0],
                        end[1],
                        w=2.0,
                        path_proximity_map=-np.copy(path_pred) + 1,
                        use_uniform_step_cost=use_uniform_step_cost,
                    )

        a_star_waypoints, a_star_grid_cost, a_star_grid_expansions, _ = run_astar_2D(
            full_free, start[0], start[1], end[0], end[1], use_uniform_step_cost=use_uniform_step_cost
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
                end[1],
                use_uniform_step_cost=use_uniform_step_cost,
            )
            avoidance_possible = len(avoidance_path) > 0

        category_key = "possible" if avoidance_possible else "impossible"
        obstacle_metrics["counts"][category_key] += 1


        semantic_category_key = None
        semantic_grid_i = None
        dist_to_semantic_obstacles = None
        if semantic_obstacles is not None:
            semantic_grid_i = semantic_obstacles[i]
            if torch.is_tensor(semantic_grid_i):
                semantic_grid_i = semantic_grid_i.cpu().numpy()
            semantic_grid_i = np.asarray(semantic_grid_i, dtype=bool)
            if semantic_grid_i.ndim == 3:
                semantic_grid_i = semantic_grid_i[0]

            if np.any(semantic_grid_i):
                dist_to_semantic_obstacles = distance_transform_edt(~semantic_grid_i)
            else:
                dist_to_semantic_obstacles = np.full(semantic_grid_i.shape, np.inf, dtype=np.float32)

            semantic_enlarged_obstacles = dist_to_semantic_obstacles < (4.0 - 1e-8)
            semantic_enlarged_free = ~semantic_enlarged_obstacles

            semantic_avoidance_possible = False
            if semantic_enlarged_free[start[0], start[1]] and semantic_enlarged_free[end[0], end[1]]:
                semantic_avoidance_path, _, _, _ = run_astar_2D(
                    semantic_enlarged_free,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    use_uniform_step_cost=use_uniform_step_cost,
                )
                semantic_avoidance_possible = len(semantic_avoidance_path) > 0

            semantic_category_key = "possible" if semantic_avoidance_possible else "impossible"

        if len(pred_path_waypoints) > 0:
            # valid path
            # cost factor should be >= 1.0 with smaller is better: achieved / optimal
            cost_factor_i = a_star_path_cost / max(a_star_grid_cost, 1e-8)
            expansion_ratio_i = a_star_path_expansions / a_star_grid_expansions

            path_confidence_i = np.sum(path_pred[pred_path_waypoints[:, 0], pred_path_waypoints[:, 1]]) / pred_path_waypoints.shape[0]
            voxel_confidence_i = np.mean(path_pred)

            opt_planning_expansions_i = a_star_path_expansions / a_star_waypoints.shape[0]

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

            if semantic_obstacles is not None:
                min_semantic_dist = float(np.min(dist_to_semantic_obstacles[path_coords[:, 0], path_coords[:, 1]]))
                semantic_full_avoidance_value = 1.0 if min_semantic_dist >= (4.0 - 1e-8) else 0.0
                semantic_full_avoidance_at4["all"].append(semantic_full_avoidance_value)
                semantic_full_avoidance_at4[semantic_category_key].append(semantic_full_avoidance_value)

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

            # save some visual debugging artifacts for high cost factor samples
            # if expansion_ratio_i <= 1.00:
            #     from PIL import Image

            #     img = np.zeros(shape=(64, 64, 3))

            #     img[obstacle_grid] = [76, 76, 255]

            #     for x, y in pred_path_waypoints:
            #         img[x, y, 0] += 255

            #     Image.fromarray((img).clip(min=0, max=255).astype(np.uint8)).save(f"imgs/{expansion_ratio_i}.png")
        else:
            # valid flag 0 for hard-enough sample without valid path
            validity_list.append(0)
            if semantic_obstacles is not None:
                semantic_full_avoidance_at4["all"].append(0.0)
                semantic_full_avoidance_at4[semantic_category_key].append(0.0)

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
        semantic_full_avoidance_at4,
    )
            


def compute_path_metrics_iastar(
    obstacle_grids: np.ndarray,
    path_preds: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    expansion_maps: np.ndarray,
    desired_obstacle_distance: float = 1.0,
    min_hardness: float = 1.05,
    astar: bool = False,
    mhastar: bool = False,
    waypoints = None,
    semantic_obstacles = None,
    use_uniform_step_cost: bool = False,
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
    semantic_full_avoidance_at4 = {
        "all": [],
        "possible": [],
        "impossible": [],
    }
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
    
    # print(path_preds.dtype, expansion_maps.dtype)
    # print(path_preds.sum(axis=(1, 2)), expansion_maps.sum(axis=(1, 2)))
    # print("Uniques: ", np.unique(path_preds), np.unique(expansion_maps))

    for i, (obstacle_grid, path_pred, expansion_map, start, end) in enumerate(zip(obstacle_grids, path_preds, expansion_maps, starts, ends)):
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
            # continue

        path_mask[collision_map] = 0.0  # mask out possible collisions for pathfinding

        # Full traversable map (True=free, False=obstacle)
        full_free = ~obstacle_grid

        pred_path_waypoints, a_star_path_cost, _, _ = run_astar_2D(
            path_mask, start[0], start[1], end[0], end[1], use_uniform_step_cost=use_uniform_step_cost
        )
        
        a_star_path_expansions = np.sum(expansion_map)

        # A* on full free-space map
        # a_star_waypoints, a_star_grid_cost, a_star_grid_expansions = run_focal_search_2D(
        # 	full_free, start[0], start[1], end[0], end[1], w=1.0
        # )

        a_star_waypoints, a_star_grid_cost, a_star_grid_expansions, _ = run_astar_2D(
            full_free, start[0], start[1], end[0], end[1], use_uniform_step_cost=use_uniform_step_cost
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
                end[1],
                use_uniform_step_cost=use_uniform_step_cost
            )
            avoidance_possible = len(avoidance_path) > 0

        category_key = "possible" if avoidance_possible else "impossible"
        obstacle_metrics["counts"][category_key] += 1

        semantic_category_key = None
        semantic_grid_i = None
        dist_to_semantic_obstacles = None
        if semantic_obstacles is not None:
            semantic_grid_i = semantic_obstacles[i]
            if torch.is_tensor(semantic_grid_i):
                semantic_grid_i = semantic_grid_i.cpu().numpy()
            semantic_grid_i = np.asarray(semantic_grid_i, dtype=bool)
            if semantic_grid_i.ndim == 3:
                semantic_grid_i = semantic_grid_i[0]

            if np.any(semantic_grid_i):
                dist_to_semantic_obstacles = distance_transform_edt(~semantic_grid_i)
            else:
                dist_to_semantic_obstacles = np.full(semantic_grid_i.shape, np.inf, dtype=np.float32)

            semantic_enlarged_obstacles = dist_to_semantic_obstacles < (4.0 - 1e-8)
            semantic_enlarged_free = ~semantic_enlarged_obstacles

            semantic_avoidance_possible = False
            if semantic_enlarged_free[start[0], start[1]] and semantic_enlarged_free[end[0], end[1]]:
                semantic_avoidance_path, _, _, _ = run_astar_2D(
                    semantic_enlarged_free,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    use_uniform_step_cost=use_uniform_step_cost
                )
                semantic_avoidance_possible = len(semantic_avoidance_path) > 0

            semantic_category_key = "possible" if semantic_avoidance_possible else "impossible"

        if len(pred_path_waypoints) > 0:
            # valid path
            # cost factor should be >= 1.0 with smaller is better: achieved / optimal
            cost_factor_i = a_star_path_cost / max(a_star_grid_cost, 1e-8)
            expansion_ratio_i = a_star_path_expansions / a_star_grid_expansions

            path_confidence_i = np.sum(path_pred[pred_path_waypoints[:, 0], pred_path_waypoints[:, 1]]) / pred_path_waypoints.shape[0]
            voxel_confidence_i = np.mean(path_pred)

            validity_list.append(1)
            cost_factors.append(cost_factor_i)
            # record raw costs as well
            pred_path_costs.append(a_star_path_cost)
            optimal_path_costs.append(a_star_grid_cost)
            expansion_ratios.append(expansion_ratio_i)

            obstacle_metrics["valids"][category_key] += 1

            path_coords = np.asarray(pred_path_waypoints, dtype=np.intp)
            path_waypoint_mask = np.zeros_like(obstacle_grid, dtype=bool)
            path_waypoint_mask[path_coords[:, 0], path_coords[:, 1]] = True
            path_waypoint_mask[start[0], start[1]] = True
            path_waypoint_mask[end[0], end[1]] = True
            path_confidences.append(path_confidence_i)
            voxel_confidences.append(voxel_confidence_i)
            opt_planning_expansions.append(a_star_path_expansions / a_star_waypoints.shape[0])

            if semantic_obstacles is not None:
                min_semantic_dist = float(np.min(dist_to_semantic_obstacles[path_coords[:, 0], path_coords[:, 1]]))
                semantic_full_avoidance_value = 1.0 if min_semantic_dist >= (4.0 - 1e-8) else 0.0
                semantic_full_avoidance_at4["all"].append(semantic_full_avoidance_value)
                semantic_full_avoidance_at4[semantic_category_key].append(semantic_full_avoidance_value)

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

            # save some visual debugging artifacts for high cost factor samples
            # if cost_factor_i >= 1.0:
            # 	from PIL import Image

            # 	img = np.zeros(shape=(64, 64, 3))

            # 	img[:, :, 0] = path_pred
            # 	img[:, :, 1] = 0.5 * full_free

            # 	for x, y in pred_path_waypoints:
            # 		img[x, y, 1] += 0.5

            # 	for x, y in a_star_waypoints:
            # 		img[x, y, 2] += 0.5

            # 	Image.fromarray((img * 255).clip(min=0, max=255).astype(np.uint8)).save(f"imgs/{cost_factor_i}.png")
        else:
            # valid flag 0 for hard-enough sample without valid path
            validity_list.append(0)
            if semantic_obstacles is not None:
                semantic_full_avoidance_at4["all"].append(0.0)
                semantic_full_avoidance_at4[semantic_category_key].append(0.0)

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
        semantic_full_avoidance_at4,
    )
            