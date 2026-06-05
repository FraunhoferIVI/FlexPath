"""visualize_paths

Iterate over a dataset and save RGB PNG visualizations for:
1) raw predicted path mask (no planner)
2) focal-search path using model prediction as heuristic

Samples with hardness below `--min_hardness` are skipped.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cstar.pathfinding import run_astar_2D, run_focal_search_2D, run_mhastar_2D
from src.evaluation.metrics.general import (
    load_actor_from_experiment,
    binary_obstacle_from_image,
    octile_distance,
)
from src.models.actor.transPath.autoencoder import Autoencoder

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def _extract_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out


def _waypoints_to_mask(waypoints: np.ndarray, shape, start, end) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if waypoints is not None and len(waypoints) > 0:
        waypoints = np.asarray(waypoints, dtype=np.int64)
        mask[waypoints[:, 0], waypoints[:, 1]] = True
    mask[start[0], start[1]] = True
    mask[end[0], end[1]] = True
    return mask


def _render_overlay(base_rgb: np.ndarray, obstacle_grid: np.ndarray, path_mask: np.ndarray, expansion_mask: np.ndarray, start, end) -> np.ndarray:
    """Create RGB visualization.

    Colors:
    - free space: from base image
    - obstacles: dark blue tint
    - path: red
    - start: green
    - goal: yellow
    """
    canvas = base_rgb.copy().astype(np.int64)

    # path in red
    canvas[path_mask] += np.array([127, 0, 0], dtype=np.int64)

    canvas[np.logical_and(expansion_mask, ~path_mask)] += np.array([0, 127, 0], dtype=np.int64)

    # start / goal
    sy, sx = int(start[0]), int(start[1])
    gy, gx = int(end[0]), int(end[1])
    canvas[sy, sx] = np.array([255, 255, 76], dtype=np.int64)
    canvas[gy, gx] = np.array([255, 255, 76], dtype=np.int64)

    return np.clip(canvas, a_min=0, a_max=255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize predicted and planner paths over a dataset")
    parser.add_argument("exp_dir", help="Experiment directory containing config.yaml and checkpoints")
    parser.add_argument("input_dir", help="Path to folder with png images of maps.")
    parser.add_argument("output_dir", help="Directory where PNG files are saved")
    parser.add_argument("--use_best", action="store_true", help="Load best.pth instead of latest.pth")
    parser.add_argument("--device", default=None, help="Torch device to use (auto-detected if omitted)")
    parser.add_argument("--max_samples", type=int, default=int(1e16), help="Maximum number of dataset samples to iterate")
    parser.add_argument("--min_hardness", type=float, default=1.05, help="Skip samples with hardness below this value")
    parser.add_argument("--transPath", action="store_true", help="Load transPath pretrained checkpoint instead")
    parser.add_argument("--restrict_search_space", action="store_true", help="Restrict focal search to predicted mask")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for predicted path mask")
    parser.add_argument("--save_gradient_maps", action="store_true", help="Save gradient maps.")
    parser.add_argument("--mhastar", action="store_true", help="Use MHA*.")
    parser.add_argument("--waypoints", action="store_true", help="Use waypoints.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.transPath:
        logger.info("Loading model from hardcoded transPath checkpoint path: baselines/transPath/weights/focal.pth")
        model = Autoencoder(in_channels=2, rgb=False)
        model.load_state_dict(torch.load("baselines/transPath/weights/focal.pth", weights_only=True))
        model = model.to(device=device)
        args.exp_dir = "baselines/transPath/"
    else:
        logger.info(f"Loading model from experiment: {args.exp_dir}")
        model = load_actor_from_experiment(args.exp_dir, device=device, use_best=args.use_best)

    model.eval()

    if args.save_gradient_maps:
        from src.differentiable_reward.rewards.obstacle_objective import reward_obstacle_with_cost_penalty
        reward_f = reward_obstacle_with_cost_penalty

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predicted_path"
    focal_dir = out_dir / "focal_with_prediction_heuristic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    focal_dir.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    saved = 0
    skipped_hardness = 0

    start_encoding = [255, 76, 76]
    end_encoding = [76, 255, 76]

    files = [f for f in Path(args.input_dir).iterdir()
         if f.is_file() and f.suffix.lower() == ".png"]

    for idx, file in tqdm(enumerate(files), total=len(files)):

        basemap = np.asarray(Image.open(file))

        if total_seen >= args.max_samples:
            break

        total_seen += 1

        start = np.argwhere(np.logical_and(np.logical_and(basemap[:, :, 0] == start_encoding[0], basemap[:, :, 1] == start_encoding[1]), basemap[:, :, 2] == start_encoding[2])).astype(np.int64)[0]
        end = np.argwhere(np.logical_and(np.logical_and(basemap[:, :, 0] == end_encoding[0], basemap[:, :, 1] == end_encoding[1]), basemap[:, :, 2] == end_encoding[2])).astype(np.int64)[0]

        img_tensor = torch.from_numpy(basemap).permute(2, 0, 1).unsqueeze(0).float() / 255.0 # [1, C, H, W], normalized

        if args.waypoints:
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

        img_np = torch.clamp(img_tensor * 255, min=0, max=255).cpu().numpy()
        img_np_uint8 = img_np.astype(np.uint8).transpose(0, 2, 3, 1)  # [1, H, W, C]
        base_rgb = img_np_uint8[0]

        obstacle_grid = binary_obstacle_from_image(img_np_uint8)[0]
        full_free = ~obstacle_grid

        _, a_star_grid_cost, _, _ = run_astar_2D(full_free, start[0], start[1], end[0], end[1])
        heuristic = max(float(octile_distance(start[0], start[1], end[0], end[1])), 1e-8)
        hardness_i = float(a_star_grid_cost) / heuristic
        if hardness_i < args.min_hardness:
            skipped_hardness += 1
            continue

        if args.save_gradient_maps:
            X = img_tensor.to(device=device, non_blocking=True)
            out = model(X)
        else:
            with torch.no_grad():
                X = img_tensor.to(device=device, non_blocking=True)
                out = model(X)

        logits_raw = _extract_logits(out)


        if not args.transPath:
            logits = torch.nn.functional.tanh(logits_raw)
        else:
            logits = logits_raw

        preds_activated = (logits + 1.0) / 2.0
        path_pred = preds_activated.detach().cpu().numpy()[0]
        logits_raw = logits_raw.detach().cpu().numpy()[0]

        if path_pred.ndim == 3:
            path_pred = path_pred[0]
            logits_raw = logits_raw[0]

        pred_path_mask = path_pred >= args.threshold
        pred_path_mask[start[0], start[1]] = True
        pred_path_mask[end[0], end[1]] = True

        collision_map = np.logical_and(pred_path_mask, obstacle_grid)
        pred_path_mask[collision_map] = False

        planning_space = pred_path_mask if args.restrict_search_space else full_free

        if args.mhastar:
            focal_waypoints, _, _, expansion_mask = run_mhastar_2D(
                planning_space,
                100 * (1-np.copy(path_pred)),
                start[0],
                start[1],
                end[0],
                end[1],
                w1 = 3.5,   #3.2,
                w2 = 5.0,   #5.0,
            )
        else:
            if args.waypoints:
                focal_waypoints0, _, _, expansion_mask0 = run_focal_search_2D(
                    planning_space,
                    start[0],
                    start[1],
                    wps_numpy[0, 0],
                    wps_numpy[0, 1],
                    w=2.0,
                    path_proximity_map=(-np.copy(path_pred) + 1.0),
                )
                focal_waypoints1, _, _, expansion_mask1 = run_focal_search_2D(
                    planning_space,
                    wps_numpy[0, 0],
                    wps_numpy[0, 1],
                    end[0],
                    end[1],
                    w=2.0,
                    path_proximity_map=(-np.copy(path_pred) + 1.0),
                )

                focal_waypoints = np.concat((focal_waypoints0, focal_waypoints1), axis=0)
                expansion_mask = np.logical_or(expansion_mask0, expansion_mask1)
                
            else:
                focal_waypoints, _, _, expansion_mask = run_focal_search_2D(
                    planning_space,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    w=2.0,
                    path_proximity_map=(-np.copy(path_pred) + 1.0),
                )
        focal_mask = _waypoints_to_mask(focal_waypoints, pred_path_mask.shape, start, end)

        pred_rgb = _render_overlay(base_rgb, obstacle_grid, pred_path_mask, np.zeros_like(expansion_mask), start, end)
        focal_rgb = _render_overlay(base_rgb, obstacle_grid, focal_mask, expansion_mask, start, end)

        stem = f"sample_{idx:07d}_hardness_{hardness_i:.4f}"

        if args.save_gradient_maps:
            logits.retain_grad()
            reward_vec, _ = reward_f(X, logits.unsqueeze(0), torch.zeros_like(logits.unsqueeze(0)), pixel_sum_penalty_scale=0.2)
            loss = reward_vec.mean()
            loss.backward()

            plt.figure(figsize=(6, 6))
            plt.imshow(logits[0].cpu().detach().numpy(), cmap="coolwarm")
            plt.axis("off")  # remove axes

            plt.savefig(
                pred_dir / f"{stem}_gradient_heatmap.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300
            )
            plt.close()

        Image.fromarray(np.clip(path_pred * 255, a_min=0, a_max=255).astype(np.uint8)).save(pred_dir / f"{stem}_prob_heatmap.png")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(logits_raw, cmap="coolwarm")
        plt.axis("off")  # remove axes

        plt.savefig(
            pred_dir / f"{stem}_logits_heatmap.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300
        )
        plt.close()

        Image.fromarray(pred_rgb).save(pred_dir / f"{stem}.png")
        Image.fromarray(focal_rgb).save(focal_dir / f"{stem}.png")
        Image.fromarray(focal_mask).save(focal_dir / f"{stem}_path.png")
        saved += 1

    logger.info(f"Finished. Seen={total_seen}, saved={saved}, skipped_low_hardness={skipped_hardness}")
    print(f"Saved images to: {out_dir}")
    print(f"Seen={total_seen}, saved={saved}, skipped_low_hardness={skipped_hardness}")


if __name__ == "__main__":
    main()
