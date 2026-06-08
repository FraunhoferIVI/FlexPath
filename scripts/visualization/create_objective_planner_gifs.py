#!/usr/bin/env python3
"""Create planner expansion GIFs for each single-shot objective.

The defaults mirror the visualization notebooks in ``notebooks/``:

- shortest paths: ``basemodel_mindist_640k`` on ``TMP_640k``
- obstacle avoidance: ``basemodel_obstacleavoid_dist2_640k`` on ``TMP_640k``
- semantic avoidance: ``basemodel_semanticavoidance_640k`` on semantic TMP
- waypoints: ``basemodel_waypoints_640k`` on waypoint TMP

For every selected sample, the script writes one GIF per objective and one
combined GIF with all objectives in a single row.
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from cstar.pathfinding import run_astar_2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics.general import (
    binary_obstacle_from_image,
    load_actor_from_experiment,
    load_dataset_via_factory,
    octile_distance,
)
from src.utils.swap_color_encoding import (
    swap_obst,
    swap_obstlevels,
    swap_optimal,
    swap_waypoint,
)


Coord = Tuple[int, int]


@dataclass(frozen=True)
class ObjectiveConfig:
    key: str
    title: str
    exp_dir: str
    dataset_config: str
    planner: str
    swap_name: str


@dataclass
class SampleAnimation:
    objective: ObjectiveConfig
    sample_idx: int
    frames: List[np.ndarray]
    hardness: float
    path_length: int
    expansions: int


DEFAULT_OBJECTIVES: Tuple[ObjectiveConfig, ...] = (
    ObjectiveConfig(
        key="mindist",
        title="Shortest paths",
        exp_dir="experiments/runs/basemodel_mindist_640k",
        dataset_config="configs/data/TMP_640k.yaml",
        planner="focal",
        swap_name="optimal",
    ),
    ObjectiveConfig(
        key="obstacle",
        title="Obstacle avoidance",
        exp_dir="experiments/runs/basemodel_obstacleavoid_dist2_640k",
        dataset_config="configs/data/TMP_640k.yaml",
        planner="mhastar",
        swap_name="obstacle",
    ),
    ObjectiveConfig(
        key="semantic",
        title="Semantic avoidance",
        exp_dir="experiments/runs/basemodel_semanticavoidance_640k",
        dataset_config="configs/data/TMP_640k_semantic_obstacle.yaml",
        planner="mhastar",
        swap_name="semantic",
    ),
    ObjectiveConfig(
        key="waypoint",
        title="Waypoints",
        exp_dir="experiments/runs/basemodel_waypoints_640k",
        dataset_config="configs/data/TMP_640k_waypoints.yaml",
        planner="waypoint",
        swap_name="waypoint",
    ),
)

SWAP_FNS = {
    "optimal": swap_optimal,
    "obstacle": swap_obst,
    "semantic": swap_obstlevels,
    "waypoint": swap_waypoint,
}

NEIGHBORS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create objective planner GIFs with incremental expansion overlays."
    )
    parser.add_argument("--k", type=int, default=2, help="Number of samples per objective.")
    parser.add_argument("--split", default="test", help="Dataset split to try first.")
    parser.add_argument("--fallback-split", default="validation", help="Fallback split.")
    parser.add_argument("--output-dir", default="outputs/objective_gifs", help="Where GIFs are written.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to CUDA when available.")
    parser.add_argument("--use-best", action="store_true", help="Load best.pth instead of latest.pth.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for display.")
    parser.add_argument("--min-hardness", type=float, default=0.0, help="Skip samples below this hardness.")
    parser.add_argument("--max-scan", type=int, default=200, help="Maximum dataset items scanned per objective.")
    parser.add_argument("--expansions-per-frame", type=int, default=2, help="Number of planner expansions advanced per frame. Shared across objectives.")
    parser.add_argument("--final-hold-seconds", type=float, default=2.0, help="Seconds to hold the final combined frame after all objectives finish.")
    parser.add_argument("--separator-width", type=int, default=8, help="Separator width between combined panels.")
    parser.add_argument("--fps", type=float, default=12.0, help="GIF frames per second.")
    parser.add_argument("--scale", type=int, default=8, help="Pixel scale for each 64x64 map panel.")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=[cfg.key for cfg in DEFAULT_OBJECTIVES],
        choices=[cfg.key for cfg in DEFAULT_OBJECTIVES],
        help="Objective keys to render.",
    )
    return parser.parse_args()


def extract_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out


def sample_to_numpy(sample: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, Coord, Coord]:
    img_tensor = sample["images"]
    start_arr = sample["starts"].cpu().numpy().astype(np.int64)
    end_arr = sample["ends"].cpu().numpy().astype(np.int64)
    img_np = torch.clamp(img_tensor.unsqueeze(0) * 255, 0, 255).cpu().numpy()
    img_np_uint8 = img_np.astype(np.uint8).transpose(0, 2, 3, 1)
    obstacle_grid = binary_obstacle_from_image(img_np_uint8)[0]
    return img_np_uint8[0], obstacle_grid, (int(start_arr[0]), int(start_arr[1])), (int(end_arr[0]), int(end_arr[1]))


def predict_path_map(model: torch.nn.Module, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
    logits = torch.tanh(extract_logits(out))
    path_pred = ((logits + 1.0) / 2.0).detach().cpu().numpy()[0]
    if path_pred.ndim == 3:
        path_pred = path_pred[0]
    return np.asarray(path_pred, dtype=np.float32)


def find_color(img: np.ndarray, color: Sequence[int]) -> Optional[Coord]:
    target = np.asarray(color, dtype=np.uint8)
    coords = np.argwhere(np.all(img == target[None, None, :], axis=-1))
    if coords.shape[0] == 0:
        return None
    y, x = coords[0]
    return int(y), int(x)


def compute_hardness(free: np.ndarray, start: Coord, end: Coord) -> float:
    _, cost, _, _ = run_astar_2D(free, start[0], start[1], end[0], end[1])
    direct = max(float(octile_distance(start[0], start[1], end[0], end[1])), 1e-8)
    return float(cost) / direct


def octile_grid(shape: Tuple[int, int], goal: Coord) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    dy = np.abs(yy - goal[0])
    dx = np.abs(xx - goal[1])
    return np.maximum(dx, dy) + (math.sqrt(2.0) - 1.0) * np.minimum(dx, dy)


def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def instrumented_astar(
    free: np.ndarray,
    start: Coord,
    goal: Coord,
    heuristic_grid: Optional[np.ndarray] = None,
    heuristic_weight: float = 1.0,
    proximity_cost: Optional[np.ndarray] = None,
    proximity_weight: float = 0.0,
) -> Tuple[List[Coord], List[Coord]]:
    h_grid = octile_grid(free.shape, goal)
    if heuristic_grid is not None:
        h_grid = heuristic_weight * h_grid + np.asarray(heuristic_grid, dtype=np.float32)
    else:
        h_grid = heuristic_weight * h_grid

    open_heap: List[Tuple[float, float, Coord]] = []
    heapq.heappush(open_heap, (float(h_grid[start]), 0.0, start))
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    expanded_order: List[Coord] = []
    closed = set()
    h, w = free.shape

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded_order.append(current)
        if current == goal:
            return reconstruct_path(came_from, current), expanded_order

        cy, cx = current
        for dy, dx, step_cost in NEIGHBORS:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not free[ny, nx]:
                continue
            neighbor = (ny, nx)
            bias = 0.0 if proximity_cost is None else proximity_weight * float(proximity_cost[ny, nx])
            tentative_g = current_g + step_cost + bias
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score = tentative_g + float(h_grid[ny, nx])
            heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return [], expanded_order


def run_planner(
    cfg: ObjectiveConfig,
    free: np.ndarray,
    path_pred: np.ndarray,
    start: Coord,
    end: Coord,
    waypoint: Optional[Coord],
) -> Tuple[List[Coord], List[Coord]]:
    proximity = 1.0 - np.copy(path_pred)
    if cfg.planner == "focal":
        return instrumented_astar(
            free,
            start,
            end,
            heuristic_weight=1.0,
            proximity_cost=proximity,
            proximity_weight=10.0,
        )
    if cfg.planner == "mhastar":
        restricted = np.logical_and(free, path_pred >= 0.5)
        restricted[end[0], end[1]] = True
        restricted[end[1], end[0]] = True
        heuristic = 5.0 * proximity
        return instrumented_astar(
            restricted,
            start,
            end,
            heuristic_grid=heuristic,
            heuristic_weight=1.0,
            proximity_cost=proximity,
            proximity_weight=10.0,
        )
    if cfg.planner == "waypoint":
        if waypoint is None:
            return instrumented_astar(
                free,
                start,
                end,
                heuristic_weight=1.0,
                proximity_cost=proximity,
                proximity_weight=10.0,
            )
        path_a, expanded_a = instrumented_astar(
            free,
            start,
            waypoint,
            heuristic_weight=1.0,
            proximity_cost=proximity,
            proximity_weight=10.0,
        )
        path_b, expanded_b = instrumented_astar(
            free,
            waypoint,
            end,
            heuristic_weight=1.0,
            proximity_cost=proximity,
            proximity_weight=10.0,
        )
        path = path_a + path_b[1:] if path_a and path_b else path_a or path_b
        return path, expanded_a + expanded_b
    raise ValueError(f"Unknown planner: {cfg.planner}")


def coords_to_mask(coords: Iterable[Coord], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    coords_list = list(coords)
    if coords_list:
        arr = np.asarray(coords_list, dtype=np.int64)
        mask[arr[:, 0], arr[:, 1]] = True
    return mask


def render_overlay(
    base_rgb: np.ndarray,
    path_mask: np.ndarray,
    expansion_mask: np.ndarray,
    start: Coord,
    end: Coord,
    waypoint: Optional[Coord],
) -> np.ndarray:
    canvas = base_rgb.copy().astype(np.int16)
    expanded_only = np.logical_and(expansion_mask, ~path_mask)
    canvas[expanded_only] = np.array([0, 127, 0]).astype(np.int16)
    canvas[path_mask] = np.array([127, 0, 0], dtype=np.int16)
    canvas[start[0], start[1]] = np.array([255, 255, 76], dtype=np.int16)
    canvas[end[0], end[1]] = np.array([255, 255, 76], dtype=np.int16)
    if waypoint is not None:
        canvas[waypoint[0], waypoint[1]] = np.array([255, 255, 76], dtype=np.int16)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def apply_swap(cfg: ObjectiveConfig, img: np.ndarray) -> np.ndarray:
    return SWAP_FNS[cfg.swap_name](img, dot_lines=False)


def scale_frame(img: np.ndarray, scale: int) -> np.ndarray:
    panel = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.Resampling.NEAREST)
    return np.asarray(panel, dtype=np.uint8)


def expansion_steps(total: int, expansions_per_frame: int) -> List[int]:
    if total <= 0:
        return [0]
    stride = max(1, int(expansions_per_frame))
    steps = list(range(0, total + 1, stride))
    if steps[-1] != total:
        steps.append(total)
    return steps


def make_frames(
    cfg: ObjectiveConfig,
    base_rgb: np.ndarray,
    path: List[Coord],
    expanded: List[Coord],
    start: Coord,
    end: Coord,
    waypoint: Optional[Coord],
    hardness: float,
    expansions_per_frame: int,
    scale: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    shape = base_rgb.shape[:2]
    path_mask = coords_to_mask(path, shape)
    if path:
        path_mask[start[0], start[1]] = True
        path_mask[end[0], end[1]] = True
    for step in expansion_steps(len(expanded), expansions_per_frame):
        expansion_mask = coords_to_mask(expanded[:step], shape)
        visible_path = path_mask if step >= len(expanded) else np.zeros(shape, dtype=bool)
        overlay = render_overlay(base_rgb, visible_path, expansion_mask, start, end, waypoint)
        overlay = apply_swap(cfg, overlay)
        frames.append(scale_frame(overlay, scale))
    return frames


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    duration_ms = max(1, int(round(1000.0 / fps)))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def pad_to_shape(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.full((height, width, 3), 255, dtype=np.uint8)
    out[: frame.shape[0], : frame.shape[1]] = frame[:, :, :3]
    return out


def synchronized_length(frames_by_objective: Sequence[Sequence[np.ndarray]], fps: float, final_hold_seconds: float) -> int:
    max_len = max(len(frames) for frames in frames_by_objective)
    hold_frames = max(0, int(round(final_hold_seconds * fps)))
    return max_len + hold_frames


def freeze_to_length(frames: Sequence[np.ndarray], total_len: int) -> List[np.ndarray]:
    if not frames:
        raise ValueError("Cannot synchronize an empty frame sequence.")
    frozen = [np.asarray(frame, dtype=np.uint8) for frame in frames[:total_len]]
    if len(frozen) < total_len:
        frozen.extend([frozen[-1].copy() for _ in range(total_len - len(frozen))])
    return frozen


def combine_row(
    synced_frames_by_objective: Sequence[Sequence[np.ndarray]],
    separator_width: int,
) -> List[np.ndarray]:
    max_h = max(frame.shape[0] for frames in synced_frames_by_objective for frame in frames)
    max_w = max(frame.shape[1] for frames in synced_frames_by_objective for frame in frames)
    separator = np.full((max_h, max(0, separator_width), 3), 230, dtype=np.uint8)
    row_frames: List[np.ndarray] = []
    for i in range(len(synced_frames_by_objective[0])):
        panels = []
        for panel_idx, frames in enumerate(synced_frames_by_objective):
            if panel_idx > 0 and separator_width > 0:
                panels.append(separator)
            panels.append(pad_to_shape(frames[i], max_h, max_w))
        row_frames.append(np.concatenate(panels, axis=1))
    return row_frames


def load_dataset_with_fallback(cfg: ObjectiveConfig, split: str, fallback_split: str):
    try:
        return load_dataset_via_factory(cfg.dataset_config, split=split), split
    except Exception:
        return load_dataset_via_factory(cfg.dataset_config, split=fallback_split), fallback_split


def create_objective_animations(
    cfg: ObjectiveConfig,
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> List[SampleAnimation]:
    model = load_actor_from_experiment(cfg.exp_dir, device=device, use_best=args.use_best)
    model.eval()
    dataset, _ = load_dataset_with_fallback(cfg, args.split, args.fallback_split)
    animations: List[SampleAnimation] = []
    scanned = 0
    for idx in range(len(dataset)):
        if len(animations) >= args.k or scanned >= args.max_scan:
            break
        scanned += 1
        sample = dataset[idx]
        base_rgb, obstacle_grid, start, end = sample_to_numpy(sample)
        free = ~obstacle_grid
        hardness = compute_hardness(free, start, end)
        if hardness < args.min_hardness:
            continue
        path_pred = predict_path_map(model, sample["images"], device)
        pred_mask = path_pred >= args.threshold
        pred_mask[start[0], start[1]] = True
        pred_mask[end[0], end[1]] = True
        pred_mask[np.logical_and(pred_mask, obstacle_grid)] = False
        waypoint = find_color(base_rgb, [255, 255, 76]) if cfg.planner == "waypoint" else None
        path, expanded = run_planner(cfg, free, path_pred, start, end, waypoint)
        frames = make_frames(
            cfg,
            base_rgb,
            path,
            expanded,
            start,
            end,
            waypoint,
            hardness,
            args.expansions_per_frame,
            args.scale,
        )
        anim = SampleAnimation(
            objective=cfg,
            sample_idx=idx,
            frames=frames,
            hardness=hardness,
            path_length=len(path),
            expansions=len(expanded),
        )
        animations.append(anim)

    if len(animations) < args.k:
        print(f"warning: {cfg.key} produced {len(animations)} samples after scanning {scanned}.")
    return animations


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    selected = [cfg for cfg in DEFAULT_OBJECTIVES if cfg.key in set(args.objectives)]
    all_animations: Dict[str, List[SampleAnimation]] = {}
    print(f"device={device} output_dir={output_dir}")
    for cfg in selected:
        print(f"rendering {cfg.key}: {cfg.exp_dir} on {cfg.dataset_config}")
        all_animations[cfg.key] = create_objective_animations(cfg, args, device, output_dir)

    row_count = min(len(anims) for anims in all_animations.values()) if all_animations else 0
    for ordinal in range(row_count):
        raw_frames_by_objective = [all_animations[cfg.key][ordinal].frames for cfg in selected]
        total_len = synchronized_length(raw_frames_by_objective, args.fps, args.final_hold_seconds)
        synced_frames_by_objective = [freeze_to_length(frames, total_len) for frames in raw_frames_by_objective]
        sample_bits = "_".join(f"{cfg.key}{all_animations[cfg.key][ordinal].sample_idx:05d}" for cfg in selected)

        for cfg, frames in zip(selected, synced_frames_by_objective):
            anim = all_animations[cfg.key][ordinal]
            out_path = output_dir / cfg.key / f"{cfg.key}_{anim.sample_idx:05d}_synced.gif"
            save_gif(frames, out_path, args.fps)
            print(f"wrote {out_path} ({len(frames)} frames)")

        row_frames = combine_row(synced_frames_by_objective, separator_width=args.separator_width)
        row_path = output_dir / "combined" / f"objectives_row_sample_{ordinal:03d}_{sample_bits}.gif"
        save_gif(row_frames, row_path, args.fps)
        print(f"wrote {row_path} ({len(row_frames)} frames)")

    print("done")


if __name__ == "__main__":
    main()
