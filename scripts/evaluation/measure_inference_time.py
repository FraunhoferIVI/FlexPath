"""measure_inference_time_minimal

Minimal runtime measurement on a dataset:
- model forward inference time
- additional planner time (focal search by default, or A* with --astar)
- combined time

Outputs
- Pretty summary printed to stdout.
- JSON written to <exp_dir>/evaluation_results/inference_time_<timestamp>.json
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from src.evaluation.metrics.general import (
    load_actor_from_experiment,
    load_dataset_via_factory,
)
from src.evaluation.metrics.inference_time import measure_inference_time
from src.utils.common import seed_everything

logger = logging.getLogger(__name__)


def fmt(m, s):
    if m is None:
        return "N/A"
    return f"{m:.6f} ± {s:.6f}"


def fmt_ms(m, s):
    if m is None:
        return "N/A"
    return f"{m * 1000.0:.3f} ± {s * 1000.0:.3f} ms"


def mean_std(arr: np.ndarray):
    if arr is None:
        return (None, None)
    a = np.asarray(arr)
    if a.size == 0:
        return (None, None)
    return (float(np.mean(a)), float(np.std(a)))


def main():
    parser = argparse.ArgumentParser(description="Measure model + planning inference runtime on a dataset")
    parser.add_argument("exp_dir", help="Experiment directory containing config.yaml and checkpoints")
    parser.add_argument("dataset_config", help="Path to dataset config yaml used by DatasetFactory")
    parser.add_argument("--use_best", action="store_true", help="Load best.pth instead of latest.pth")
    parser.add_argument("--device", default=None, help="Torch device to use (auto-detected if omitted)")
    parser.add_argument("--max_samples", type=int, default=int(1e16), help="Maximum samples to run")
    parser.add_argument("--astar", action="store_true", help="Use A* on prediction instead of focal search")
    parser.add_argument("--wastar", action="store_true", help="Use WA* on prediction instead of focal search")
    parser.add_argument("--min_hardness", type=float, default=1.05, help="Filter out samples with hardness below this value")
    parser.add_argument("--restrict_search_space", action="store_true", help="Restrict planner to predicted mask")
    args = parser.parse_args()

    seed_everything()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from experiment: %s", args.exp_dir)
    model = load_actor_from_experiment(args.exp_dir, device=device, use_best=args.use_best)

    logger.info("Loading dataset via DatasetFactory from config: %s", args.dataset_config)
    try:
        dataset = load_dataset_via_factory(args.dataset_config, split="test")
    except Exception:
        logger.info("Dataset does not have a test split, falling back to validation split.")
        dataset = load_dataset_via_factory(args.dataset_config, split="validation")

    results = measure_inference_time(
        model,
        dataset,
        device=device,
        max_samples=args.max_samples,
        min_hardness=args.min_hardness,
        astar=args.astar,
        wastar=args.wastar,
        restrict_search_space=args.restrict_search_space,
    )

    m = results["metrics"]
    inf = m["inference_time_s"]
    plan = m["planning_time_s"]
    focal_no_nn = m["focal_no_nn_time_s"]
    comb = m["combined_time_s"]
    hard = m["hardness"]

    lines = []
    lines.append("Inference timing results")
    lines.append("========================")
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Experiment dir: {args.exp_dir}")
    lines.append(f"Dataset config: {args.dataset_config}")
    lines.append("")
    lines.append(f"Total samples processed: {results['total_samples']}")
    lines.append(f"Timed samples (after hardness filter): {results['timed_samples']}")
    lines.append(f"Hardness filter min_hardness: {args.min_hardness}")
    lines.append("")
    lines.append("Metric (mean ± std):")
    lines.append(f"  Inference time: {fmt_ms(inf['mean'], inf['std'])} (n={inf['count']})")
    lines.append(f"  Planning time:  {fmt_ms(plan['mean'], plan['std'])} (n={plan['count']})")
    lines.append(f"  Focal time (no NN heuristic): {fmt_ms(focal_no_nn['mean'], focal_no_nn['std'])} (n={focal_no_nn['count']})")
    lines.append(f"  Combined time:  {fmt_ms(comb['mean'], comb['std'])} (n={comb['count']})")
    lines.append(f"  Inference time (seconds): {fmt(inf['mean'], inf['std'])}")
    lines.append(f"  Planning time (seconds):  {fmt(plan['mean'], plan['std'])}")
    lines.append(f"  Focal time no NN (seconds):  {fmt(focal_no_nn['mean'], focal_no_nn['std'])}")
    lines.append(f"  Combined time (seconds):  {fmt(comb['mean'], comb['std'])}")
    lines.append(f"  Hardness: {fmt(hard['mean'], hard['std'])} (n={hard['count']})")

    # Extract cost and expansion metrics produced during inference measurement
    cost_factors = np.array(results.get("cost_factors", []))
    expansion_ratios = np.array(results.get("expansion_ratios", []))
    cost_mean, cost_std = mean_std(cost_factors)
    exp_mean, exp_std = mean_std(expansion_ratios)
    m["cost_factor"] = {"mean": cost_mean, "std": cost_std, "count": int(cost_factors.size)}
    m["expansion_ratio"] = {"mean": exp_mean, "std": exp_std, "count": int(expansion_ratios.size)}

    lines.append(f"  Cost factor (achieved / optimal): {fmt(cost_mean, cost_std)}")
    lines.append(f"  Expansion ratio (path expansions ratio): {fmt(exp_mean, exp_std)}")

    pretty = "\n".join(lines)
    print(pretty)

    out = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_dir": args.exp_dir,
        "dataset_config": args.dataset_config,
        "total_samples": int(results["total_samples"]),
        "timed_samples": int(results["timed_samples"]),
        "combined_inside_model": bool(results["combined_inside_model"]),
        "metrics": m,
        "command_line": {
            "use_best": bool(args.use_best),
            "device": args.device,
            "max_samples": int(args.max_samples),
            "astar": bool(args.astar),
            "wastar": bool(args.wastar),
            "min_hardness": float(args.min_hardness),
            "restrict_search_space": bool(args.restrict_search_space),
        },
    }

    out_dir = Path(args.exp_dir) / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inference_time_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Wrote inference timing results to: %s", out_path)


if __name__ == "__main__":
    main()
