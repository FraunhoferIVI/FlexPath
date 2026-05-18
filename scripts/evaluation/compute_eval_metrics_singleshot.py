"""compute_eval_metrics_singleshot

Compute evaluation metrics for a single-shot model.

Values computed
- validity: fraction of "hard-enough" samples (hardness >= `--min_hardness`) where a valid
	path was found (mean ± std).
- path_confidence: mean ± std of the confidence assigned to the final path.
- voxel_confidence: mean ± std of confidences across all predicted voxels.
- cost_factor: mean ± std of achieved_cost / optimal_cost (1.0 = optimal).
- predicted_path_cost / optimal_path_cost: mean ± std of predicted and optimal path costs.
- expansion_ratio: mean ± std of expansion ratio (search expansions relative metric).
- hardness: mean ± std of sample hardness values used to filter evaluation.
- optimal_found_ratio: fraction of samples with cost factor ~= 1.0 (within eps).
- obstacle avoidance metrics: per-category avg clearance distance, avoidance ratios and
	full-avoidance statistics, and sample counts.
- collisions and total sample counts.

Outputs
- Pretty-printed summary is printed to stdout.
- A comprehensive JSON file is written into ``<exp_dir>/evaluation_results/`` with filename
	``eval_<UTC-timestamp>.json`` that contains all computed statistics, per-category
	obstacle metrics, and the CLI options used.

Usage
	python scripts/evaluation/compute_eval_metrics_singleshot.py <exp_dir> <dataset_config> [options]

Positional arguments
- exp_dir: Directory of the experiment (must contain config and checkpoints).
- dataset_config: Path to the dataset factory YAML used to load the test (or validation) split.

Important options
- --use_best: Load best.pth instead of latest.pth from the experiment.
- --device: Torch device to use (auto-detected if omitted: CUDA if available, otherwise CPU).
- --max_samples: Maximum number of samples to evaluate (default: very large).
- --astar: Use A* planner when running predictions.
- --mhastar: Use MH-A* planner when running predictions.
- --restrict_search_space: Constrain search to the predicted path mask.
- --desired_obstacle_distance: Clearance target used when computing obstacle-avoidance metrics.
- --min_hardness: Minimum hardness to include a sample in metric computations (default 1.05).

Notes
- The script attempts to load the dataset with split "test" and falls back to "validation"
	if a test split is not available.
- Validity and several other metrics are computed only for the filtered set of "hard-enough"
	samples (see ``--min_hardness``).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging
import argparse

from src.evaluation.metrics.general import load_actor_from_experiment, load_dataset_via_factory, iterate_dataset_and_predict
import numpy as np
import json
from datetime import datetime

logger = logging.getLogger(__name__)


# Helper to compute mean/std safely
def mean_std(arr):
	if arr.size == 0:
		return (None, None)
	return (float(np.mean(arr)), float(np.std(arr)))


def fmt(m, s):
		if m is None:
			return "N/A"
		return f"{m:.6f} ± {s:.6f}"


def main():
	parser = argparse.ArgumentParser(description="Evaluate model predictions on a dataset using DatasetFactory")
	parser.add_argument("exp_dir", help="Experiment directory containing config.yaml and checkpoints")
	parser.add_argument("dataset_config", help="Path to dataset config yaml used by DatasetFactory")
	parser.add_argument("--use_best", action="store_true", help="Load best.pth instead of latest.pth")
	parser.add_argument("--device", default=None, help="Torch device to use (auto-detected if omitted)")
	parser.add_argument("--max_samples", type=int, default=1e16, help="Maximum samples to run")
	parser.add_argument("--astar", action="store_true", help="Use astar on prediction instead of focal search.")
	parser.add_argument("--mhastar", action="store_true", help="Use mhastar on prediction instead of focal search.")
	parser.add_argument("--desired_obstacle_distance", type=float, default=1.0, help="Desired minimum Euclidean distance from obstacles for evaluation metrics")
	parser.add_argument("--min_hardness", type=float, default=1.05, help="Filters out samples with a hardness below this value.")
	parser.add_argument("--restrict_search_space", action="store_true")
	args = parser.parse_args()

	# Load model
	device = args.device
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	logger.info(f"Loading model from experiment: {args.exp_dir}")
	model = load_actor_from_experiment(args.exp_dir, device=device, use_best=args.use_best)

	# Load dataset using DatasetFactory
	logger.info(f"Loading dataset via DatasetFactory from config: {args.dataset_config}")
	try:
		split = "test" if "voxelgym" not in args.dataset_config else "validation"
		dataset = load_dataset_via_factory(args.dataset_config, split=split)
	except Exception:
		logger.info("Dataset does not have a test split, falling back to validation split.")
		dataset = load_dataset_via_factory(args.dataset_config, split="validation")
		
	# Iterate and run predictions
	results = iterate_dataset_and_predict(
		model,
		dataset,
		device=device,
		max_samples=args.max_samples,
		min_hardness=args.min_hardness,
		desired_obstacle_distance=args.desired_obstacle_distance,
		astar=args.astar,
		restrict_search_space=args.restrict_search_space,
		mhastar=args.mhastar,
	)
	# opt_planning_expansions
	# results contains lists and counters
	validity_list = np.array(results["validity_list"]) if "validity_list" in results else np.array([])
	path_confidences = np.array(results["path_confidences"]) if "path_confidences" in results else np.array([])
	voxel_confidences = np.array(results["voxel_confidences"]) if "voxel_confidences" in results else np.array([]) 
	cost_factors = np.array(results["cost_factors"]) if "cost_factors" in results else np.array([])
	expansion_ratios = np.array(results["expansion_ratios"]) if "expansion_ratios" in results else np.array([])
	opt_planning_expansions_ratio = np.array(results["opt_planning_expansions"]) if "opt_planning_expansions" in results else np.array([])
	pred_path_costs = np.array(results["pred_path_costs"]) if "pred_path_costs" in results else np.array([])
	optimal_path_costs = np.array(results["optimal_path_costs"]) if "optimal_path_costs" in results else np.array([])
	hardness_list = np.array(results["hardness_list"]) if "hardness_list" in results else np.array([])
	collisions = results.get("collisions", 0)
	total_samples = results.get("total_samples", 0)

	validity_mean, validity_std = mean_std(validity_list)
	path_confidence_mean, path_confidence_std = mean_std(path_confidences)
	voxel_confidences_mean, voxel_confidences_std = mean_std(voxel_confidences)
	opt_planning_expansions_ratio_mean, opt_planning_expansions_ratio_std = mean_std(opt_planning_expansions_ratio)
	cost_mean, cost_std = mean_std(cost_factors)
	expansion_mean, expansion_std = mean_std(expansion_ratios)
	hardness_mean, hardness_std = mean_std(hardness_list)
	# predicted/optimal path costs mean/std
	pred_cost_mean, pred_cost_std = mean_std(pred_path_costs)
	opt_cost_mean, opt_cost_std = mean_std(optimal_path_costs)

	# New metric: optimal found ratio (cost factor approx 1 within eps)
	eps = 1e-4
	if cost_factors.size > 0:
		optimal_mask = np.isclose(cost_factors, 1.0, atol=eps)
		optimal_ratio = float(np.sum(optimal_mask) / cost_factors.size)
	else:
		optimal_ratio = None

	obstacle_metrics = results.get("obstacle_metrics", {"categories": {}, "counts": {}, "valids": {}})

	def _category_array(category_name: str, metric_key: str) -> np.ndarray:
		return np.array(
			obstacle_metrics.get("categories", {}).get(category_name, {}).get(metric_key, []),
			dtype=np.float32,
		)

	obstacle_category_definitions = (
		("All paths", "all"),
		("Avoidance possible", "possible"),
		("Full avoidance impossible", "impossible"),
	)

	category_stats = {}
	for label, key in obstacle_category_definitions:
		dist_arr = _category_array(key, "avg_distances")
		ratio_arr = _category_array(key, "avoidance_ratios")
		full_arr = _category_array(key, "full_avoidance")
		category_stats[key] = {
			"label": label,
			"avg_distance": mean_std(dist_arr),
			"avoidance_ratio": mean_std(ratio_arr),
			"full_avoidance": mean_std(full_arr),
			"count": int(dist_arr.size),
		}

	# Prepare pretty output
	lines = []
	lines.append("Evaluation results")
	lines.append("==================")
	lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")
	lines.append(f"Experiment dir: {args.exp_dir}")
	lines.append(f"Dataset config: {args.dataset_config}")
	lines.append("")
	lines.append(f"Total samples processed: {total_samples}")
	lines.append(f"Collisions (count): {int(collisions)}")
	lines.append("")
	lines.append("Metric (mean ± std):")

	lines.append(f"  Validity (hard-enough): {fmt(validity_mean, validity_std)}  (fraction of hard samples with valid path)")
	lines.append(f"  Path-confidence: {fmt(path_confidence_mean, path_confidence_std)}  (Avg confidence of final path)")  # voxel_confidences
	lines.append(f"  Voxel-confidence: {fmt(voxel_confidences_mean, voxel_confidences_std)}  (Avg confidence of all voxels)")
	lines.append(f"  Cost factor (achieved / optimal): {fmt(cost_mean, cost_std)}")
	lines.append(f"  Predicted path cost (mean ± std): {fmt(pred_cost_mean, pred_cost_std)}")
	lines.append(f"  Optimal path cost (mean ± std): {fmt(opt_cost_mean, opt_cost_std)}")
	lines.append(f"  Expansion ratio (path expansions ratio): {fmt(expansion_mean, expansion_std)}")
	lines.append(f"  Optimal Expansion factor: {fmt(opt_planning_expansions_ratio_mean, opt_planning_expansions_ratio_std)}")
	lines.append(f"  Hardness (normalized): {fmt(hardness_mean, hardness_std)}")
	lines.append(f"  Optimal found ratio (|cost-optimal_cost| <= {eps}): {optimal_ratio if optimal_ratio is not None else 'N/A'}")
	obstacle_counts = obstacle_metrics.get("counts", {})
	possible_count = int(obstacle_counts.get("possible", 0))
	impossible_count = int(obstacle_counts.get("impossible", 0))
	total_obstacle_samples = possible_count + impossible_count

	obstacle_valids = obstacle_metrics.get("valids", {})
	possible_valids = int(obstacle_valids.get("possible", 0))
	impossible_valids = int(obstacle_valids.get("impossible", 0))

	possible_validity = possible_valids / possible_count if possible_count > 0 else 0.0
	impossible_validity = impossible_valids / impossible_count if impossible_count > 0 else 0.0

	lines.append("")
	lines.append(f"Obstacle avoidance clearance target: {args.desired_obstacle_distance}")
	lines.append(f"  Avoidance possible samples: {possible_count}, impossible: {impossible_count} (total {total_obstacle_samples})")
	lines.append(f"  Avoidance possible validity: impossible: {impossible_validity} possible: {possible_validity}")
	lines.append("Obstacle avoidance metrics (mean ± std):")
	for label, key in obstacle_category_definitions:
		stats = category_stats[key]
		lines.append(
			f"  {stats['label']} (n={stats['count']}): avg dist {fmt(*stats['avg_distance'])}, ratio {fmt(*stats['avoidance_ratio'])}, full avoidance {fmt(*stats['full_avoidance'])}"
		)

	pretty = "\n".join(lines)
	print(pretty)

	# Write a comprehensive JSON file with documentation
	out = {
		"timestamp_utc": datetime.utcnow().isoformat() + "Z",
		"experiment_dir": args.exp_dir,
		"dataset_config": args.dataset_config,
		"notes": "Metrics computed on test or validation split. Validity is computed only for 'hard-enough' samples (hardness>=1.05). Cost factors and expansion ratios are only for valid paths.",
		"eps_optimal_cost": eps,
		"total_samples": int(total_samples),
		"collisions": int(collisions),
		"metrics": {
			"validity": {"mean": validity_mean, "std": validity_std, "count": int(validity_list.size) if validity_list is not None else 0},
			"path_confidence": {"mean": path_confidence_mean, "std": path_confidence_std, "count": int(path_confidences.size) if path_confidences is not None else 0},
			"voxel_confidence": {"mean": voxel_confidences_mean, "std": voxel_confidences_std, "count": int(voxel_confidences.size) if voxel_confidences is not None else 0},
			"cost_factor": {"mean": cost_mean, "std": cost_std, "count": int(cost_factors.size) if cost_factors is not None else 0},
			"predicted_path_cost": {"mean": pred_cost_mean, "std": pred_cost_std, "count": int(pred_path_costs.size) if pred_path_costs is not None else 0},
			"optimal_path_cost": {"mean": opt_cost_mean, "std": opt_cost_std, "count": int(optimal_path_costs.size) if optimal_path_costs is not None else 0},
			"expansion_ratio": {"mean": expansion_mean, "std": expansion_std, "count": int(expansion_ratios.size) if expansion_ratios is not None else 0},
			"opt_expansion_ratio": {"mean": opt_planning_expansions_ratio_mean, "std": opt_planning_expansions_ratio_std, "count": int(opt_planning_expansions_ratio.size) if opt_planning_expansions_ratio is not None else 0},
			"hardness": {"mean": hardness_mean, "std": hardness_std, "count": int(hardness_list.size) if hardness_list is not None else 0},
			"optimal_found_ratio": optimal_ratio,
			"obstacle_avoidance": {
				"desired_obstacle_distance": float(args.desired_obstacle_distance),
				"counts": {
					"possible": possible_count,
					"impossible": impossible_count,
					"total": total_obstacle_samples,
				},
				"categories": {
					key: {
						"label": category_stats[key]["label"],
						"avg_distance": {
							"mean": category_stats[key]["avg_distance"][0],
							"std": category_stats[key]["avg_distance"][1],
							"count": category_stats[key]["count"],
						},
						"avoidance_ratio": {
							"mean": category_stats[key]["avoidance_ratio"][0],
							"std": category_stats[key]["avoidance_ratio"][1],
							"count": category_stats[key]["count"],
						},
						"full_avoidance": {
							"mean": category_stats[key]["full_avoidance"][0],
							"std": category_stats[key]["full_avoidance"][1],
							"count": category_stats[key]["count"],
						},
					} for _, key in obstacle_category_definitions
				}
			},
		},
		"command_line": {
			"use_best": bool(args.use_best),
			"device": args.device,
			"max_samples": int(args.max_samples),
			"desired_obstacle_distance": float(args.desired_obstacle_distance)
		}
	}

	out_dir = Path(args.exp_dir) / "evaluation_results"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"eval_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2)

	logger.info(f"Wrote evaluation results to: {out_path}")


if __name__ == "__main__":
	main()
