import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging
import argparse

from src.evaluation.metrics.general import load_actor_from_experiment, load_dataset_via_factory, iterate_dataset_and_predict
import numpy as np

from src.models.actor.transPath.autoencoder import Autoencoder

import pandas as pd


logger = logging.getLogger(__name__)


# Helper to compute mean/std safely
def mean_std(arr):
    if arr.size == 0:
        return (None, None)
    return (float(np.mean(arr)), float(np.std(arr)))


def fmt(m, s=None):
        if m is None:
            return "N/A"
        elif s is None:
            return f"{round(m, 4)}"
        return f"{round(m, 4)} ± {round(s, 4)}"

dataset_configs = [
    ("TMP", "configs/data/TMP_640k.yaml"),
    ("Voxelgym", "configs/data/actor_data_hf.yaml"),
    ("CSM", "configs/data/csm_1900.yaml"),
    ("SC", "configs/data/starcraft_6k.yaml"),
]

dataset_names = [x[0] for x in dataset_configs]

metrics = [
    "Hard Validity",
    "Path confidence",
    "Cost Factor",
    "Expansion Ratio",
    "Optimal Found Ratio"
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions on a dataset using DatasetFactory")
    parser.add_argument("exp_dir", help="Experiment directory containing config.yaml and checkpoints")
    parser.add_argument("--use_best", action="store_true", help="Load best.pth instead of latest.pth")
    parser.add_argument("--device", default=None, help="Torch device to use (auto-detected if omitted)")
    parser.add_argument("--max_samples", type=int, default=1e16, help="Maximum samples to run")
    parser.add_argument("--transPath", action="store_true", help="Load transPath pretrained checkpoint instead.")
    parser.add_argument("--iastar", action="store_true", help="Use iastar.")
    parser.add_argument("--min_hardness", type=float, default=1.05, help="Filters out samples with a hardness below this value.")
    parser.add_argument("--use_uniform_step_cost", action="store_true", help="Use uniform step costs. ")
    args = parser.parse_args()

    metrics_table = pd.DataFrame("", index=dataset_names, columns=metrics)

    # Load model
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.transPath:
        logger.info("Loading model from hardcoded transPath checkpoint path: baselines/transPath/weights/focal.pth")
        model = Autoencoder(in_channels=2, rgb=False)
        model.load_state_dict(torch.load("baselines/transPath/weights/focal.pth", weights_only=True))
        model = model.to(device="cuda")
        args.exp_dir = "baselines/transPath/"
    else:
        logger.info(f"Loading model from experiment: {args.exp_dir}")
        model = load_actor_from_experiment(args.exp_dir, device=device, use_best=args.use_best)
        
    for name, dataset_config in dataset_configs:

        # model = NeuralAstar(
        # 	encoder_input="m+",
        # 	encoder_arch="CNN",
        # 	encoder_depth=4,
        # 	learn_obstacles=False,
        # 	Tmax=1.0,
        # )
        # model.load_state_dict(
        # 	load_from_ptl_checkpoint(f"neuralastar/model/epoch=33-step=272.ckpt")
        # )

        # model = model.to(device="cuda")

        # args.exp_dir = "neuralastar/"

        # Load dataset using DatasetFactory
        logger.info(f"Loading dataset via DatasetFactory from config: {dataset_config}")
        split = "test" if name != "Voxelgym" else "validation"  # voxelgym has only validation split
        dataset = load_dataset_via_factory(dataset_config, split=split) 

        # Iterate and run predictions
        results = iterate_dataset_and_predict(
            model,
            dataset,
            device=device,
            max_samples=args.max_samples,
            min_hardness=args.min_hardness,
            transPath=args.transPath,
            desired_obstacle_distance=1.0,
            astar=False,
            iastar=args.iastar,
            use_uniform_step_cost=args.use_uniform_step_cost,
        )

        if args.iastar:
            hard_validity_mean, hard_validity_std = -1.0, -1.0  # uses guidance maps, therefore hard validity concept cannot be applied
        
        else:
            results_astar = iterate_dataset_and_predict(
                model,
                dataset,
                device=device,
                max_samples=args.max_samples,
                min_hardness=args.min_hardness,
                transPath=args.transPath,
                desired_obstacle_distance=1.0,
                astar=True,
                iastar=args.iastar,
                use_uniform_step_cost=args.use_uniform_step_cost,
            )
            hard_validity_mean, hard_validity_std = mean_std(np.array(results_astar["validity_list"]) if "validity_list" in results else np.array([]))

        # opt_planning_expansions
        # results contains lists and counters
        path_confidences = np.array(results["path_confidences"]) if "path_confidences" in results else np.array([]) 
        cost_factors = np.array(results["cost_factors"]) if "cost_factors" in results else np.array([])
        expansion_ratios = np.array(results["expansion_ratios"]) if "expansion_ratios" in results else np.array([])

        path_confidence_mean, path_confidence_std = mean_std(path_confidences)
        cost_mean, cost_std = mean_std(cost_factors)
        expansion_mean, expansion_std = mean_std(expansion_ratios)

        # New metric: optimal found ratio (cost factor approx 1 within eps)
        eps = 1e-4
        if cost_factors.size > 0:
            optimal_mask = np.isclose(cost_factors, 1.0, atol=eps)
            optimal_ratio = float(np.sum(optimal_mask) / cost_factors.size)
        else:
            optimal_ratio = None

        metrics_table.loc[name, "Hard Validity"] = fmt(hard_validity_mean)
        metrics_table.loc[name, "Path confidence"] = fmt(path_confidence_mean, path_confidence_std)
        metrics_table.loc[name, "Cost Factor"] = fmt(cost_mean, cost_std)
        metrics_table.loc[name, "Expansion Ratio"] = fmt(expansion_mean, expansion_std)
        metrics_table.loc[name, "Optimal Found Ratio"] = fmt(optimal_ratio)

    metrics_table.to_excel(Path.joinpath(Path(args.exp_dir), Path("OptimalityMetrics.xlsx")), index=True)

if __name__ == "__main__":
    main()
