"""
measure_inference_time

Example:

Focal Search:
- TransPath: python scripts/evaluation/measure_inference_time.py <Some dir> <dataconfig> --transPath
- Ours: python scripts/evaluation/measure_inference_time.py <Ours dir> <dataconfig>

A*:
- TransPath: python scripts/evaluation/measure_inference_time.py <Some dir> <dataconfig> --transPath --astar
- Ours: python scripts/evaluation/measure_inference_time.py <Ours dir> <dataconfig> --astar

Neural A* Search Backend:
- DAA*: python scripts/evaluation/measure_inference_time.py <DAA* dir> <dataconfig> --use_no_rgb --daastar
- Neural A*: python scripts/evaluation/measure_inference_time.py <Neural A* dir> <dataconfig> --use_no_rgb --iastar
- iA*: python scripts/evaluation/measure_inference_time.py <iA* dir> <dataconfig> --iastar
- Ours: python scripts/evaluation/measure_inference_time.py <Ours dir> <dataconfig> --ours --use_ours_with_diff_astar


Measure runtime on a dataset:
- model forward inference time
- additional planner time (focal search by default, or A* with ``--astar``)
- combined time

For iA* / NeuralA* style models, planning is considered part of forward runtime,
so the planner time is reported as 0 and combined==inference.

Outputs
- Pretty summary printed to stdout.
- JSON written to ``<exp_dir>/evaluation_results/inference_time_<timestamp>.json``.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import argparse
import torch

import json
from datetime import datetime

from src.evaluation.metrics.general import load_actor_from_experiment, load_dataset_via_factory
from src.evaluation.metrics.inference_time import measure_inference_time
import numpy as np
from src.models.actor.transPath.autoencoder import Autoencoder

from src.models.actor.neuralastar.astar import NeuralAstar
from src.models.actor.neuralastar.planner_module import load_from_ptl_checkpoint

from src.evaluation.diffastarModule import NeuralAstarModule
from src.models.actor.daastar.training import DAAStarPlannerModule
from src.utils.daastar_util import parse_args

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
    parser.add_argument("--transPath", action="store_true", help="Load transPath pretrained checkpoint instead")
    parser.add_argument("--astar", action="store_true", help="Use A* on prediction instead of focal search")
    parser.add_argument("--wastar", action="store_true", help="Use WA* on prediction instead of focal search")
    parser.add_argument("--iastar", action="store_true", help="Treat model as iA* (planning included in forward)")
    parser.add_argument("--daastar", action="store_true", help="Treat model as DAA* (planning included in forward)")
    parser.add_argument("--min_hardness", type=float, default=1.05, help="Filter out samples with hardness below this value")
    parser.add_argument("--restrict_search_space", action="store_true", help="Restrict planner to predicted mask")
    parser.add_argument("--use_ours_with_diff_astar", action="store_true", help="Use differentiable A* planning module on top of prediction of our model")
    parser.add_argument("--use_no_rgb", action="store_true", help="")
    parser.add_argument("--ours", action="store_true", help="")
    parser.add_argument("--use_daastar_checkpoint", action="store_true", help="Use DAA* baseline checkpoint")
    parser.add_argument("--use_neuralastar_checkpoint", action="store_true", help="Use Neural A* checkpoint. ")
    args = parser.parse_args()

    if args.use_ours_with_diff_astar and args.iastar:
        raise ValueError("--use_ours_with_diff_astar cannot be combined with --transPath or --iastar")
    if args.daastar and args.iastar:
        raise ValueError("--daastar cannot be combined with --iastar")
    if args.use_daastar_checkpoint and not args.daastar:
        raise ValueError("--use_daastar_checkpoint requires --daastar")

    seed_everything()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.transPath:
        logger.info("Loading model from hardcoded transPath checkpoint path: baselines/transPath/weights/focal.pth")
        model = Autoencoder(in_channels=2, rgb=False)
        model.load_state_dict(torch.load("baselines/transPath/weights/focal.pth", weights_only=True))
        model = model.to(device=device)
        args.exp_dir = "baselines/transPath/"
    elif args.use_daastar_checkpoint:
        logger.info("Loading model from hardcoded daastar checkpoint path: baselines/daastar/model/epoch=44-step=347040.ckpt")
        config = parse_args("baselines/daastar/model/full_config.yaml")
        model = DAAStarPlannerModule(config)
        model.planner.load_transpath_pretrained_model("baselines/daastar/model/epoch=44-step=347040.ckpt")
        model = model.to(device=device)
        args.exp_dir = "baselines/daastar/"
    elif args.use_neuralastar_checkpoint:
        logger.info("Loading model from hardcoded Neural A* checkpoint path: baselines/neuralastar/model/epoch=33-step=272.ckpt")
        model = NeuralAstar(encoder_arch="checkpoint")
        state_dict = load_from_ptl_checkpoint("baselines/neuralastar/model/epoch=33-step=272.ckpt")
        model.load_state_dict(state_dict)
        model = model.to(device="cuda")
        args.exp_dir = "baselines/neuralastar/"
    else:
        logger.info(f"Loading model from experiment: {args.exp_dir}")
        model = load_actor_from_experiment(args.exp_dir, device=device, use_best=args.use_best)

    classname = str(model.__class__)
    # set g_ratio to tuned inference value, this is required for a fair comparison as elsewise every model has a vastly different tradeoff of eploration to optimality
    # example: Neural A* with g_ratio of 0.5 has costfactor ~1.0015 and expansion ratio ~0.7 (almost A* behavior)
    print(classname)
    if "NeuralAstar" in classname:
        model.g_ratio = 0.25
        model.astar.g_ratio = 0.25
    elif "iastar" in classname:
        model.g_ratio = 0.4
        model.dastar.g_ratio = 0.4
    elif "daastar" in classname:
        model.planner.astar.config.enable_angle = False
        model.planner.astar.g_ratio_act = None
        model.planner.astar.infer_optimality = True

    if args.use_ours_with_diff_astar:
        logger.info("Wrapping loaded model with NeuralAstarModule for differentiable A* planning")
        model = NeuralAstarModule(model).to(device=device)


    logger.info(f"Loading dataset via DatasetFactory from config: {args.dataset_config}")
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
        transPath=args.transPath,
        daastar=args.daastar,
        iastar=args.iastar,
        astar=args.astar,
        wastar=args.wastar,
        restrict_search_space=args.restrict_search_space,
        use_ours_with_diff_astar=args.use_ours_with_diff_astar,
        use_rgb=not args.use_no_rgb,
        ours=args.ours
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
    lines.append(f"Use NeuralAstarModule wrapper: {args.use_ours_with_diff_astar}")
    lines.append(f"Planning included in model forward: {results['combined_inside_model']}")
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
        "notes": "For iA*/NeuralA* models, planning time is included in model forward; planning_time is therefore 0.",
        "total_samples": int(results["total_samples"]),
        "timed_samples": int(results["timed_samples"]),
        "combined_inside_model": bool(results["combined_inside_model"]),
        "metrics": m,
        "command_line": {
            "use_best": bool(args.use_best),
            "device": args.device,
            "max_samples": int(args.max_samples),
            "transPath": bool(args.transPath),
            "daastar": bool(args.daastar),
            "astar": bool(args.astar),
            "wastar": bool(args.wastar),
            "iastar": bool(args.iastar),
            "min_hardness": float(args.min_hardness),
            "restrict_search_space": bool(args.restrict_search_space),
            "use_ours_with_diff_astar": bool(args.use_ours_with_diff_astar),
            "use_daastar_checkpoint": bool(args.use_daastar_checkpoint),
        },
    }

    out_dir = Path(args.exp_dir) / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inference_time_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Wrote inference timing results to: {out_path}")
    
if __name__ == "__main__":
    main()
