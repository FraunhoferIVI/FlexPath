# FlexPath 2D

## Environment

Install dependencies: 

```bash
uv sync
```

## Download Checkpoints:

```bash
curl -L -o experiments.zip https://owncloud.fraunhofer.de/index.php/s/1OkjP39nWU7bM3w/download
unzip experiments.zip
```

# Download Datasets

```bash
mkdir data
cd data

# TMP (512/64/64k)
curl -L -o TMP_640k_rgb.zip https://owncloud.fraunhofer.de/index.php/s/Md4nGgqQCGrUii8/download
unzip TMP_640k_rgb.zip

# CSM 1.9k (test only)
curl -L -o csm_1900.zip https://owncloud.fraunhofer.de/index.php/s/OdaJnfZwU1NGu4Z/download
unzip csm_1900.zip

# Starcraft 6k (test only)
curl -L -o starcraft_6k.zip https://owncloud.fraunhofer.de/index.php/s/51Dc4CPEwtztXYd/download
unzip starcraft_6k.zip
```

## Retrain Our Model:

**Pretraining:**

```bash
uv run python scripts/pretraining/train_actor.py --config-name config_actor_ours
```

This will create an experiment directory in experiments/pretraining containing checkpoints, training logs and tensorboard stats.

**Finetuning:**

Use the experiment directory of the pretraining, for the training config we provide config files for every objective in configs/training.

```bash
uv run python scripts/training/train_singleshot_dr.py \
  model.single_shot.actor.experiment_dir=<path to pretraining dir> \
  training=<name of training config file>
```

## Retraining Baselines

Launch retraining with:

```bash
uv run python scripts/pretraining/train_actor.py --config-name config_actor_neuralastar
uv run python scripts/pretraining/train_actor.py --config-name config_actor_iastar
uv run python scripts/pretraining/train_actor.py --config-name config_actor_daastar_baseline
```

Overwriting common parameters:

```bash
uv run python scripts/pretraining/train_actor.py \
  --config-name config_actor_iastar \
  training.batch_size=300 \
  training.num_workers=20 \
  training.num_epochs=60
```

Each run writes to:

```text
experiments/pretraining/<timestamp>_actor_<model_name>/
```

Important outputs:

- `config.yaml`: resolved run config.
- `training.log`: training log when file logging is enabled.
- `checkpoints/best.pth`: best checkpoint.
- `checkpoints/latest.pth`: latest checkpoint.
- `logs/`: TensorBoard files.

## Evaluating Model Quality

We use retrained models for all baselines except TransPath, this is crucial for fairness as model capacity (Neural A* ~0.4M, iA ~30M params) and assumed edge cost (Neural A* and DAA* use uniform costs in all 8 directions) differ significantly between those models. We use the TransPath checkpoint as it has only few more parameters (~20% more) than our model and follows the same cost assumption and dataset.

For evaluating the TransPath checkpoint:

1. Download from https://github.com/AIRI-Institute/TransPath/blob/main/weights/focal.pth
2. Save to baselines/transPath/weights

Use `compute_eval_metrics_singleshot.py` with an experiment directory and a dataset config path. The loader expects `config.yaml` plus `checkpoints/best.pth` or `checkpoints/latest.pth`.

```bash
uv run python scripts/evaluation/compute_eval_metrics_singleshot.py \
  experiments/pretraining/<neuralastar_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --iastar

uv run python scripts/evaluation/compute_eval_metrics_singleshot.py \
  experiments/pretraining/<iastar_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --iastar

uv run python scripts/evaluation/compute_eval_metrics_singleshot.py \
  experiments/pretraining/<daastar_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --daastar

uv run python scripts/evaluation/compute_eval_metrics_singleshot.py \
  baselines/transPath/weights \
  configs/pretraining/data/TMP_640k.yaml \
  --transPath
```

Useful options:

- `--max_samples N`: evaluate a subset.
- `--device cuda`: force a device.
- `--min_hardness 1.05`: filter easy samples from metric aggregation.
- `--astar`, `--iastar`, `--daastar`: choose the inference/planning path.
- `--use_uniform_step_cost`: evaluate with uniform movement costs.
- `--restrict_search_space`: restrict planning to the predicted mask.
- `--waypoints`: enable waypoint-aware evaluation.
- `--semantic_obstacle`: add semantic obstacle metrics.

Results are printed and written as JSON to:

```text
<exp_dir>/evaluation_results/eval_<UTC timestamp>.json
```

The JSON includes validity, confidence, cost factor, predicted and optimal path costs, expansion ratio, optimal-found ratio, hardness, collisions, and obstacle avoidance metrics.

## Measuring Inference Time

Neural A* Planning Backend:

```bash
uv run python scripts/evaluation/measure_inference_time.py \
  experiments/pretraining/<neuralastar_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --use_no_rgb \
  --iastar

uv run python scripts/evaluation/measure_inference_time.py \
  experiments/pretraining/<iastar_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --iastar

uv run python scripts/evaluation/measure_inference_time.py \
  experiments/pretraining/<daastar_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --use_no_rgb \
  --daastar

uv run python scripts/evaluation/measure_inference_time.py \
  experiments/runs/<our_run> \
  configs/pretraining/data/TMP_640k.yaml \
  --ours \
  --use_ours_with_diff_astar
  
```

Focal Search:

```bash
uv run python scripts/evaluation/measure_inference_time.py \
  experiments/runs/<our_run> \
  configs/pretraining/data/TMP_640k.yaml

uv run python scripts/evaluation/measure_inference_time.py \
  baselines/transPath/weights \
  configs/pretraining/data/TMP_640k.yaml \
  --transPath
```

Timing results are printed and written to:

```text
<exp_dir>/evaluation_results/inference_time_<UTC timestamp>.json
```

## Obstacle Avoidance Metrics

`compute_eval_metrics_singleshot.py` computes obstacle-avoidance metrics for every evaluated model. Set the clearance target with:

```bash
--desired_obstacle_distance 2.0
```

A desired_obstacle_distance of 2.0 aims for one free pixel between path and obstacle.

To compute the same metrics for the hand-coded cost map obstacle avoiding A* planner, run:

```bash
uv run python scripts/evaluation/compute_eval_metrics_singleshot.py \
  baselines/obstacle_avoiding_astar \
  configs/pretraining/data/TMP_640k.yaml \
  --obstacle_avoiding_astar \
  --desired_obstacle_distance 2.0
```

For semantic obstacle reporting, add:

```bash
--semantic_obstacle
```

The obstacle section reports counts for samples where full clearance is possible or impossible, validity in each group, average clearance distance, avoidance ratio, and full-avoidance rate.
