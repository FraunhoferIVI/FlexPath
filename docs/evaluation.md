# Evaluation guide

This guide covers the single-shot evaluation script and its inputs and outputs.

## Usage

The evaluation script requires an experiment directory and a dataset config
(YAML). It loads the model from the experiment and evaluates on the test split
if available, otherwise it falls back to validation.

```bash
python scripts/evaluation/compute_eval_metrics_singleshot.py \
  <exp_dir> \
  <dataset_config>
```

Required arguments:

- exp_dir: experiment directory containing config.yaml and checkpoints.
- dataset_config: path to a dataset config YAML (for example configs/data/TMP_640k.yaml).

Common options:

- --max_samples: limit number of batches evaluated.
- --astar or --mhastar: use A* / Multi Heuristic A* for planning
- --restrict_search_space: constrain search to only pixels with a minimum confidence of 50%.
- --desired_obstacle_distance: clearance target for obstacle metrics.
- --min_hardness: filter out easy samples below this hardness.

Example:

```bash
python scripts/evaluation/compute_eval_metrics_singleshot.py \
  experiments/runs/my_run \
  configs/data/TMP_640k.yaml \
  --use_best \
  --max_samples 20
```

## Outputs

- Prints a summary table of metrics to stdout.
- Writes a JSON report under <exp_dir>/evaluation_results/ with filename
  eval_<UTC-timestamp>.json.
