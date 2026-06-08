# Stage II (Finetuning) guide

Stage II finetuning refines a pretrained model to align with different objectives.

## Relevant files

Configs:
- configs/config_singleshot.yaml: top-level config
- configs/model/single_shot/drpg_singleshot.yaml: contains path to pretrained model
- configs/training/drpg.yaml: training and PSO-related hyperparams
- configs/data/*.yaml: dataset configs for finetuning

Training:
- scripts/training/train_singleshot_pso.py: Hydra entrypoint for finetuning.

PSO:
- src/pso/objectives/*.py: differentiable reward implementations.
- src/pso/pso_registry.py: reward function registry and component names.
- src/pso/pso_objective.py: wrapper that selects and calls reward functions.

Dataset Handling:
- src/data/dataset_factory.py: dataset construction.

## Available PSO objectives

1. 'mindist': Shortest paths
2. 'obstacle_v2': Obstacle avoidance
3. 'obstacle_levels': Semantic obstacle avoidance
4. 'reward_waypoint_mindist': Waypoints

## Config overview

General Note: The hyperparam **total_epochs** does not correspond to the 'standard' epoch, we do sampling without replacement, therefore each epoch contains **batches_per_epoch** steps and after each 'epoch' an evaluation is performed.

Training Note: While we recommend training for the full 250k steps for full reproducibility, a shorter run of 50k steps with a higher learning rate of 4e-4 yields only slightly worse results.

Top-level composition: configs/config_singleshot.yaml

- defaults: picks the model, training, and data config.
- experiment.*: output dir, seed, device.

Model config: configs/model/single_shot/drpg_singleshot.yaml

- model.single_shot.actor.experiment_dir: auto-loads model type and weights
  from a pretraining experiment directory.
- model.single_shot.actor.use_weights: whether to load checkpoint weights. Only disable if you want to finetune from scratch. This is not recommended and only works for the shortest-paths objective.

Training config: configs/training/drpg.yaml

- training.hyperparams.reward_f: PSO objective key from pso_registry.
- training.hyperparams.total_epochs, batch_size, learning_rate: core training
  hyperparameters.

Objective-specific training overrides:

- configs/training/mindist.yaml: shortest-paths objective
- configs/training/obstacleavoidance.yaml: obstacle avoidance objective
- configs/training/semanticavoidance.yaml: semantic obstacle avoidance objective
- configs/training/waypoints.yaml: waypoints objective

Dataset config: configs/data/*.yaml

- dataset.source: hf, localnpz, or localzarr.
- dataset.data_dir or dataset.hf_repo: dataset location.
- dataset.image_height and dataset.image_width: input size.

## Extend PSO objectives

To add a new reward function:

1. Implement it under src/pso/objectives/.
2. Register it in src/pso/pso_registry.py:
   - Add a new key to AVAILABLE_REWARD_FUNCTIONS.
   - Add its component names to REWARD_COMPONENTS (used for tensorboard).
3. Use it in training via config:

```bash
python scripts/training/train_singleshot_pso.py \
  training.hyperparams.reward_f=my_reward
```

## Start finetuning

Run with defaults:

```bash
python scripts/training/train_singleshot_pso.py \
  training=<name of training file for objective>
```
Objective-specific training files:

- configs/training/mindist.yaml: shortest-paths objective
- configs/training/obstacleavoidance.yaml: obstacle avoidance objective
- configs/training/semanticavoidance.yaml: semantic obstacle avoidance objective
- configs/training/waypoints.yaml: waypoints objective

For the semantic objectives you need to point to the right dataset, add data=<dataset config file name>.

Point to a pretrained actor:

```bash
python scripts/training/train_singleshot_pso.py \
  model.single_shot.actor.experiment_dir=experiments/pretraining/my_run
```

Adjust core hyperparameters:

```bash
python scripts/training/train_singleshot_pso.py \
  training.hyperparams.total_epochs=50 \
  training.hyperparams.batch_size=256 \
  training.hyperparams.learning_rate=1e-5
```
