## FlexPath: Learned Semantic Path Priors for Image-Based Planning

This repository contains:

- Training code for Stage I (pretraining) and II (finetuning)
- Implementation of all PSOs
- Basic evaluation scripts

This branch is kept lightweight for easier extendability, see 'full' branch for baseline implementations and remaining evaluation scripts.

## 1) Basic setup

### Requirements

- Python 3.9 (see `pyproject.toml`)
- CUDA GPU recommended for training

### Clone repo:

```bash
git clone ...
```

### Install dependencies

Option 1 (uv):
```bash
uv sync
```

Option 2 (pip):
```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Repository layout (high level)

- `scripts/pretraining/train_actor.py`: actor pretraining entrypoint
- `scripts/training/train_singleshot_pso.py`: single-shot fine-tuning entrypoint
- `configs/`: Hydra configs for training, models, and datasets
- `docs/`: detailed guides for pretraining, finetuning, and evaluation
- `src/`: core code (datasets, models, PSO objectives, training loops)
- `experiments/`: default output location for runs

## 2) Pretraining and fine-tuning

### Stage I (Pretraining)

Entrypoint: `scripts/pretraining/train_actor.py`

For a detailed guide, see [docs/pretraining.md](docs/pretraining.md).

Run with defaults:

```bash
python scripts/pretraining/train_actor.py
```

### Stage II (Finetuning)

Entrypoint: `scripts/training/train_singleshot_pso.py`

For a detailed guide, see [docs/finetuning.md](docs/finetuning.md).

Run with defaults:

```bash
python scripts/training/train_singleshot_pso.py
```

## Evaluation

See [docs/evaluation.md](docs/evaluation.md) for how to run the evaluation
script and interpret its outputs.

## 3) Extending the system

### Add a new dataset (config-only)

If your data matches an existing adapter, you only need a new config file.

1. Copy a template config and edit it:
	 - Pretraining: `configs/pretraining/data/` (see `train_with_any_dataset.yaml`)
	 - Fine-tuning: `configs/data/` (see existing dataset configs)

2. Update fields:
	 - `dataset.source`: `hf`, `localnpz`, or `localzarr`
	 - `dataset.hf_repo` or `dataset.data_dir`
	 - `dataset.image_height`, `dataset.image_width`
	 - `normalization.*` as needed

3. Run with your new config:

```bash
python scripts/pretraining/train_actor.py data=my_new_dataset
```

### Add a new data source / adapter

If your data is not HF, NPZ, or Zarr:

1. Implement a new adapter in `src/data/adapters/`.
2. Register it in `src/data/dataset_factory.py`.
3. Add a config that sets `dataset.source` to your new adapter key.

### Add a new PSO reward (objective)

1. Implement the reward function in `src/pso/objectives/`.
2. Register it in `src/pso/pso_registry.py`:
	 - Add to `AVAILABLE_REWARD_FUNCTIONS`.
	 - Add its component names to `REWARD_COMPONENTS`.
3. Select it in config:

```bash
python scripts/training/train_singleshot_pso.py training.hyperparams.reward_f=my_reward
```

### Add a new actor model

1. Implement the model in `src/models/actor/`.
2. Register it in `src/utils/model_registry.py` under `ACTOR_REGISTRY`.
3. Add a model config under `configs/pretraining/model/` and set
	 `model.type` to your registry key.
4. Use it in pretraining:

```bash
python scripts/pretraining/train_actor.py model=my_new_model
```

## Notes

- All scripts are Hydra entrypoints; any config value can be overridden at the
	CLI with `key=value` syntax.
- The resolved config for each run is saved into the experiment directory.
- GPU selection is controlled by `experiment.device`.
- Always use the 'latest' checkpoint over the 'best'. This is because the scaling of some components increase over the course of the training, which causes total loss / reward to become worse.
