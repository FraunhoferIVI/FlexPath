# Stage I (Pretraining) guide

Stage I pretraining trains the actor network to predict path masks from images.
Training is configured with Hydra and runs from the pretraining entrypoint.

## Relevant files

Configs:
- configs/pretraining/config_actor.yaml: main pretraining config and defaults
- configs/pretraining/model/*.yaml: backbones
- configs/pretraining/data/*.yaml: datasets

Training:
- scripts/pretraining/train_actor.py: Main script for pretraining
- src/training/train_actor_loop.py: training loop

Backbone:
- src/models/actor/: actor backbone implementations
- src/utils/model_registry.py: actor model registry

Dataset Handling: 
- src/data/dataset_factory.py: dataset creation and adapters

## Backbone implementation and extension

To add a new actor backbone:

1. Add your model implementation under src/models/actor/.
2. Register the model in src/utils/model_registry.py under ACTOR_REGISTRY.
3. Add a new model config under configs/pretraining/model/ with a unique name:

```yaml
# configs/pretraining/model/my_backbone_config_file.yaml
type: my_backbone
experiment_name: my_backbone
params:
	<constructor_param1>: <constructor_param1_value>
    <contructor_param2>: <constructor_param2_value>
    ...
```

All values under 'params' will be fed as kwargs into the model constructor.

4. Train with the new model config:

```bash
python scripts/pretraining/train_actor.py model=my_backbone_config_file
```

## Config overview

Main config: configs/pretraining/config_actor.yaml

Key defaults:

```yaml
defaults:
	- model: unet_transformer
	- data: TMP_640k
```

What the main config controls:

- experiment.*: output dir, run naming, device, seed.
- training.*: epochs, batch size, optimizer, mixed precision, schedules.
- loss.*: BCE or focal loss settings.

Dataset config example: configs/pretraining/data/train_with_any_dataset.yaml

```yaml
dataset:
	source: localzarr
	data_dir: <TODO>
	image_height: 64
	image_width: 64
normalization:
	use_simple_norm: true
```

All datasets require the following fields:
- 'image': RGB encoded input (uint8)
- 'path_label': One-hot encoded target path (bool)

DatasetFactory currently supports these sources:

- hf: HuggingFace datasets (dataset.hf_repo)
- localnpz: local NPZ directory
- localzarr: local Zarr directory

## Start training

Run with defaults:

```bash
python scripts/pretraining/train_actor.py
```

Select a dataset and output directory:

```bash
python scripts/pretraining/train_actor.py \
	data=TMP_640k \
	experiment.output_dir=experiments/pretraining/my_run
```

Override key hyperparameters:

```bash
python scripts/pretraining/train_actor.py \
	training.num_epochs=200 \
	training.batch_size=256 \
	training.learning_rate=1e-4
```
