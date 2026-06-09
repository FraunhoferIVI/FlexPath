# FlexPath: Learned Semantic Path Priors for Image-Based Planning

## TL;DR

Train a neural network to predict *where paths can go* (Stage 1), then adapt it to decide *which path you actually want* (Stage 2), shortest, safest, semantic-aware, or waypoint-guided, all from the same pretrained model.

| Optimal | Obstacle Avoidance | Semantic Avoidance | Waypoint |
|:-------:|:------------------:|:------------------:|:--------:|
| <img src="resources/optimal.gif" width="180"> | <img src="resources/obstavoidance.gif" width="180"> | <img src="resources/semavoidance.gif" width="180"> | <img src="resources/waypoint.gif" width="180"> |

> The model predicts a heuristic in a single-shot fashion, then a classical search algorithms (Focal Search) navigates the map. Dark green nodes were expanded, the last frame shows the final path. Our heuristic strongly pushes the search algorithm towards the target almost without unnecessary exploration, all while aligning with the trained preference.

---

## This Repository Contains

- ✅ Training code for Stage I (pretraining) and Stage II (fine-tuning)
- ✅ Implementation of all PSOs
- ✅ Basic evaluation scripts

> [!NOTE]
> This branch is kept lightweight for easier extendability. See the `full` branch for baseline implementations used in the paper.

---

<details>
<summary>
<h2>1) Basic setup</h2>
</summary>

### Requirements

- Python 3.9 (see `pyproject.toml`)
- CUDA GPU recommended for training

### Clone repo:

```bash
git clone https://github.com/FraunhoferIVI/FlexPath
```

### Install dependencies

**Compilation dependencies:**:
```bash
apt install g++
```
(for cstar dependency and torch.compile)

**Python dependencies:**
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

### Download Checkpoints:

```bash
curl -L -o experiments.zip https://owncloud.fraunhofer.de/index.php/s/pvO2lJaNFI1IOId/download
unzip experiments.zip
```

### Download Datasets

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

# Generate semantic dataset extensions

This is only required if you want to retrain or evaluate the waypoint/semantic obstacle avoidance objectives.

```bash
uv run python scripts/datagen/generate_semantic_obstacle_dataset.py
uv run python scripts/datagen/generate_waypoint_dataset.py
```

### Repository layout (high level)

- `scripts/pretraining/train_actor.py`: actor pretraining entrypoint
- `scripts/training/train_singleshot_pso.py`: single-shot fine-tuning entrypoint
- `configs/`: Hydra configs for training, models, and datasets
- `docs/`: detailed guides for pretraining, finetuning, and evaluation
- `src/`: core code (datasets, models, PSO objectives, training loops)
- `experiments/`: default output location for runs

</details>

<details>
<summary>
<h2>2) Pretraining and fine-tuning</h2>
</summary>

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

</details>

<details>
<summary>
<h2>3) Evaluation</h2>
</summary>

See [docs/evaluation.md](docs/evaluation.md) for how to run the evaluation
script and interpret its outputs.

Take a look at the notebooks folder for visualizations.

</details>

<details>
<summary>
<h2>4) Extending the system</h2>
</summary>

See [docs/pretraining.md](docs/pretraining.md) and [docs/finetuning.md](docs/finetuning.md) on how to add new backbones, PSOs or datasets.

</details>

## Notes

- All scripts are Hydra entrypoints; any config value can be overridden at the
	CLI with `key=value` syntax.
- Always use the 'latest' checkpoint over the 'best'. This is because the scaling of some components increase over the course of the training, which causes total loss / reward to become worse.


## Citation
```bibtex
@misc{FlexPath2026,
  title         = {FlexPath: Learned Semantic Path Priors for Image-Based Planning},
  author        = {Taehyoung Kim and Tim Sch{\"o}nbrod and David Eckel and Henri Mee{\ss}},
  year          = {2026},
  eprint        = {2026.00000},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2026.00000}
}
```
