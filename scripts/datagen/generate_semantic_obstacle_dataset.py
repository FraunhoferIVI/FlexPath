import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import argparse

from src.datagen.semantic_dataset_sampling import sample_obstacles
from src.utils.common import seed_everything


def main():
    """Add additional obstacle levels to existing dataset splits.

    The input .npz files must contain the arrays expected by
    `sample_additional_obstacles` (see implementation for details).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/TMP_640k_rgb",
        help="Directory containing input split .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/TMP_640k_semantic_obstacle",
        help="Directory to save modified datasets.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="validation,test,train",
        help="Comma-separated list of splits to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible obstacle placement.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        sample_obstacles(
            f"{args.input_dir}/{split}.npz",
            f"{args.output_dir}/{split}.npz",
        )


if __name__ == '__main__':
    main()
