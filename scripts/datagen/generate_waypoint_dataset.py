import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import argparse

from src.datagen.semantic_dataset_sampling import sample_waypoints
from src.utils.common import seed_everything


def main():
    """Sample waypoints and write modified .npz splits.

    Input .npz files should contain the arrays expected by
    `sample_waypoints` (see implementation for exact names/shapes).
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
        default="data/TMP_640k_waypoints",
        help="Directory to save modified datasets.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="validation,test,train",
        help="Comma-separated list of splits to process.",
    )
    parser.add_argument(
        "--proximity",
        type=int,
        default=200,
        help="Max distance from path to place waypoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible waypoint sampling.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        sample_waypoints(
            f"{args.input_dir}/{split}.npz",
            f"{args.output_dir}/{split}.npz",
            proximity=args.proximity,
        )


if __name__ == '__main__':
    main()
