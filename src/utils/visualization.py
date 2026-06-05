
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Get repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.metrics import compute_path_optimality, compute_bins, compute_bins_clipped

from typing import Callable

from scipy.ndimage import label

from src.data.adapters import HuggingFaceAdapter, NpzAdapter
from src.data.unified_dataset import ActorDataset

from PIL import Image

import os


class VisualisationCallback():
    """Callback for creating visualisation, called after eval. """  # "data/TMP_64k_rgb"
    # def __init__(self, exp_name: str, logger, model_pred_f: Callable, dataset: str = "Cubpaw/voxelgym_5c_42x42_500", get_target: Callable = None, get_orig_target: Callable = None, deterministic: bool = True, tmp: bool = True):
    def __init__(self, exp_name: str, logger, model_pred_f: Callable, dataset: str = "data/TMP_640k_rgb", get_target: Callable = None, get_orig_target: Callable = None, deterministic: bool = True, tmp: bool = True, latents: tuple = None, N: int = 1):
        # datasets: TMP_640k_obstacle_levels_rgb_v2
        self.exp_name = exp_name
        self.output_parent_dir = os.path.join("experiments", "runs", self.exp_name, "visualisations")
        self.deterministic = deterministic

        self.get_target = get_target
        self.get_orig_target = get_orig_target

        if tmp:
            adapter = NpzAdapter(data_dir=dataset)
            
            self.dataset = ActorDataset(
                source_adapter=adapter,
                split="validation",
                image_height=64,
                image_width=64,
                shift_p=0.0,
                disable_augmentation=True
            )
        else:
            adapter = HuggingFaceAdapter(
                repo=dataset,
            )
            
            self.dataset = ActorDataset(
                source_adapter=adapter,
                split="train",
                image_height=42,
                image_width=42,
                shift_p=0.0,
                disable_augmentation=True
            )

        self.n_calls = 0

        self.logger = logger
        self.model_pred_f = model_pred_f


    def _on_step(self, step) -> None:
            
        evaluate_actor_nonreasoning(
            self.model_pred_f, 
            self.dataset, 
            "cuda",
            self.logger,
            step=step,
            num_samples=20, 
            output_dir=os.path.join(self.output_parent_dir, str(self.n_calls)),
            diagonal_movements_at_obstacle=True,
            deterministic=self.deterministic,
            get_target=self.get_target,
            get_orig_target=self.get_orig_target
        )

        self.n_calls += 1
    
    def __call__(self, step):
        self._on_step(step)


def overlay_path_on_image(image, path_mask, color=[1.0, 0.0, 0.0], alpha=0.5):
    """
    Overlay a path mask on an image with transparency.

    Args:
        image: numpy array [H, W, 3]
        path_mask: numpy array [H, W] binary mask
        color: RGB color for the path
        alpha: transparency (0=transparent, 1=opaque)

    Returns:
        Overlaid image
    """
    overlay = image.copy()
    mask_3d = np.stack([path_mask] * 3, axis=-1)
    overlay = np.where(mask_3d, image * (1 - alpha) + np.array(color) * alpha, image)
    return overlay


def evaluate_actor_nonreasoning(model, dataset, device, logger, step, num_samples=10, output_dir=None, diagonal_movements_at_obstacle=False, deterministic=True, get_target: Callable = None, get_orig_target: Callable = None):
    """Evaluate actor and create visualizations."""
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    first_preds_path = os.path.join(output_dir.parent, "first_preds.npy")
    last_preds_path = os.path.join(output_dir.parent, "last_preds.npy")

    if os.path.exists(last_preds_path):
        # load data
        first_preds = np.load(file=first_preds_path, allow_pickle=False)
        last_preds = np.load(file=last_preds_path, allow_pickle=False)
    
    else:
        # first eval -> nothing to compare against
        last_preds = None

    current_preds = list()

    print(f"\nGenerating visualizations for {num_samples} samples...")

    metrics = {
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    path_optimality_metrics = {
        "is_collision": [],
        "is_valid": [],
        "is_optimal": [],
        "cost_factor": [],
        "expansion_ratio": []
    }

    valid_paths = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            starts = batch["starts"].cpu().numpy()
            ends = batch["ends"].cpu().numpy()
            unnormalized_image = batch["unnormalized_image"].to(device)  # for optimality_metric calc

            zeros = torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device, dtype=images.dtype)

            # Forward pass
            logits = model(torch.cat((images, zeros), dim=1), deterministic=deterministic)[0]
            sampled_pred = model(torch.cat((images, zeros), dim=1), deterministic=False)[0]

            try:
                logits = logits.cpu().detach().numpy()
                sampled_pred = sampled_pred.cpu().detach().numpy()
            except():
                # already numpy
                pass

            logits = logits.reshape(-1, 64, 64)  # undo flattening

            confidence_map = np.tanh(logits)

            logits = logits[0]

            # add logits to current preds
            current_preds.append(logits)

            predicted = confidence_map > 0

            # Convert to numpy
            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            label_np = labels[0].cpu().numpy().astype(bool)
            pred_np = predicted[0].astype(bool).reshape(64, 64)
            confidence_map_np = confidence_map[0]
            unnormalized_image_np = unnormalized_image[0].cpu().numpy()
            predicted_tanh_np = confidence_map[0]

            # Calculate metrics
            # traditonal metrics
            intersection = np.logical_and(pred_np, label_np).sum()
            union = np.logical_or(pred_np, label_np).sum()
            iou = intersection / union if union > 0 else 0.0

            tp = intersection
            fp = np.logical_and(pred_np, ~label_np).sum()
            fn = np.logical_and(~pred_np, label_np).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["iou"].append(iou)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)

            is_collision, is_valid, is_optimal, cost_factor, expansion_ratio = compute_path_optimality(
                predicted_path_occupancy_map=pred_np,
                state=unnormalized_image_np,
                diagonal_movements_at_obstacle=diagonal_movements_at_obstacle
            )

            path_optimality_metrics["is_collision"].append(is_collision)
            path_optimality_metrics["is_valid"].append(is_valid)

            if is_valid:
                path_optimality_metrics["is_optimal"].append(is_optimal)
                path_optimality_metrics["cost_factor"].append(cost_factor)
                path_optimality_metrics["expansion_ratio"].append(expansion_ratio)
                valid_paths += 1  # increment counter for valid paths

            # Create visualization with overlays
            fig, axes = plt.subplots(4, 4, figsize=(15, 10))

            # also include sampled pred
            sampled_pred = sampled_pred.reshape(-1, 64, 64)[0] >= 0.0
            img_with_sampled_pred = overlay_path_on_image(image_np, sampled_pred, color=[1.0, 0.0, 0.0], alpha=0.6)
            axes[3, 2].imshow(img_with_sampled_pred)
            axes[3, 2].set_title("Input + Predicted Path (Sampled) \n(Red overlay)", fontsize=10)
            axes[3, 2].axis("off")

            # Row 1: Overlays
            # Original image with A* path (ground truth) in green
            img_with_gt = overlay_path_on_image(image_np, label_np, color=[0.0, 1.0, 0.0], alpha=0.6)
            axes[0, 0].imshow(img_with_gt)
            axes[0, 0].set_title("Input + A* Path (GT)\n(Green overlay)", fontsize=10)
            axes[0, 0].axis("off")

            # Original image with predicted path in red
            img_with_pred = overlay_path_on_image(image_np, pred_np, color=[1.0, 0.0, 0.0], alpha=0.6)
            axes[0, 1].imshow(img_with_pred)
            axes[0, 1].set_title("Input + Predicted Path\n(Red overlay)", fontsize=10)
            axes[0, 1].axis("off")

            # Both paths overlaid (GT=Green, Pred=Red)
            img_both = image_np.copy()
            img_both = overlay_path_on_image(img_both, label_np, color=[0.0, 1.0, 0.0], alpha=0.4)
            img_both = overlay_path_on_image(img_both, pred_np, color=[1.0, 0.0, 0.0], alpha=0.4)
            axes[0, 2].imshow(img_both)
            axes[0, 2].set_title("Both Overlaid\n(GT=Green, Pred=Red)", fontsize=10)
            axes[0, 2].axis("off")

            # Row 2: Individual masks and difference
            axes[1, 0].imshow(label_np, cmap="Greens", vmin=0, vmax=1)
            axes[1, 0].set_title("A* Path (Ground Truth)", fontsize=10)
            axes[1, 0].axis("off")

            axes[1, 1].imshow(confidence_map_np, cmap="Reds", vmin=0, vmax=1)
            axes[1, 1].set_title("Predicted Path Confidence", fontsize=10)
            axes[1, 1].axis("off")

            # Difference visualization: TP=White, FP=Red, FN=Blue
            diff = np.zeros((*pred_np.shape, 3))
            tp_mask = np.logical_and(pred_np, label_np)
            fp_mask = np.logical_and(pred_np, ~label_np)
            fn_mask = np.logical_and(~pred_np, label_np)

            diff[tp_mask] = [1.0, 1.0, 1.0]  # True positives in white
            diff[fp_mask] = [1.0, 0.0, 0.0]  # False positives in red
            diff[fn_mask] = [0.0, 0.0, 1.0]  # False negatives in blue

            axes[1, 2].imshow(diff)
            axes[1, 2].set_title("Difference\nTP=White, FP=Red, FN=Blue", fontsize=10)
            axes[1, 2].axis("off")

            # Histogram
            # print(predicted_tanh_np, predicted_tanh_np.shape)
            bin_counts_now, bin_edges = compute_bins(var=predicted_tanh_np, bin_width=0.05)
            axes[1, 3].bar(bin_edges[:-1], bin_counts_now, width=0.05, align='edge', edgecolor='black')
            axes[1, 3].set_xlabel('Threshold')
            axes[1, 3].set_yscale('log')
            axes[1, 3].set_ylabel('Predicted Path Elements')
            axes[1, 3].set_title('Threshold Histogram')

            bin_counts, bin_edges = compute_bins_clipped(var=logits, bin_width=0.5, lower_bound=-10, upper_bound=10)
            axes[0, 3].bar(bin_edges[:-1], bin_counts, width=0.5, align='edge', edgecolor='black')
            axes[0, 3].set_xlabel('Logit')
            axes[0, 3].set_yscale('log')
            axes[0, 3].set_ylabel('Predicted Path Elements')
            axes[0, 3].set_title('Threshold Histogram unsquashed')

            if last_preds is not None:
                logit_diff_to_first = logits - first_preds[i]
                logit_diff_to_last = logits - last_preds[i]

                vmax_first = 5

                axes[2, 0].imshow(logit_diff_to_first, cmap="bwr", vmin=-vmax_first, vmax=vmax_first)
                axes[2, 0].set_title("Logit change compared to first eval", fontsize=10)
                axes[2, 0].axis("off")

                vmax_last = 1

                axes[2, 1].imshow(logit_diff_to_last, cmap="bwr", vmin=-vmax_last, vmax=vmax_last)
                axes[2, 1].set_title("Logit change compared to last eval", fontsize=10)
                axes[2, 1].axis("off")

                bin_counts, bin_edges = compute_bins(var=np.tanh(last_preds[i]), bin_width=0.05)
                bin_dif_last = bin_counts_now - bin_counts
                axes[2, 2].bar(bin_edges[:-1], bin_dif_last, width=0.005, align='edge', edgecolor='black')
                axes[2, 2].set_xlabel('Tanh')
                axes[2, 2].set_ylabel('Predicted Path Elements (clip=10)')
                axes[2, 2].set_title('Histogram Difference to last eval')

                bin_counts, bin_edges = compute_bins(var=np.tanh(first_preds[i]), bin_width=0.05)
                bin_dif_start = bin_counts_now - bin_counts
                axes[2, 3].bar(bin_edges[:-1], bin_dif_start, width=0.05, align='edge', edgecolor='black')
                axes[2, 3].set_xlabel('Tanh')
                axes[2, 3].set_ylabel('Predicted Path Elements (clip=10)')
                axes[2, 3].set_title('Histogram Difference to first eval')

                # clusters
                connected_path, _, _ = compute_connected_mask_with_metrics(
                    path=pred_np,
                    start=tuple(starts[0]),
                    target=tuple(ends[0])
                )

                unconnected_path = np.logical_and(pred_np, ~connected_path)

                outlier_map = overlay_path_on_image(image_np, connected_path, color=[1.0, 0.0, 0.0], alpha=0.4)
                outlier_map = overlay_path_on_image(outlier_map, unconnected_path, color=[1.0, 1.0, 0.0], alpha=0.4)

                axes[3, 0].imshow(outlier_map)
                axes[3, 0].set_title("Main path and outliers", fontsize=10)
                axes[3, 0].axis("off")

                if get_target is not None:
                    # get target
                    target = get_target(i)[0]

                    if get_orig_target is not None:
                        # overlay
                        # get original target
                        orig_target = get_orig_target(i)[0]

                        target_overlay = overlay_path_on_image(image_np, target, color=[1.0, 0.0, 0.0], alpha=0.4)
                        target_overlay = overlay_path_on_image(target_overlay, orig_target, color=[0.0, 1.0, 0.0], alpha=0.4)

                        # plot
                        axes[3, 1].imshow(target_overlay)
                        axes[3, 1].set_title("Raw (untransformed) target in green and transformed target in red ", fontsize=10)
                        axes[3, 1].axis("off")

                    else:
                        # only target
                        axes[3, 1].imshow(target, vmin=0, vmax=1)
                        axes[3, 1].set_title("Current Target", fontsize=10)
                        axes[3, 1].axis("off")

            # Add metrics as text
            if cost_factor is None:
                fig.suptitle(
                    f"Sample {i+1} | IoU: {iou:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Cost: - | Expansion Factor: -",
                    fontsize=12,
                    fontweight="bold"
                )
            else:
                fig.suptitle(
                    f"Sample {i+1} | IoU: {iou:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Cost: {cost_factor:.3f} | Expansion Factor: {expansion_ratio:.3f}",
                    fontsize=12,
                    fontweight="bold"
                )

            plt.tight_layout()

            if output_dir:
                save_path = output_dir / f"actor_sample_{i+1:03d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  [{i+1}/{num_samples}] Saved: {save_path.name}")
            else:
                plt.show()

            # save raw image of prediction seperately
            Image.fromarray(np.clip(img_with_pred* 255, a_min=0, a_max=255).astype(np.uint8)).save(output_dir / f"pred{i+1:03d}.png")
            Image.fromarray(np.clip(img_with_gt* 255, a_min=0, a_max=255).astype(np.uint8)).save(output_dir / f"gt{i+1:03d}.png")

            plt.close()

    # Print aggregate metrics
    print(f"\n{'='*60}")
    print(f"Aggregate Metrics (n={num_samples}):")
    print(f"{'='*60}")
    print(f"  Mean IoU:                  {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
    print(f"  Mean Precision:            {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
    print(f"  Mean Recall:               {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
    print(f"  Mean F1:                   {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")

    print(f"  Collision Ratio:           {np.mean(path_optimality_metrics['is_collision']):.4f} ± {np.std(path_optimality_metrics['is_collision']):.4f}")
    print(f"  Path Validity:             {np.mean(path_optimality_metrics['is_valid']):.4f} ± {np.std(path_optimality_metrics['is_valid']):.4f}")
    
    if valid_paths > 0:
        print(f"  Optimal Path Found Ratio:  {np.mean(path_optimality_metrics['is_optimal']):.4f} ± {np.std(path_optimality_metrics['is_optimal']):.4f}")
        print(f"  Avg. Cost Factor           {np.mean(path_optimality_metrics['cost_factor']):.4f} ± {np.std(path_optimality_metrics['cost_factor']):.4f}")
        print(f"  Avg. Expansion Ratio       {np.mean(path_optimality_metrics['expansion_ratio']):.4f} ± {np.std(path_optimality_metrics['expansion_ratio']):.4f}")
    
    else:
        print("  Optimal Path Found Ratio:  No valid paths!")
        print("  Avg. Cost Factor           No valid paths!")
        print("  Avg. Expansion Ratio       No valid paths!")

    print(f"{'='*60}\n")

    # Save metrics to file
    if output_dir:
        metrics_file = output_dir / "metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("Actor Evaluation Metrics\n")
            f.write("=" * 60 + "\n")
            f.write(f"Number of samples: {num_samples}\n\n")
            f.write(f"Mean IoU:                    {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}\n")
            f.write(f"Mean Precision:              {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}\n")
            f.write(f"Mean Recall:                 {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}\n")
            f.write(f"Mean F1:                     {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}\n")

            f.write(f"  Collision Ratio:           {np.mean(path_optimality_metrics['is_collision']):.4f} ± {np.std(path_optimality_metrics['is_collision']):.4f}\n")
            f.write(f"  Path Validity:             {np.mean(path_optimality_metrics['is_valid']):.4f} ± {np.std(path_optimality_metrics['is_valid']):.4f}\n")

            logger.add_scalar("eval/colision_ratio", np.mean(path_optimality_metrics['is_collision']), step)
            logger.add_scalar("eval/validity_ratio", np.mean(path_optimality_metrics['is_valid']), step)
                
            if valid_paths > 0:
                f.write(f"  Optimal Path Found Ratio:  {np.mean(path_optimality_metrics['is_optimal']):.4f} ± {np.std(path_optimality_metrics['is_optimal']):.4f}\n")
                f.write(f"  Avg. Cost Factor           {np.mean(path_optimality_metrics['cost_factor']):.4f} ± {np.std(path_optimality_metrics['cost_factor']):.4f}\n")
                f.write(f"  Avg. Expansion Ratio       {np.mean(path_optimality_metrics['expansion_ratio']):.4f} ± {np.std(path_optimality_metrics['expansion_ratio']):.4f}\n")

                logger.add_scalar("eval/avg_cost_factor", np.mean(path_optimality_metrics['cost_factor']), step)
                logger.add_scalar("eval/expansion_ratio", np.mean(path_optimality_metrics['expansion_ratio']), step)

            else:
                f.write("  Optimal Path Found Ratio:  No valid paths!")
                f.write("  Avg. Cost Factor           No valid paths!")
                f.write("  Avg. Expansion Ratio       No valid paths!")

        print(f"Metrics saved to: {metrics_file}")

        current_preds = np.stack(current_preds)  # [N, H, W]

        if last_preds is None:
            # if first eval -> save first preds
            np.save(first_preds_path, current_preds)

        # print(current_preds.shape)
        np.save(last_preds_path, arr=current_preds)

def compute_connected_mask_with_metrics(
        path: np.ndarray, 
        start: tuple,
        target: tuple
):
    """
    Filters the path down to only those elements connected to either start or target.

    Parameters:
    - path: Binary 2D map with 1: path, 0: no path
    - start: tuple of start coordinates
    - target: tuple of target coordinates

    Returns:
    - Connected path map
    - Pixel sum of parts of connected path
    - Pixel sum of outliers

    """

    # set start and targets to 1 (edge case where prediction only goes up to next to the start/target)
    path[start] = 1
    path[target] = 1

    structure = np.ones((3, 3))
    
    labels, _ = label(path, structure=structure)

    s_label = labels[start]
    t_label = labels[target]

    # keep pixels connected to start OR target
    connected_path = (labels == s_label) | (labels == t_label)

    connected_path_pixelsum = np.sum(connected_path)
    outlier_pixelsum = np.sum(path) - connected_path_pixelsum

    return (
        connected_path,
        connected_path_pixelsum,
        outlier_pixelsum
    )
