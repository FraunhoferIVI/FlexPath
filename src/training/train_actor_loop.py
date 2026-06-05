"""
Training loop for actor (UNET) pretraining.
Binary path segmentation task using BCEWithLogitsLoss.
"""

import logging
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import evaluate

from src.utils.checkpoint import CheckpointManager
from src.utils.custom_loss_functions import WeighedFocalLossWithLogits, MSEWLL

from src.utils.lr_schedules import create_scheduler_from_config

from src.buffers.fast_dataset import FastDataloader
from src.buffers.replay_buffer import CUDAPrefetcher


from copy import deepcopy


logger = logging.getLogger(__name__)


def ema_step_(
    trained_model: torch.nn.Module, 
    ema_model: torch.nn.Module, 
    alpha: float, 
    is_compiled: bool = True
) -> torch.nn.Module:
    """Update EMA parameters in-place.

    Parameters
    ----------
    trained_model : torch.nn.Module
        Source model that provides the current parameters.
    ema_model : torch.nn.Module
        Exponential moving average (EMA) shadow model to update.
    alpha : float
        EMA decay factor. The update is
        ``ema = alpha * ema + (1 - alpha) * trained``.
    is_compiled : bool, optional
        If True, uses ``_orig_mod`` parameters from ``torch.compile`` models.

    Returns
    -------
    torch.nn.Module
        The updated EMA model.
    """
    if is_compiled:
        with torch.no_grad():
            for p_trained, p_ema in zip(trained_model._orig_mod.parameters(), ema_model._orig_mod.parameters()):
                p_ema.mul_(alpha).add_(p_trained, alpha=1 - alpha)
    else:
        with torch.no_grad():
            for p_trained, p_ema in zip(trained_model.parameters(), ema_model.parameters()):
                p_ema.mul_(alpha).add_(p_trained, alpha=1 - alpha)

    return ema_model

def train_actor(model, train_dataset, val_dataset, config, device="cuda", diagonal_movements_at_obstacle=False, compile_model: bool = True):
    """Run the actor pretraining loop.

    Parameters
    ----------
    model : torch.nn.Module
        Actor segmentation model (e.g., UNet-style).
    train_dataset : torch.utils.data.Dataset
        Training dataset.
    val_dataset : torch.utils.data.Dataset
        Validation dataset.
    config : DictConfig
        Hydra configuration object with ``training`` and ``loss`` sections.
    device : str, optional
        Torch device string, by default ``"cuda"``.
    diagonal_movements_at_obstacle : bool, optional
        Whether labels allow diagonal moves at obstacles; used only for optional
        path-optimality metrics.
    compile_model : bool, optional
        If True, compiles the model with ``torch.compile`` for training.

    Returns
    -------
    torch.nn.Module
        The trained EMA model (or the original model if EMA is disabled).
    """

    torch.set_float32_matmul_precision('high')
    
    # Extract config params
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate
    num_workers = config.training.num_workers
    validation_epoch = config.training.validation_epoch
    checkpoint_epoch = config.training.checkpoint_epoch
    class_imbalance = config.training.class_imbalance
    gradient_clipping_by_norm = config.training.gradient_clipping_by_norm
    ema_alpha = config.training.ema_alpha

    # Optimizer hyperparameters (configurable, fallback to defaults)
    opt_cfg = config.training.get("optimizer", {})
    weight_decay = opt_cfg.get("weight_decay", 0.03)
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
    eps = opt_cfg.get("eps", 1e-8)
    fused = opt_cfg.get("fused", True)
    optimizer_type = opt_cfg.get("type", "adamw").lower()

    # Mixed precision settings
    use_mixed_precision = config.training.get("use_mixed_precision", False)
    precision = config.training.get("precision", "fp32")

    # Validate precision setting
    if precision not in ["fp32", "fp16", "bf16"]:
        raise ValueError(f"Invalid precision '{precision}'. Must be one of: 'fp32', 'fp16', 'bf16'")

    # Only enable mixed precision on CUDA with fp16/bf16
    enable_amp = use_mixed_precision and device == "cuda" and precision in ["fp16", "bf16"]

    # Set dtype for autocast
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16  # Default for fp16

    # Gradient scaling only needed for FP16 (not BF16)
    # BF16 has same exponent range as FP32, so it doesn't suffer from gradient underflow
    enable_grad_scaling = enable_amp and precision == "fp16"

    # Setup directories
    from pathlib import Path
    output_dir = Path(config.experiment.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs" / config.experiment.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    from omegaconf import OmegaConf
    OmegaConf.save(config, output_dir / "config.yaml")

    # DataLoaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )

    train_loader = CUDAPrefetcher(FastDataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    ))

    val_loader = CUDAPrefetcher(FastDataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    ))

    # Model to device
    model = model.to(device)

    # setup ema
    if ema_alpha != 0:
        ema_model = deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()
        is_ema = True
    else:
        ema_model = model
        is_ema = False

    if compile_model:
        model = torch.compile(
            model=model, 
            backend="inductor", 
            mode="reduce-overhead",
            dynamic=False   # last batch must be dropped then
        )

        if is_ema:
            ema_model = torch.compile(
                model=ema_model, 
                backend="inductor", 
                mode="reduce-overhead",
                dynamic=False   # last batch must be dropped then
            )

    # Optimizer (configurable)
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused,
        )
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Gradient scaler for mixed precision (only needed for FP16, not BF16)
    scaler = GradScaler(enabled=enable_grad_scaling)

    lr_schedule = create_scheduler_from_config(
        optimizer=optimizer,
        scheduler_cfg=config.training.get("lr_schedule", {}),
        total_epochs=num_epochs,
    )

    # Loss
    # Binary Cross Entropy With Logits Loss
    if config.loss.type == "BCEWithLogitsLoss":
        # With class weight
        if config.loss.apply_weighted_loss:
            loss_fn = WeighedFocalLossWithLogits(alpha=class_imbalance, gamma=0.0)  # Focal loss with gamma=0 -> WBCE
        # Without class weight
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    # Focal Loss With Logits
    elif config.loss.type == "Focal":
        # Load alpha and gamma params from config
        alpha = config.loss.focal_alpha
        gamma = config.loss.focal_gamma
        loss_fn = WeighedFocalLossWithLogits(alpha=alpha, gamma=gamma)

    # Mean Squared Error With Logits
    # For experimentation, not sure if MSE is really useful for this 
    elif config.loss.type == "MSEWLL":
        # Sigmoid is not really needed here for numerical stability
        # However, to avoid making changes to the model architecture it will be included here
        loss_fn = MSEWLL() # chains sigmoid and MSE for convenience

    # Throw error if loss_fn from config is invalid
    else:
        logger.error("loss_fn specified in configs/pretraining/config_actor.yaml must be one of {'BCEWithLogitsLoss', 'Focal', 'MSEWLL'}")
        exit(1)

    # Metrics
    metric = evaluate.load("mean_iou")

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        exp_dir=str(output_dir),
        keep_last_n=config.checkpoint.get("keep_last_n", 0)
    )

    # Training loop
    model.train()
    best_val_loss = float('inf')

    logger.info("=" * 60)
    logger.info("Starting Actor Pretraining")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Mixed precision: {'Enabled' if enable_amp else 'Disabled'} (precision: {precision})")
    logger.info("=" * 60)

    # Cache lengths of train and test loaders
    train_loader_length = len(train_loader)
    val_loader_length = len(val_loader)

    for epoch in range(num_epochs):
        # Training phase
        epoch_loss = 0.0
        gradient_sum = 0.0

        logger.info(f"[Epoch {epoch+1}/{num_epochs}]")

        for batch in tqdm(train_loader, desc="Training", leave=False):
            model.train()

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(dtype=amp_dtype, enabled=enable_amp):
                outputs = model(
                    images=images,
                    labels=labels,
                    loss_fn=loss_fn
                )
                loss = outputs.loss

            logits = outputs.logits

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Optionally apply gradient normalization
            if gradient_clipping_by_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            scaler.step(optimizer)
            scaler.update()

            # Track gradients
            if hasattr(model, 'first_conv'):
                gradients = model.first_conv.weight.grad
                if gradients is not None:
                    gradient_sum += gradients.mean().item()

            epoch_loss += loss.item()

            # ema update
            if is_ema:
                ema_step_(
                    trained_model=model,
                    ema_model=ema_model,
                    alpha=ema_alpha,
                    is_compiled=compile_model
                )

        # Calculate average loss and gradients
        epoch_loss /= train_loader_length
        gradient_sum /= train_loader_length

        # Calculate metrics on last batch
        with torch.no_grad():
            threshold = 0.0
            predicted = (torch.tanh(logits) >= threshold).int()
            metrics = metric._compute(
                num_labels=1,
                predictions=predicted.detach().cpu(),
                references=labels.detach().cpu(),
                ignore_index=255,
                reduce_labels=False
            )

            # Count avg. number of path pixel predictions per sample
            path_pixel_predictios = torch.sum(predicted).item() / predicted.shape[0]  # divide by batch dim of last batch

        logger.info(f"  Train Loss: {epoch_loss:.4f} | IoU: {metrics['mean_iou']:.4f} | Acc: {metrics['mean_accuracy']:.4f} | Path predictions: {path_pixel_predictios:.6f} | Grad: {gradient_sum:.6f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Metrics/train_iou", metrics["mean_iou"], epoch)
        writer.add_scalar("Metrics/train_accuracy", metrics["mean_accuracy"], epoch)
        writer.add_scalar("Gradients/mean_first_layer", gradient_sum, epoch)

        lr_schedule.step(epoch=epoch)

        # Validation phase
        if epoch % validation_epoch == 0:
            ema_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    images = batch["images"].to(device)
                    labels = batch["labels"].to(device)
                    unnormalized_images = batch["unnormalized_image"].to(device)

                    # Validation with mixed precision
                    with autocast(dtype=amp_dtype, enabled=enable_amp):
                        outputs = ema_model(
                            images=images,
                            labels=labels,
                            loss_fn=loss_fn
                        )
                        loss = outputs.loss
                        logits = outputs.logits

                    val_loss += loss.item()

                # Calculate metrics on last batch
                predicted = (torch.tanh(logits) >= threshold).int()
                val_metrics = metric._compute(
                    num_labels=1,
                    predictions=predicted.detach().cpu(),
                    references=labels.detach().cpu(),
                    ignore_index=255,
                    reduce_labels=False
                )

                # Convert to numpy
                pred_np = predicted.cpu().numpy().astype(bool)
                unnormalized_image_np = unnormalized_images.cpu().numpy()

                # define metrics
                path_optimality_metrics = {
                    "is_collision": [],
                    "is_valid": [],
                    "is_optimal": [],
                    "cost_factor": [],
                    "expansion_ratio": []

                }

                # calculate path optimality metrics (does not support batches)
                last_batch_size = len(pred_np)
                valid_paths = 0  # count valid paths
                # for i in range(last_batch_size):
                #     is_collision, is_valid, is_optimal, cost_factor, expansion_ratio = compute_path_optimality(
                #         predicted_path_occupancy_map=pred_np[i],
                #         state=unnormalized_image_np[i],
                #         diagonal_movements_at_obstacle=diagonal_movements_at_obstacle
                #     )

                #     path_optimality_metrics["is_collision"].append(is_collision)
                #     path_optimality_metrics["is_valid"].append(is_valid)

                #     # only use those metrics if path is valid
                #     if is_valid:
                #         path_optimality_metrics["is_optimal"].append(is_optimal)
                #         path_optimality_metrics["cost_factor"].append(cost_factor)
                #         path_optimality_metrics["expansion_ratio"].append(expansion_ratio)
                #         valid_paths += 1

                # Count avg. number of path pixel predictions per sample
                path_pixel_predictios = torch.sum(predicted).item() / predicted.shape[0]  # divide by batch dim of last batch

            val_loss /= val_loader_length

            logger.info(f"  Val Loss: {val_loss:.4f} | IoU: {val_metrics['mean_iou']:.4f} | Acc: {val_metrics['mean_accuracy']:.4f}")

            # Log to TensorBoard
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/val_iou", val_metrics["mean_iou"], epoch)
            writer.add_scalar("Metrics/val_accuracy", val_metrics["mean_accuracy"], epoch)
            writer.add_scalar("Metrics/path_pixel_predictions", path_pixel_predictios, epoch)

            # writer.add_scalar("Metrics/val_collision_ratio", sum(path_optimality_metrics["is_collision"]) / last_batch_size, epoch)
            # writer.add_scalar("Metrics/val_path_validity_ratio", sum(path_optimality_metrics["is_valid"]) / last_batch_size, epoch)

            # if valid_paths > 0:  
            #     # those metrics are only appended for valid paths
            #     # without check -> potentially throws division by zero error
            #     writer.add_scalar("Metrics/val_path_optimality_ratio", sum(path_optimality_metrics["is_optimal"]) / valid_paths, epoch)   # length <= batch_size
            #     writer.add_scalar("Metrics/val_avg_cost_factor", sum(path_optimality_metrics["cost_factor"]) / valid_paths, epoch)
            #     writer.add_scalar("Metrics/val_avg_expansion_ratio", sum(path_optimality_metrics["expansion_ratio"]) / valid_paths, epoch)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_manager.save_checkpoint(
                    model={
                        "state_dict": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    step=epoch,
                    metadata={
                        "val_loss": val_loss,
                        "val_iou": val_metrics["mean_iou"],
                        "val_accuracy": val_metrics["mean_accuracy"]
                    },
                    is_best=True
                )
                logger.info(f"  ✓ Best checkpoint saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % checkpoint_epoch == 0 and epoch > 0:
            checkpoint_manager.save_checkpoint(
                model={
                    "state_dict": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                },
                step=epoch,
                metadata={"train_loss": epoch_loss},
                is_best=False
            )
            logger.info(f"  ✓ Checkpoint saved at epoch {epoch}")

    # Save final checkpoint as latest
    checkpoint_manager.save_checkpoint(
        model={
            "state_dict": ema_model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        step=num_epochs,
        metadata={"train_loss": epoch_loss},
        is_best=False
    )

    writer.close()

    logger.info("=" * 60)
    logger.info("✓ Training completed!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs: {log_dir}")
    logger.info("=" * 60)

    return ema_model
