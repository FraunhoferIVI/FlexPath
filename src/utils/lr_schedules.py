import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR, OneCycleLR


def make_delayed_cosine_schedule(
    optimizer: torch.optim.Optimizer, 
    total_epochs: int, 
    cosine_decay_start_epoch: int, 
    warmup_epochs: int,
    end_lr: float
):
    """
    Warmup (p epochs) → Flat (until k) → CosineAnnealingLR (PyTorch built-in)
    """
    assert cosine_decay_start_epoch < total_epochs, "k (decay start) must be < total_epochs"

    # --- 1) Warmup + Flat phase via LambdaLR ---
    def lr_lambda(epoch):
        # warmup 0 → 1
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        # flat 1.0
        if epoch < cosine_decay_start_epoch:
            return 1.0
        # after k, this LambdaLR stops modifying LR, cosine scheduler takes over
        return 1.0

    warmup_and_flat = LambdaLR(optimizer, lr_lambda)

    # --- 2) Cosine decay using built-in CosineAnnealingLR ---
    cosine_epochs = total_epochs - cosine_decay_start_epoch
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=end_lr  # or set a nonzero minimum if desired
    )

    # Combine both schedulers into a step function
    def schedule_step(epoch):
        """Call this instead of scheduler.step()."""
        if epoch < cosine_decay_start_epoch:
            warmup_and_flat.step()
        else:
            # advance warmup scheduler to maintain consistency
            warmup_and_flat.step()
            cosine.step()

    # Return a small wrapper object to behave like a scheduler
    class CombinedScheduler:
        def step(self, epoch=None):
            if epoch is None:
                raise ValueError("Call scheduler.step(epoch) with the epoch index.")
            schedule_step(epoch)

        def get_last_lr(self):
            return optimizer.param_groups[0]["lr"]

    return CombinedScheduler()


def create_scheduler_from_config(optimizer: torch.optim.Optimizer, scheduler_cfg: dict, total_epochs: int):
    """
    Build a learning rate scheduler from a lightweight config dict.

    Supported types:
      - "delayed_cosine" (default): warmup + flat + cosine decay (uses make_delayed_cosine_schedule)
      - "cosine": CosineAnnealingLR with T_max=total_epochs
      - "step": StepLR with step_size / gamma
      - "multistep": MultiStepLR with milestones / gamma
      - "exp": ExponentialLR with gamma
      - "constant": no-op scheduler that leaves LR untouched
    """

    if scheduler_cfg is None:
        scheduler_cfg = {}

    scheduler_type = scheduler_cfg.get("type", "delayed_cosine").lower()

    if scheduler_type == "delayed_cosine":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
        cosine_start = scheduler_cfg.get("cosine_decay_start_epoch", 0)
        end_lr = scheduler_cfg.get("end_lr", 1e-10)
        return make_delayed_cosine_schedule(
            optimizer=optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            cosine_decay_start_epoch=cosine_start,
            end_lr=end_lr,
        )

    if scheduler_type == "cosine":
        eta_min = scheduler_cfg.get("eta_min", 0.0)
        print("eta min: ", eta_min)
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    if scheduler_type == "step":
        step_size = scheduler_cfg.get("step_size", max(1, total_epochs // 3))
        gamma = scheduler_cfg.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if scheduler_type == "multistep":
        milestones = scheduler_cfg.get("milestones", [total_epochs // 2])
        gamma = scheduler_cfg.get("gamma", 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if scheduler_type == "exp":
        gamma = scheduler_cfg.get("gamma", 0.99)
        return ExponentialLR(optimizer, gamma=gamma)
    
    if scheduler_type == "onecycle":
        return OneCycleLR(
            optimizer=optimizer,
            max_lr=scheduler_cfg.get("max_lr", 4e-4),
            epochs=total_epochs,
            steps_per_epoch=1,
        )

    if scheduler_type == "constant":
        class _NoOp:
            def step(self, epoch=None):
                return None
            def get_last_lr(self):
                return optimizer.param_groups[0]["lr"]
        return _NoOp()

    raise ValueError(f"Unsupported lr_schedule.type: {scheduler_type}")
