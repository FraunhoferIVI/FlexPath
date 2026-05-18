import torch


def get_pixel_penalty_schedule_linear(
    start_step: int, 
    total_steps: int, 
    start_val: float, 
    stop_val: float
) -> torch.Tensor:

    """

    Create a linear schedule for pixel penalty scaling.

    Parameters
    ----------
    start_step : int
        Number of initial steps with zero penalty.
    total_steps : int
        Total training steps.
    start_val : float
        Penalty value at the start of scheduling.
    stop_val : float
        Penalty value at the end of training.

    Returns
    -------
    torch.Tensor
        Schedule tensor of length `total_steps`.

    """

    warmup = torch.zeros(size=(start_step, ))
    schedule = torch.linspace(
        start=start_val, 
        end=stop_val,
        steps=total_steps - start_step
    )

    return torch.concat((warmup, schedule))
