from typing import Optional, List

from torch.utils.tensorboard import SummaryWriter


def log_to_tensorboard(
    writer: Optional[SummaryWriter], 
    values: List[float], 
    names: List[str], 
    step: int, 
    split: str = "train"
):
    
    """

    Log a list of scalar metrics to TensorBoard.

    Parameters
    ----------
    writer : Optional[SummaryWriter]
        TensorBoard writer.
    values : List[float]
        Values to log.
    names : List[str]
        Metric names.
    step : int
        Global logging step.
    split : str
        Namespace ("train" or "eval").

    """

    if writer is not None:
        for val, name in zip(values, names):
            writer.add_scalar(f"{split}/{name}", val, step)
