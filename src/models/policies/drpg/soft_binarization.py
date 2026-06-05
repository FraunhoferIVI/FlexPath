import torch


@torch.compile
def soft_heaviside(
    x: torch.Tensor, 
    eps=1e-3
):
    
    """

    Smooth approximation of a Heaviside step (sign) function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    eps : float
        Stabilizer to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Output in [0, 1], differentiable everywhere.

    """

    return 0.5 * (x / torch.sqrt(x*x + eps) + 1)
