import torch
import torch.nn.functional as F


class MSEWLL(torch.nn.Module):
    """
    Custom pytorch implementation of MSE Loss with preceding Sigmoid. This is supposed to be a convenience function for models that do not have Sigmoid at the end as it is technically not needed for numerical stability.

    """

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, logits: torch.Tensor, label: torch.Tensor):
        """
        Parameters:
        - logits: Output of model, without activation
        - label: Label for loss calculation

        """

        # first feed logits through sigmoid
        pred_activated = F.sigmoid(logits)

        return self.mse(pred_activated, label)


class WeighedFocalLossWithLogits(torch.nn.Module):
    """
    Custom pytorch implementation of weighed focal loss with fused sigmoid for numerical stability. Reduces loss to sum.

    Good baselines:
    - Voxelgym2D: alpha=0.8, gamma=2
    - TMP Dataset (TransPath): alpha=0.9, gamma=2

    In general: higher alpha -> thicker paths 

    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2):
        """
        Parameters:
        - alpha: Class weight assigned to positive class
        - gamma: information asymetry weight

        """

        super().__init__()

        self.alpha = alpha
        self.ialpha = 1 - alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters:
        - pred: Prediction of the network
        - target: Label of the sample

        Returns:
        - Loss (sum)

        """
        
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # combine log and sigmoid terms here for numerical stability
        pred = F.sigmoid(pred)
        pt = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * self.ialpha
        return torch.sum(alpha_t * bce_loss * ((1 - pt) ** self.gamma)) / pred.shape[0]
    