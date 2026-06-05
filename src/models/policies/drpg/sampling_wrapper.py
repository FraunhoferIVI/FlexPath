import torch
import torch.nn as nn
import torch.nn.functional as F



class SamplingActor(nn.Module):
    """
    Wraps an actor with sampling capabilities. Sampling is made from a squashed gaussian dist

    Overwrites forward() to accept a flag <deterministic> which determines whether to use only the mean or sample with the std before squashing.
    
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()

        self.actor_model = model

    def forward_nonreasoning(self, X: torch.Tensor, deterministic: bool = False):
        means, logstds = self.actor_model(X)

        if deterministic:
            sampled = means
        
        else:
            sampled = means + torch.exp(logstds) * torch.randn(size=means.shape, dtype=means.dtype, device=means.device)
        
        squashed = F.tanh(sampled)

        return squashed, None
    
    def forward(
        self,
        X: torch.Tensor,
        deterministic: bool = False,
    ):
        return self.forward_nonreasoning(X=X, deterministic=deterministic)
    