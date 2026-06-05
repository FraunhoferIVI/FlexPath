from src.models.actor.actor_rnt_s_4_transpath_equivalent import ActorRNT_S_4_TP_Equiv
from torch import nn


class ActorRNT_S_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ActorRNT_S_4_TP_Equiv(2, 1, 64, 64, keep_x=True)  # defaults that we also use

    def forward(self, x):
        logits = self.model(x).logits.unsqueeze(1)
        
        return logits