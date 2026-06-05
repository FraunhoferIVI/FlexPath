import torch
from torch import nn

from src.models.actor.transPath.modules.encoder import Encoder
from src.models.actor.transPath.modules.decoder import Decoder
from src.models.actor.transPath.modules.attention import SpatialTransformer
from src.models.actor.transPath.modules.pos_emb import PosEmbeds

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SemanticSegmenterOutput


class Autoencoder(nn.Module):
    def __init__(self, 
                in_channels=3, 
                out_channels=1, 
                hidden_channels=64,
                attn_blocks=4,
                attn_heads=4,
                cnn_dropout=0.15,
                attn_dropout=0.15,
                downsample_steps=3, 
                resolution=(64, 64),
                rgb: bool = False,
                mse: bool = True,
                squash_logits: bool = True,
                *args,
                **kwargs):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
        self.pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.transformer = SpatialTransformer(
            hidden_channels, 
            attn_heads,
            heads_dim,
            attn_blocks, 
            attn_dropout
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)

        self.rgb = rgb
        self.squash = squash_logits

        self.mse = mse

    def forward(
        self,
        images: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        starts: Optional[torch.LongTensor] = None,
        ends: Optional[torch.LongTensor] = None,
        loss_fn: nn.Module = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        
        if self.rgb:
            X = images
        else:
            obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=images.device)  # BLUE
            obstacle_map = torch.sigmoid((images - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
            obstacle_map = (obstacle_map < 1.51)  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

            start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=images.device)  # BLUE
            start_map = torch.sigmoid((images - start_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
            start_map = (start_map < 1.51)  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

            end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)  # BLUE
            end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
            end_map = (end_map < 1.51)  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

            startend = start_map + end_map

            X = torch.concat([obstacle_map, startend], dim=1).float().detach()

        x = self.encoder(X)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        logits = self.decoder(x)  # logit range: [-1, 1]

        if self.squash:
            logits = torch.tanh(logits)
        
        logits_squeezed = logits.clone().reshape(logits.size(0), logits.size(2), logits.size(3))

        if loss_fn is None:
            # Inference mode, no loss calculation
            loss = None
        else:
            # Training mode, loss calculation with loss_fn
            if self.mse:
                # move squashed logit vals from [-1, 1] to [0, 1]
                loss = loss_fn((logits_squeezed + 1) / 2.0, labels.float())
            else:   
                loss = loss_fn(logits_squeezed, labels.float())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits_squeezed,
            hidden_states=None,
            attentions=None,
        )
