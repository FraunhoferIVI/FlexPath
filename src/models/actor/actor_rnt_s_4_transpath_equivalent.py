import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.blocks.resnet import ResNetBlock
from src.models.blocks.transformer import TransformerBlock

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SemanticSegmenterOutput


class ActorRNT_S_4_TP_Equiv(nn.Module):
    """
    Actor with ResNet+Transformer architecture - Small size.

    Architecture:
    - 3x Encoder/Decoder stages (ResNet blocks)
    - Transformer at bottleneck
    - Skip connections concatenated
    - Each downsampling halves resolution and doubles channels

    Formerly: ActorUnetAtt_s
    """

    def __init__(self, in_channels: int, out_channels: int, base_dim: int, resolution: int, dropout_probability: float = 0.2, rgb: bool = True, keep_x: bool = False):
        super().__init__()

        ### nn design hardcoded here:
        # resolutions are in practice 42 -> 21 -> 10 -> 5
        # self.resolutions = [42, 21, 10, 5]
        self.resolutions = [resolution]
        self.resolutions.append(resolution // 2)
        self.resolutions.append(self.resolutions[-1] // 2)
        self.resolutions.append(self.resolutions[-1] // 2)

        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))

        self.in_channels = in_channels
        self.keep_x = keep_x

        # helper var
        channels = base_dim

        # conv that map channels to base_dim at the beginning
        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=base_dim, kernel_size=(3, 3), padding=1)

        # encoder design
        self.resnet_enc_1 = ResNetBlock(in_channels=channels, out_channels=[channels, channels], kernel_sizes=[(3, 3), (3, 3)], strides=[1, 1], dropout_probability=dropout_probability, swish=True)

        self.resnet_enc_2 = ResNetBlock(in_channels=channels, out_channels=[channels, channels], kernel_sizes=[(3, 3), (3, 3)], strides=[1, 1], dropout_probability=dropout_probability, swish=True)

        self.resnet_enc_3 = ResNetBlock(in_channels=channels, out_channels=[channels, channels], kernel_sizes=[(3, 3), (3, 3)], strides=[1, 1], dropout_probability=dropout_probability, swish=True)

        # bottleneck design
        self.bottleneck1 = TransformerBlock(channels=channels, resolution=self.resolutions[-1], n_heads=4)
        self.bottleneck2 = TransformerBlock(channels=channels, resolution=self.resolutions[-1], n_heads=4)
        self.bottleneck3 = TransformerBlock(channels=channels, resolution=self.resolutions[-1], n_heads=4)
        self.bottleneck4 = TransformerBlock(channels=channels, resolution=self.resolutions[-1], n_heads=4)

        self.resnet_dec_1 = ResNetBlock(in_channels=int(2 * channels), out_channels=[channels, channels], kernel_sizes=[(3, 3), (3, 3)], strides=[1, 1], dropout_probability=dropout_probability, swish=True)

        self.resnet_dec_2 = ResNetBlock(in_channels=int(2 * channels), out_channels=[channels, channels], kernel_sizes=[(3, 3), (3, 3)], strides=[1, 1], dropout_probability=dropout_probability, swish=True)

        self.resnet_dec_3 = ResNetBlock(in_channels=int(2 * channels), out_channels=[channels, channels], kernel_sizes=[(3, 3), (3, 3)], strides=[1, 1], dropout_probability=dropout_probability, swish=True)

        # conv that maps base_dim back to target dimension
        self.last_conv = nn.Conv2d(in_channels=base_dim, out_channels=out_channels, kernel_size=1, padding=0)

        self.rgb = rgb

    def forward(self,
            images: torch.FloatTensor,
            labels: Optional[torch.LongTensor] = None,
            starts: Optional[torch.LongTensor] = None,
            ends: Optional[torch.LongTensor] = None,
            loss_fn: nn.Module = None
        ) -> Union[Tuple, SemanticSegmenterOutput]:

        if self.keep_x:
            X = images
        elif self.rgb:
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

            if self.in_channels == 3:
                X = torch.concat([obstacle_map, start_map, end_map], dim=1).float().detach()
            else:
                startend = start_map + end_map
                X = torch.concat([obstacle_map, startend], dim=1).float().detach()

        ### Inference
        # [3, 42, 42] -> [base_dim, 42, 42]
        x0 = self.first_conv(X)

        # [base_dim, 42, 42]  -> [2*base_dim, 21, 21]
        x_skip_1 = self.resnet_enc_1(x0)
        x_e_1 = self.pooling(x_skip_1)

        # [2*base_dim, 21, 21]   -> [4*base_dim, 11, 11]
        x_skip_2 = self.resnet_enc_2(x_e_1)
        x_e_2 = self.pooling(x_skip_2)

        # [4*base_dim, 11, 11]   -> [8*base_dim, 6, 6]
        x_skip_3 = self.resnet_enc_3(x_e_2)
        x_e_3 = self.pooling(x_skip_3)

        # keeps channels and resolution
        b_1 = self.bottleneck4(self.bottleneck3(self.bottleneck2(self.bottleneck1(x_e_3))))

        # [8*base_dim, 6, 6]     -> [4*base_dim, 11, 11]
        x_d_1 = self.resnet_dec_1(torch.concat((F.interpolate(b_1, size=(self.resolutions[-2], self.resolutions[-2]), mode='nearest'), x_skip_3), dim=1))

        # [4*base_dim, 11, 11]     -> [2*base_dim, 21, 21]
        x_d_2 = self.resnet_dec_2(torch.concat((F.interpolate(x_d_1, size=(self.resolutions[-3], self.resolutions[-3]), mode='nearest'), x_skip_2), dim=1))

        # [2*base_dim, 21, 21]   -> [base_dim, 42, 42]
        x_d_3 = self.resnet_dec_3(torch.concat((F.interpolate(x_d_2, size=(self.resolutions[-4], self.resolutions[-4]), mode='nearest'), x_skip_1), dim=1))

        # [base_dim, 42, 42] -> [1, 42, 42]
        logits = self.last_conv(x_d_3)

        ### Loss calculation (optional)

        # labels are [B, H, W]
        # logits are [B, 1, H, W]
        # -> removed channel dim
        logits_squeezed = logits.clone().reshape(logits.size(0), logits.size(2), logits.size(3))

        if loss_fn is None:
            # Inference mode, no loss calculation
            loss = None
        else:
            # Training mode, loss calculation with loss_fn
            loss = loss_fn(logits_squeezed, labels.float())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits_squeezed,
            hidden_states=None,
            attentions=None,
        )
