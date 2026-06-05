import torch.nn as nn
import torch


class TargetEncoder(nn.Module):
    def __init__(self, height: int, width: int, embed_dim: int = 3):
        super().__init__()

        self.conv_x1 = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(5, 5), padding=2)
        self.conv_x2 = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(5, 5), padding=2)

        self.positional_encoding_x1 = torch.nn.Parameter(
            data=torch.full(
                size=(1, height, width),
                fill_value=0.0
            ),
            requires_grad=True
        )

        self.positional_encoding_x2 = torch.nn.Parameter(
            data=torch.full(
                size=(1, height, width),
                fill_value=0.0
            ),
            requires_grad=True
        )

        self.mhca_x1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            batch_first=True,
            num_heads=1
        )

        self.mhca_x2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            batch_first=True,
            num_heads=1
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a single seperate convolution to each input then applies pairwise cross attention.
        x1, x2: [B, 1, H, W]
        """

        # extract features
        x1 = self.conv_x1(x1)
        x2 = self.conv_x2(x2)

        # inject positional encoding
        x1 = x1 + self.positional_encoding_x1
        x2 = x2 + self.positional_encoding_x2

        # reshape
        _x1 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3]).permute(0, 2, 1)
        _x2 = x2.reshape(x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3]).permute(0, 2, 1)

        # compute mha
        r_x1, _ = self.mhca_x1(_x1, _x2, _x2)
        r_x2, _ = self.mhca_x2(_x2, _x1, _x1)

        # reshape back
        r_x1 = r_x1.permute(0, 2, 1).reshape(x1.shape)
        r_x2 = r_x2.permute(0, 2, 1).reshape(x2.shape)
    
        return r_x1, r_x2