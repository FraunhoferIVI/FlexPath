import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Standard transformer block with learned positional embeddings and without linear projections.
    """
    
    def __init__(self, channels: int, resolution: int, n_heads: int = 1, dropout_probability: float = 0.2):
        """
        Custtomizable Transformer block with learned positional encoding and without linear layers.

        Args:
        - channels: (int) In/Outgoing channels
        - resolution: (list[list]) Resolution of feature map
        - n_heads: (list[int]) Number of attention heads, must dividide channels evenly
        - dropout_probability: (float) Dropout probability
        
        """

        super().__init__()

        # cache channel count and resolution for reshaping of feature map later
        self.channels = channels
        self.resolution = resolution
        self.resolution_squared = resolution ** 2

        # initalize norms
        self.norm1 = nn.LayerNorm(normalized_shape=(channels))
        self.norm2 = nn.LayerNorm(normalized_shape=(channels))
        self.norm3 = nn.LayerNorm(normalized_shape=(channels))

        # learned positional embedding
        self.positional_encoding = nn.Parameter(data=torch.zeros((self.resolution_squared, self.channels)))

        # initalize attention heads
        self.att1 = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=dropout_probability, batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=dropout_probability, batch_first=True)

        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
        - x: tensor of shape [B, resolution, resolution]
         
        Returns: 
        - y: tensor of shape [B, resolution, resolution]
        
        """
        
        # [B, C, H, W] reshape -> [B, C, H*W] permute -> [B, H*W, C]
        x = x.reshape(-1, self.channels, self.resolution_squared).permute(0, 2, 1)
        
        # inject positional encoding
        x = x + self.positional_encoding 

        # first cycle
        normed_x1 = self.norm1(x)
        x1, _ = self.att1(normed_x1, normed_x1, normed_x1)
        x1 = x1 + x

        # second cycle
        normed_x2 = self.norm2(x1)
        x2, _ = self.att2(normed_x2, normed_x2, normed_x2)
        x2 = x2 + x1

        # permute and reshape back
        x3 = self.norm3(x2).permute(0, 2, 1).reshape(-1, self.channels, self.resolution, self.resolution)

        x2 = x2.permute(0, 2, 1).reshape(-1, self.channels, self.resolution, self.resolution)

        x4 = self.projection(x3) + x2
        # x4 = x2.permute(0, 2, 1).reshape(-1, self.channels, self.resolution, self.resolution)

        return x4
