import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    MultiHeadSelfAttention with learnable positional encoding, Layernorm and residual connection.
    """

    def __init__(self, channels: int, resolution: int, n_heads: int, dropout_probability: float = 0.2):
        """
        Single attention layer with learned positional encoding and without linear layers.

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

        # learned positional embedding
        self.positional_encoding = nn.Parameter(data=torch.zeros((self.resolution_squared, self.channels)))

        # init norm and attention module
        self.norm = nn.LayerNorm(normalized_shape=(channels))
        self.att = nn.MultiheadAttention(embed_dim=self.channels, num_heads=n_heads, dropout=dropout_probability, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
        - x: tensor of shape [B, resolution, resolution]
         
        Returns: 
        - y: tensor of shape [B, resolution, resolution]
        
        """

        # [B, C, H, W] reshape -> [B, C, H*W] permute -> [B, H*W, C]
        x = x.reshape(-1, self.channels, self.resolution_squared).permute(0, 2, 1) + self.positional_encoding

        # apply norm -> attention -> add residual
        normed_x = self.norm(x)
        att_x, _ = self.att(normed_x, normed_x, normed_x)
        res_x = att_x + x
        
        # reshape & permute back to [B, C, H, W]
        return res_x.permute(0, 2, 1).reshape(-1, self.channels, self.resolution, self.resolution)
    