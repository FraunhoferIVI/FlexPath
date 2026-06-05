import torch


class CrossAttentionGate(torch.nn.Module):
    """
    Fuses skip connection features from encoder with decoder features via cross attention.
    """
    
    def __init__(self, channels: int, resolution: int, n_heads: int, p_dropout):
        """
        CrossAttentionGate fuses decoder features with the corresponding encoder skip features. It norms the two inputs then computes the attention with the query from the decoder features and the keys and values from the skip connection.

        Args:
        - channels: (int) In/Outgoing channels
        - resolution: (list[list]) Resolution of feature map
        - n_heads: (list[int]) Number of attention heads, must dividide channels evenly
        - p_dropout: (float) Dropout probability

        """

        super().__init__()

        # cache channel count and resolution for reshaping of feature map later
        self.channels = channels
        self.resolution = resolution
        self.resolution_squared = resolution ** 2
        
        # learned positional encodings
        self.decoder_feature_embedding = torch.nn.Parameter(data=torch.zeros((self.resolution_squared, self.channels), requires_grad=True))
        self.skip_embedding = torch.nn.Parameter(data=torch.zeros((self.resolution_squared, self.channels), requires_grad=True))

        # norms
        self.decoder_feature_norm = torch.nn.LayerNorm(normalized_shape=(channels))
        self.skip_feature_norm = torch.nn.LayerNorm(normalized_shape=(channels))

        self.att = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=p_dropout, batch_first=True)

    def forward(self, decoder_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        # reshape to [B, C, H*W] and permute to [B, H*W, C]
        decoder_features = self.decoder_feature_norm(decoder_features.reshape(shape=(-1, self.channels, self.resolution_squared)).permute(0, 2, 1))
        skip_features = self.skip_feature_norm(skip_features.reshape(shape=(-1, self.channels, self.resolution_squared)).permute(0, 2, 1))

        # add positional encoding
        decoder_features_embed = decoder_features + self.decoder_feature_embedding
        skip_features_embed = skip_features + self.skip_embedding

        # fuse features with cross attention
        fused_features, _ = self.att(decoder_features_embed, skip_features_embed, skip_features_embed)

        # permute back to [B, C, H*W]
        fused_features = fused_features.permute(0, 2, 1)

        # reshape back to [B, C, H, W]
        fused_features = fused_features.reshape(shape=(-1, self.channels, self.resolution, self.resolution))

        return fused_features
