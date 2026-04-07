"""
models/transformer_encoder.py — Transformer Encoder for Tabular IIoT Data
Input: (batch, input_dim) tabular features → Output: (batch, embed_dim) embeddings
"""
import torch
import torch.nn as nn
import math


class TabularTransformerEncoder(nn.Module):
    """
    Transformer encoder adapted for tabular network flow features.
    Projects input features to embedding space, applies self-attention,
    then outputs L2-normalized embeddings for prototype computation.
    """
    def __init__(self, input_dim, embed_dim=128, num_heads=4,
                 num_layers=3, dropout=0.1, ff_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Project input features to embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Feature-wise positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 2, embed_dim) * 0.02)  # 2 = cls + feature

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Projection head for contrastive learning (MLP)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Classification head (built dynamically)
        self.classifier = None
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_classifier(self, num_classes):
        """Attach a classification head for cross-entropy loss."""
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, num_classes),
        )
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Move to same device as the rest of the model
        device = next(self.parameters()).device
        self.classifier = self.classifier.to(device)

    def encode(self, x):
        """
        Get raw (unnormalized) embeddings.
        x: (B, input_dim) -> h: (B, embed_dim)
        """
        B = x.shape[0]

        # Project features
        feat = self.input_proj(x).unsqueeze(1)  # (B, 1, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, feat], dim=1)   # (B, 2, D)

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        # Transformer
        h = self.transformer(tokens)
        h = self.norm(h[:, 0, :])  # Take CLS token output (B, D)
        return h

    def forward(self, x, return_embeddings=False):
        """
        Forward pass.
        Returns: (logits, z_normalized) or just z_normalized
        """
        h = self.encode(x)

        # Contrastive projection (L2 normalized)
        z = self.proj_head(h)
        z_norm = nn.functional.normalize(z, dim=1, p=2)

        if return_embeddings:
            return z_norm, h

        if self.classifier is not None:
            logits = self.classifier(h)
            return logits, z_norm

        return z_norm

    def get_num_params(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total
