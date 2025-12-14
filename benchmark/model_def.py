"""
Benchmark model definition for Templar Tournament.

This module defines the SimpleLM model used for testing and benchmarking.
It must be importable so that torch.load() can reconstruct the model.
"""

import torch
import torch.nn as nn


class SimpleLM(nn.Module):
    """Simple language model for testing (matches reference training loop).
    
    This is a small transformer-based model designed for fast local testing.
    Production would use a larger model (e.g., 150M parameters).
    """
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                dropout=0.0,  # Deterministic for testing
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        h = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        for layer in self.layers:
            h = layer(h)
        logits = self.output(h)  # (batch_size, seq_len, vocab_size)
        return logits


def create_model(
    vocab_size: int = 32000,
    hidden_size: int = 256,
    num_layers: int = 2,
) -> SimpleLM:
    """Create a SimpleLM model with the given configuration."""
    return SimpleLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

