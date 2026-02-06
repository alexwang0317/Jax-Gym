"""
Transformer in Flax Linen
=========================

Based on UvA DLC Transformer tutorial.
Demonstrates:
- Scaled Dot Product Attention
- Multi-Head Attention
- Encoder Block with residuals
- Positional Encoding
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled Dot Product Attention: softmax(QK^T / sqrt(d_k)) * V

    Args:
        q: Queries (..., seq_len, d_k)
        k: Keys (..., seq_len, d_k)
        v: Values (..., seq_len, d_v)
        mask: Optional attention mask

    Returns:
        Attention output and attention weights
    """
    # TODO: Implement scaled dot product attention
    # 1. Get d_k from the last dimension of q
    # 2. Compute attention scores: QK^T / sqrt(d_k)
    #    Hint: Use jnp.matmul and jnp.swapaxes for transpose
    # 3. Apply mask if provided (set masked positions to -1e9)
    # 4. Apply softmax to get attention weights
    # 5. Compute output by multiplying weights with values
    # 6. Return output and attention weights
    pass


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Xavier initialization."""
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads

        # TODO: Implement multi-head attention
        # 1. Create Q, K, V projections using nn.Dense with xavier_uniform initialization
        # 2. Reshape Q, K, V to (batch, seq, num_heads, head_dim)
        # 3. Transpose to (batch, num_heads, seq, head_dim)
        # 4. Apply scaled_dot_product_attention
        # 5. Transpose back and reshape to (batch, seq, embed_dim)
        # 6. Apply output projection with nn.Dense
        # 7. Return output and attention weights
        pass


class TransformerEncoderBlock(nn.Module):
    """Encoder Block: MHA + residual + LayerNorm + FFN."""
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training: bool = True):
        # TODO: Implement transformer encoder block
        # 1. Apply MultiHeadAttention
        # 2. Apply dropout to attention output
        # 3. Add residual connection and layer normalization
        # 4. Apply feed-forward network:
        #    - Dense(ff_dim) -> GELU -> Dense(embed_dim) -> Dropout
        # 5. Add residual connection and layer normalization
        # 6. Return output
        pass


def sinusoidal_positional_encoding(seq_len, embed_dim):
    """Sine/cosine positional encoding."""
    # TODO: Implement sinusoidal positional encoding
    # 1. Create position indices: [0, 1, ..., seq_len-1]
    # 2. Create dimension indices: [0, 1, ..., embed_dim-1]
    # 3. Compute angles: pos / 10000^(2i/embed_dim)
    # 4. Apply sin to even indices, cos to odd indices
    # 5. Return as jnp array
    pass


class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder."""
    vocab_size: int
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 512
    max_seq_len: int = 512
    num_classes: int = 10
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training: bool = True):
        # TODO: Implement complete transformer encoder
        # 1. Get sequence length from input
        # 2. Apply token embedding using nn.Embed
        # 3. Add positional encoding (use sinusoidal_positional_encoding)
        # 4. Apply dropout
        # 5. Stack num_layers TransformerEncoderBlock
        # 6. Apply mean pooling over sequence dimension
        # 7. Apply classification head (Dense to num_classes)
        # 8. Return output
        pass


if __name__ == '__main__':
    print("Transformer Example")
    rng = jax.random.key(0)
    model = TransformerEncoder(vocab_size=1000, num_classes=10)

    # Token sequence
    x = jax.random.randint(rng, (4, 32), 0, 1000)  # batch=4, seq_len=32
    variables = model.init(rng, x, training=True)

    output = model.apply(variables, x, training=False)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
