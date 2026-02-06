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
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)

    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)

    attention_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Xavier initialization."""
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads

        # Linear projections
        q = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        k = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        v = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(x)

        # Reshape for multi-head: (batch, seq, heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Attention
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Reshape back: (batch, seq, embed_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())(attn_output)

        return output, attn_weights


class TransformerEncoderBlock(nn.Module):
    """Encoder Block: MHA + residual + LayerNorm + FFN."""
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, training: bool = True):
        # Multi-head attention with residual
        attn_output, _ = MultiHeadAttention(self.embed_dim, self.num_heads)(x, mask)
        attn_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = nn.LayerNorm()(x + attn_output)

        # Feed-forward with residual
        ff = nn.Dense(self.ff_dim)(x)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.embed_dim)(ff)
        ff = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff)
        x = nn.LayerNorm()(x + ff)

        return x


def sinusoidal_positional_encoding(seq_len, embed_dim):
    """Sine/cosine positional encoding."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(embed_dim)[np.newaxis, :]

    angles = positions / np.power(10000, (2 * (dims // 2)) / embed_dim)

    pos_encoding = np.zeros((seq_len, embed_dim))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return jnp.array(pos_encoding)


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
        seq_len = x.shape[1]

        # Token embedding
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)

        # Positional encoding
        pos_encoding = sinusoidal_positional_encoding(self.max_seq_len, self.embed_dim)
        x = x + pos_encoding[:seq_len]

        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Encoder blocks
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(
                self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate
            )(x, mask, training)

        # Classification head (use [CLS] token or mean pooling)
        x = jnp.mean(x, axis=1)  # Mean pooling
        x = nn.Dense(self.num_classes)(x)

        return x


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
