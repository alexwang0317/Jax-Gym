"""Tests for Transformer Model"""

import pytest
import jax
import jax.numpy as jnp
from model import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    TransformerEncoderBlock,
    TransformerEncoder
)


class TestTransformer:
    def test_scaled_attention(self):
        """Test scaled dot product attention output shapes."""
        q = jnp.ones((4, 8, 64))  # batch=4, seq=8, dim=64
        k = jnp.ones((4, 8, 64))
        v = jnp.ones((4, 8, 64))

        output, weights = scaled_dot_product_attention(q, k, v)
        assert output.shape == (4, 8, 64), f"Expected output shape (4, 8, 64), got {output.shape}"
        assert weights.shape == (4, 8, 8), f"Expected weights shape (4, 8, 8), got {weights.shape}"

    def test_scaled_attention_with_mask(self):
        """Test scaled dot product attention with mask."""
        q = jnp.ones((2, 4, 32))
        k = jnp.ones((2, 4, 32))
        v = jnp.ones((2, 4, 32))
        # Create a simple mask (e.g., causal mask)
        mask = jnp.tril(jnp.ones((4, 4)))

        output, weights = scaled_dot_product_attention(q, k, v, mask)
        assert output.shape == (2, 4, 32)
        assert weights.shape == (2, 4, 4)

    def test_multi_head_attention(self):
        """Test multi-head attention module."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        rng = jax.random.key(0)
        x = jnp.ones((4, 8, 64))
        variables = mha.init(rng, x)
        output, weights = mha.apply(variables, x)
        assert output.shape == (4, 8, 64), f"Expected output shape (4, 8, 64), got {output.shape}"

    def test_multi_head_attention_head_dim(self):
        """Test that embed_dim is divisible by num_heads."""
        mha = MultiHeadAttention(embed_dim=128, num_heads=4)
        rng = jax.random.key(0)
        x = jnp.ones((2, 16, 128))
        variables = mha.init(rng, x)
        output, _ = mha.apply(variables, x)
        assert output.shape == (2, 16, 128)

    def test_encoder_block(self):
        """Test transformer encoder block."""
        block = TransformerEncoderBlock(embed_dim=64, num_heads=8, ff_dim=256)
        rng = jax.random.key(0)
        x = jnp.ones((4, 8, 64))
        variables = block.init(rng, x, training=True)
        output = block.apply(variables, x, training=False)
        assert output.shape == (4, 8, 64), f"Expected output shape (4, 8, 64), got {output.shape}"

    def test_encoder_block_training_mode(self):
        """Test encoder block in training mode (dropout active)."""
        block = TransformerEncoderBlock(embed_dim=64, num_heads=8, ff_dim=256, dropout_rate=0.1)
        rng = jax.random.key(0)
        x = jnp.ones((4, 8, 64))
        variables = block.init(rng, x, training=True)
        # In training mode, need to pass rngs for dropout
        output = block.apply(variables, x, training=True, rngs={'dropout': rng})
        assert output.shape == (4, 8, 64)

    def test_full_encoder(self):
        """Test complete transformer encoder."""
        model = TransformerEncoder(vocab_size=1000, embed_dim=64, num_classes=10)
        rng = jax.random.key(0)
        x = jax.random.randint(rng, (4, 16), 0, 1000)
        variables = model.init(rng, x, training=True)
        output = model.apply(variables, x, training=False)
        assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"

    def test_full_encoder_different_seq_len(self):
        """Test encoder with different sequence lengths."""
        model = TransformerEncoder(vocab_size=500, embed_dim=32, num_classes=5, max_seq_len=100)
        rng = jax.random.key(42)

        # Test with different sequence lengths
        for seq_len in [8, 16, 32]:
            x = jax.random.randint(rng, (2, seq_len), 0, 500)
            variables = model.init(rng, x, training=True)
            output = model.apply(variables, x, training=False)
            assert output.shape == (2, 5), f"Failed for seq_len={seq_len}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
