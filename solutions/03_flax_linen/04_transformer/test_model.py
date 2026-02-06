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
        q = jnp.ones((4, 8, 64))  # batch=4, seq=8, dim=64
        k = jnp.ones((4, 8, 64))
        v = jnp.ones((4, 8, 64))

        output, weights = scaled_dot_product_attention(q, k, v)
        assert output.shape == (4, 8, 64)
        assert weights.shape == (4, 8, 8)

    def test_multi_head_attention(self):
        mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        rng = jax.random.key(0)
        x = jnp.ones((4, 8, 64))
        variables = mha.init(rng, x)
        output, weights = mha.apply(variables, x)
        assert output.shape == (4, 8, 64)

    def test_encoder_block(self):
        block = TransformerEncoderBlock(embed_dim=64, num_heads=8, ff_dim=256)
        rng = jax.random.key(0)
        x = jnp.ones((4, 8, 64))
        variables = block.init(rng, x, training=True)
        output = block.apply(variables, x, training=False)
        assert output.shape == (4, 8, 64)

    def test_full_encoder(self):
        model = TransformerEncoder(vocab_size=1000, embed_dim=64, num_classes=10)
        rng = jax.random.key(0)
        x = jax.random.randint(rng, (4, 16), 0, 1000)
        variables = model.init(rng, x, training=True)
        output = model.apply(variables, x, training=False)
        assert output.shape == (4, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
