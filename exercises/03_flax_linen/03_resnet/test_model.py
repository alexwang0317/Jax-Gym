"""Tests for ResNet Model"""

import pytest
import jax
import jax.numpy as jnp
from model import ResidualBlock, ResNet


class TestResNet:
    def test_residual_block(self):
        """Test that ResidualBlock maintains shape when stride=1."""
        block = ResidualBlock(features=64)
        rng = jax.random.key(0)
        x = jnp.ones((4, 16, 16, 64))
        variables = block.init(rng, x, training=True)
        output = block.apply(variables, x, training=False)
        assert output.shape == x.shape

    def test_resnet_forward(self):
        """Test ResNet forward pass produces correct output shape."""
        model = ResNet(num_classes=10)
        rng = jax.random.key(0)
        x = jnp.ones((4, 32, 32, 3))
        variables = model.init(rng, x, training=True)
        output = model.apply(variables, x, training=False)
        assert output.shape == (4, 10)

    def test_skip_connection(self):
        """Test skip connection with dimension change.

        When stride=2 and features change, the residual block should:
        - Reduce spatial dimensions by half
        - Change channel dimensions to match features
        """
        block = ResidualBlock(features=128, stride=2)
        rng = jax.random.key(0)
        x = jnp.ones((4, 16, 16, 64))
        variables = block.init(rng, x, training=True)
        output = block.apply(variables, x, training=False)
        assert output.shape == (4, 8, 8, 128)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
