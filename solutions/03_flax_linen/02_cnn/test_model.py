"""Tests for CNN Model"""

import pytest
import jax
import jax.numpy as jnp
from model import CNN, CNNWithBatchNorm


class TestCNN:
    def test_forward_pass(self):
        model = CNN(num_classes=10)
        rng = jax.random.key(0)
        x = jnp.ones((4, 32, 32, 3))
        variables = model.init(rng, x)
        output = model.apply(variables, x, training=False)
        assert output.shape == (4, 10)

    def test_with_batchnorm(self):
        model = CNNWithBatchNorm(num_classes=10)
        rng = jax.random.key(0)
        x = jnp.ones((4, 32, 32, 3))
        variables = model.init(rng, x, training=True)
        assert 'batch_stats' in variables

        output, _ = model.apply(
            variables, x, training=True, mutable=['batch_stats']
        )
        assert output.shape == (4, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
