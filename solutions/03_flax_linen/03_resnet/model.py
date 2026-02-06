"""
ResNet in Flax Linen
====================

Demonstrates residual connections and skip connections.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x

        # Main path
        y = nn.Conv(self.features, (3, 3), self.stride, padding='SAME')(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)

        y = nn.Conv(self.features, (3, 3), padding='SAME')(y)
        y = nn.BatchNorm(use_running_average=not training)(y)

        # Skip connection (with projection if needed)
        if residual.shape != y.shape:
            residual = nn.Conv(self.features, (1, 1), self.stride)(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """Simple ResNet for image classification."""
    num_classes: int = 10
    block_sizes: Sequence[int] = (2, 2, 2)
    features: Sequence[int] = (64, 128, 256)

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial conv
        x = nn.Conv(64, (7, 7), (2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        # Residual blocks
        for i, (num_blocks, feat) in enumerate(zip(self.block_sizes, self.features)):
            for j in range(num_blocks):
                stride = 2 if i > 0 and j == 0 else 1
                x = ResidualBlock(feat, stride)(x, training)

        # Global average pooling and classifier
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


if __name__ == '__main__':
    print("ResNet Example")
    rng = jax.random.key(0)
    model = ResNet(num_classes=10)
    x = jnp.ones((4, 32, 32, 3))
    variables = model.init(rng, x, training=True)
    output = model.apply(variables, x, training=False)
    print(f"Output shape: {output.shape}")
