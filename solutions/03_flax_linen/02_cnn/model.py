"""
CNN (Convolutional Neural Network) in Flax Linen
=================================================

Flax concepts demonstrated:
- nn.Conv, nn.max_pool, nn.avg_pool
- CIFAR-10 style architecture
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class CNN(nn.Module):
    """
    Simple CNN for image classification.
    Architecture: Conv -> Pool -> Conv -> Pool -> Dense -> Output
    """
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Conv block 1
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv block 2
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv block 3
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Dense layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)

        x = nn.Dense(self.num_classes)(x)
        return x


class CNNWithBatchNorm(nn.Module):
    """CNN with BatchNorm for better training."""
    num_classes: int = 10
    features: Sequence[int] = (32, 64, 128)

    @nn.compact
    def __call__(self, x, training: bool = True):
        for feat in self.features:
            x = nn.Conv(features=feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


if __name__ == '__main__':
    print("CNN Examples")
    print("=" * 50)

    rng = jax.random.key(0)
    model = CNN(num_classes=10)

    # CIFAR-10 like input
    x = jnp.ones((4, 32, 32, 3))
    variables = model.init(rng, x, training=True)

    print(f"Input shape: {x.shape}")
    output = model.apply(variables, x, training=False)
    print(f"Output shape: {output.shape}")

    # Parameter count
    params = jax.tree.leaves(variables['params'])
    total_params = sum(p.size for p in params)
    print(f"Total parameters: {total_params:,}")
