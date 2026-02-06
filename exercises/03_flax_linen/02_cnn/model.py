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

    Attributes:
        num_classes: Number of output classes for classification.

    The forward pass should:
    1. Conv block 1: Conv(32 features, 3x3 kernel, SAME padding) -> ReLU -> MaxPool(2x2)
    2. Conv block 2: Conv(64 features, 3x3 kernel, SAME padding) -> ReLU -> MaxPool(2x2)
    3. Conv block 3: Conv(128 features, 3x3 kernel, SAME padding) -> ReLU
    4. Global average pooling (mean over spatial dimensions)
    5. Dense(256) -> ReLU -> Dropout(0.5) -> Dense(num_classes)
    """
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training: bool = True):
        # TODO: Implement Conv block 1
        # - nn.Conv with features=32, kernel_size=(3, 3), padding='SAME'
        # - nn.relu activation
        # - nn.max_pool with window_shape=(2, 2), strides=(2, 2)
        pass

        # TODO: Implement Conv block 2
        # - nn.Conv with features=64, kernel_size=(3, 3), padding='SAME'
        # - nn.relu activation
        # - nn.max_pool with window_shape=(2, 2), strides=(2, 2)

        # TODO: Implement Conv block 3
        # - nn.Conv with features=128, kernel_size=(3, 3), padding='SAME'
        # - nn.relu activation

        # TODO: Global average pooling
        # - Use jnp.mean over spatial dimensions (axis 1 and 2)

        # TODO: Dense layers
        # - nn.Dense(256) -> nn.relu
        # - nn.Dropout(rate=0.5, deterministic=not training)
        # - nn.Dense(self.num_classes)

        return None


class CNNWithBatchNorm(nn.Module):
    """
    CNN with BatchNorm for better training.

    Attributes:
        num_classes: Number of output classes for classification.
        features: Sequence of feature sizes for each conv block.

    The forward pass should:
    1. For each feature count in self.features:
       - Conv(features, 3x3 kernel, SAME padding)
       - BatchNorm(use_running_average=not training)
       - ReLU
       - MaxPool(2x2)
    2. Global average pooling
    3. Dense(256) -> ReLU -> Dense(num_classes)
    """
    num_classes: int = 10
    features: Sequence[int] = (32, 64, 128)

    @nn.compact
    def __call__(self, x, training: bool = True):
        # TODO: Implement conv blocks with BatchNorm
        # For each feat in self.features:
        # - nn.Conv(features=feat, kernel_size=(3, 3), padding='SAME')
        # - nn.BatchNorm(use_running_average=not training)
        # - nn.relu
        # - nn.max_pool(window_shape=(2, 2), strides=(2, 2))
        pass

        # TODO: Global average pooling
        # - Use jnp.mean over spatial dimensions (axis 1 and 2)

        # TODO: Dense layers
        # - nn.Dense(256) -> nn.relu
        # - nn.Dense(self.num_classes)

        return None


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
