"""
ResNet in Flax Linen
====================

Demonstrates residual connections and skip connections.

Exercise: Implement the ResidualBlock and ResNet classes.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection.

    A residual block consists of:
    1. Main path: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
    2. Skip connection: Identity (or 1x1 Conv if dimensions change)
    3. Add residual and apply final ReLU

    Attributes:
        features: Number of output features (channels).
        stride: Stride for the first convolution. Default is 1.
    """
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Apply the residual block.

        Args:
            x: Input tensor of shape (batch, height, width, channels).
            training: Whether in training mode (affects BatchNorm).

        Returns:
            Output tensor with skip connection applied.

        Hints:
            - Save the input as `residual` for the skip connection
            - Main path: Conv(3x3, stride) -> BatchNorm -> ReLU -> Conv(3x3) -> BatchNorm
            - If residual.shape != y.shape, apply 1x1 Conv + BatchNorm to residual
            - Return ReLU of (y + residual)
        """
        # TODO: Implement the residual block
        # 1. Save input as residual
        # 2. Apply main path: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
        # 3. Apply skip connection (with projection if shapes differ)
        # 4. Add and apply final ReLU
        pass


class ResNet(nn.Module):
    """Simple ResNet for image classification.

    Architecture:
    1. Initial conv (7x7, stride 2) + BatchNorm + ReLU + MaxPool
    2. Stack of residual blocks with increasing features
    3. Global average pooling + Dense classifier

    Attributes:
        num_classes: Number of output classes.
        block_sizes: Tuple of number of blocks per stage.
        features: Tuple of feature sizes per stage.
    """
    num_classes: int = 10
    block_sizes: Sequence[int] = (2, 2, 2)
    features: Sequence[int] = (64, 128, 256)

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Apply the ResNet model.

        Args:
            x: Input tensor of shape (batch, height, width, channels).
            training: Whether in training mode (affects BatchNorm).

        Returns:
            Logits tensor of shape (batch, num_classes).

        Hints:
            - Initial: Conv(7x7, stride=2) -> BatchNorm -> ReLU -> MaxPool(3x3, stride=2)
            - For each stage i, apply block_sizes[i] ResidualBlocks with features[i]
            - First block of each stage (except stage 0) uses stride=2
            - End with global average pooling (mean over spatial dims) and Dense
        """
        # TODO: Implement the ResNet forward pass
        # 1. Initial conv block
        # 2. Residual block stages
        # 3. Global average pooling and classifier
        pass


if __name__ == '__main__':
    print("ResNet Example")
    rng = jax.random.key(0)
    model = ResNet(num_classes=10)
    x = jnp.ones((4, 32, 32, 3))
    variables = model.init(rng, x, training=True)
    output = model.apply(variables, x, training=False)
    print(f"Output shape: {output.shape}")
