"""
MLP (Multi-Layer Perceptron) in Flax Linen
==========================================

Flax concepts demonstrated:
- nn.Dense layers
- nn.relu activation
- nn.BatchNorm and nn.Dropout
- @nn.compact decorator pattern
- model.init(rng, input_example) initialization
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable


class MLP(nn.Module):
    """
    Basic MLP with configurable hidden layers.

    Attributes:
        features: Sequence of hidden layer sizes
        output_dim: Number of output features
        activation: Activation function (default: relu)
    """
    features: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)
            training: Whether in training mode (affects dropout)

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = self.activation(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class MLPWithBatchNorm(nn.Module):
    """
    MLP with BatchNorm for better training dynamics.
    """
    features: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = self.activation(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class MLPWithDropout(nn.Module):
    """
    MLP with Dropout for regularization.
    """
    features: Sequence[int]
    output_dim: int
    dropout_rate: float = 0.5
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True, *, rngs=None):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class MLPClassifier(nn.Module):
    """
    Complete MLP classifier with BatchNorm and Dropout.
    Suitable for image classification tasks like MNIST.
    """
    hidden_dims: Sequence[int] = (256, 128)
    num_classes: int = 10
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Flatten input if needed (e.g., images)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer (no activation for logits)
        x = nn.Dense(self.num_classes)(x)
        return x


def create_mlp(features: Sequence[int], output_dim: int):
    """Factory function to create MLP."""
    return MLP(features=features, output_dim=output_dim)


def init_mlp(rng, model, input_shape):
    """
    Initialize MLP parameters.

    Args:
        rng: JAX random key
        model: Flax model instance
        input_shape: Shape of input (excluding batch)

    Returns:
        Initialized parameters
    """
    dummy_input = jnp.ones((1,) + input_shape)
    variables = model.init(rng, dummy_input, training=False)
    return variables


if __name__ == '__main__':
    # Example usage
    print("MLP Examples")
    print("=" * 50)

    # Basic MLP
    rng = jax.random.key(0)
    model = MLP(features=[256, 128], output_dim=10)

    # Initialize
    dummy_input = jnp.ones((1, 784))  # MNIST-like input
    variables = model.init(rng, dummy_input)

    print(f"\nBasic MLP:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Parameters: {jax.tree.map(lambda x: x.shape, variables['params'])}")

    # Forward pass
    output = model.apply(variables, dummy_input)
    print(f"  Output shape: {output.shape}")

    # MLP with BatchNorm
    model_bn = MLPWithBatchNorm(features=[256, 128], output_dim=10)
    variables_bn = model_bn.init(rng, dummy_input, training=True)

    print(f"\nMLP with BatchNorm:")
    print(f"  Has batch_stats: {'batch_stats' in variables_bn}")

    # MLP Classifier
    model_clf = MLPClassifier(hidden_dims=[256, 128], num_classes=10)
    variables_clf = model_clf.init(rng, dummy_input, training=True)

    print(f"\nMLP Classifier:")
    output_clf = model_clf.apply(variables_clf, dummy_input, training=False)
    print(f"  Output shape: {output_clf.shape}")
