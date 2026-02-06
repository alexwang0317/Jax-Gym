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
        # TODO: Implement forward pass
        # 1. Loop through self.features and for each:
        #    - Apply nn.Dense(feat)(x)
        #    - Apply self.activation(x)
        # 2. Apply final nn.Dense(self.output_dim)(x)
        # 3. Return the output
        pass


class MLPWithBatchNorm(nn.Module):
    """
    MLP with BatchNorm for better training dynamics.
    """
    features: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        # TODO: Implement forward pass with BatchNorm
        # 1. Loop through self.features and for each:
        #    - Apply nn.Dense(feat)(x)
        #    - Apply nn.BatchNorm(use_running_average=not training)(x)
        #    - Apply self.activation(x)
        # 2. Apply final nn.Dense(self.output_dim)(x)
        # 3. Return the output
        pass


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
        # TODO: Implement forward pass with Dropout
        # 1. Loop through self.features and for each:
        #    - Apply nn.Dense(feat)(x)
        #    - Apply self.activation(x)
        #    - Apply nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        # 2. Apply final nn.Dense(self.output_dim)(x)
        # 3. Return the output
        pass


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
        # TODO: Implement forward pass
        # 1. Flatten input if needed (if x.ndim > 2):
        #    x = x.reshape((x.shape[0], -1))
        # 2. Loop through self.hidden_dims and for each:
        #    - Apply nn.Dense(dim)(x)
        #    - Apply nn.BatchNorm(use_running_average=not training)(x)
        #    - Apply nn.relu(x)
        #    - Apply nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        # 3. Apply output layer nn.Dense(self.num_classes)(x) (no activation for logits)
        # 4. Return the output
        pass


def create_mlp(features: Sequence[int], output_dim: int):
    """Factory function to create MLP."""
    # TODO: Implement
    # Return MLP(features=features, output_dim=output_dim)
    pass


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
    # TODO: Implement
    # 1. Create dummy_input = jnp.ones((1,) + input_shape)
    # 2. Initialize with variables = model.init(rng, dummy_input, training=False)
    # 3. Return variables
    pass


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
