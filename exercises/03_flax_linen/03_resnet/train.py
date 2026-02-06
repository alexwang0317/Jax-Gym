"""
Training ResNet with Optax
==========================

Uses SGD with momentum and cosine decay (classic ResNet training).

Exercise: Implement the create_train_state function.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import ResNet
import numpy as np


def create_train_state(rng, model, input_shape, learning_rate=0.1, momentum=0.9):
    """Create training state with SGD + momentum + cosine decay.

    This function sets up the complete training state including:
    1. Model initialization
    2. Learning rate schedule (cosine decay)
    3. Optimizer (SGD with momentum)

    Args:
        rng: JAX random key for initialization.
        model: The Flax model to train.
        input_shape: Shape of a single input sample (H, W, C).
        learning_rate: Initial learning rate. Default is 0.1.
        momentum: Momentum coefficient for SGD. Default is 0.9.

    Returns:
        A tuple of (TrainState, batch_stats):
        - TrainState: Contains apply_fn, params, tx, and step.
        - batch_stats: BatchNorm statistics (empty dict if not present).

    Hints:
        - Initialize model with shape (1,) + input_shape and training=True
        - Use optax.cosine_decay_schedule with decay_steps=10000
        - Use optax.sgd with the schedule and momentum
        - Extract 'params' from variables for TrainState
        - Extract 'batch_stats' from variables (use .get() with default {})
    """
    # TODO: Implement the training state creation
    # 1. Initialize model variables
    # 2. Create cosine decay learning rate schedule
    # 3. Create SGD optimizer with momentum
    # 4. Create and return TrainState and batch_stats
    pass


if __name__ == '__main__':
    print("ResNet Training Setup")
    model = ResNet(num_classes=10)
    rng = jax.random.key(0)
    state, batch_stats = create_train_state(rng, model, (32, 32, 3))
    print(f"State created, step={state.step}")
