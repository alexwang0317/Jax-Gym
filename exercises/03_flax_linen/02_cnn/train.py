"""
Training CNN with Optax
=======================

Optax concepts used:
- optax.adamw with weight decay (from 01_basic_optimizers)
- optax.exponential_decay schedule (from 02_lr_schedules)
- optax.clip_by_global_norm (from 03_gradient_clipping)
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import CNNWithBatchNorm
import numpy as np


def create_train_state(rng, model, input_shape, learning_rate, weight_decay=0.0001):
    """
    Create training state with AdamW and gradient clipping.

    Args:
        rng: JAX random key for initialization.
        model: Flax model instance.
        input_shape: Shape of input data (excluding batch dimension).
        learning_rate: Initial learning rate.
        weight_decay: Weight decay coefficient for AdamW.

    Returns:
        Tuple of (TrainState, batch_stats dict).

    Implementation steps:
    1. Create dummy input and initialize model variables
    2. Create exponential decay schedule (init_value=learning_rate,
       transition_steps=1000, decay_rate=0.96)
    3. Chain gradient clipping (clip_by_global_norm(1.0)) with AdamW optimizer
    4. Create and return TrainState with batch_stats
    """
    # TODO: Create dummy input with shape (1,) + input_shape
    dummy_input = None

    # TODO: Initialize model variables with training=True
    variables = None

    # TODO: Create exponential decay learning rate schedule
    # - init_value=learning_rate
    # - transition_steps=1000
    # - decay_rate=0.96
    schedule = None

    # TODO: Chain gradient clipping with AdamW optimizer
    # - optax.clip_by_global_norm(1.0)
    # - optax.adamw(schedule, weight_decay=weight_decay)
    tx = None

    # TODO: Create TrainState and return with batch_stats
    # - train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    # - Return tuple of (state, variables.get('batch_stats', {}))
    return None, {}


def train_cnn(num_epochs=5, batch_size=32, verbose=True):
    """
    Train CNN on synthetic data.

    Args:
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        verbose: Whether to print training progress.

    Returns:
        Tuple of (final TrainState, batch_stats dict).

    Implementation steps:
    1. Create model and training state
    2. Generate synthetic training data
    3. Define JIT-compiled train_step function
    4. Run training loop over epochs and batches
    """
    # TODO: Create model and training state
    model = None
    rng = None
    state, batch_stats = None, {}

    # TODO: Generate synthetic data
    # - train_X: shape (500, 32, 32, 3), float32
    # - train_y: shape (500,), integers 0-9
    np.random.seed(42)
    train_X = None
    train_y = None

    @jax.jit
    def train_step(state, batch_stats, batch):
        """
        Perform a single training step.

        Args:
            state: Current TrainState.
            batch_stats: Current batch normalization statistics.
            batch: Tuple of (inputs, labels).

        Returns:
            Tuple of (updated state, updated batch_stats, loss, accuracy).
        """
        def loss_fn(params):
            # TODO: Apply model with training=True and mutable=['batch_stats']
            # - model.apply({'params': params, 'batch_stats': batch_stats},
            #               batch[0], training=True, mutable=['batch_stats'])
            logits, new_state = None, None

            # TODO: Compute cross-entropy loss
            # - optax.softmax_cross_entropy_with_integer_labels(logits, batch[1])
            # - Take mean over batch
            loss = None

            return loss, (logits, new_state)

        # TODO: Compute gradients using jax.value_and_grad with has_aux=True
        (loss, (logits, new_state)), grads = None, None

        # TODO: Apply gradients to update state
        state = None

        # TODO: Compute accuracy
        acc = None

        return state, new_state.get('batch_stats', batch_stats), loss, acc

    # TODO: Implement training loop
    # - For each epoch:
    #   - Shuffle data with np.random.permutation
    #   - For each batch:
    #     - Create batch tuple of (jnp.array(inputs), jnp.array(labels))
    #     - Call train_step
    #     - Accumulate loss and accuracy
    #   - Print epoch statistics if verbose
    for epoch in range(num_epochs):
        pass

    return state, batch_stats


if __name__ == '__main__':
    print("Training CNN")
    print("=" * 50)
    train_cnn(num_epochs=5)
