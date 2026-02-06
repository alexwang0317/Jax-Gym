"""
Training MLP with Optax
=======================

Optax concepts used (from 02_optax):
- optax.adam (from 01_basic_optimizers)
- optax.softmax_cross_entropy_with_integer_labels (from 04_losses)
- Basic training loop (from 05_training_utils)
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import MLPClassifier
import numpy as np


def create_train_state(rng, model, input_shape, learning_rate):
    """
    Create initial training state.

    Args:
        rng: Random key for initialization
        model: Flax model
        input_shape: Shape of input (excluding batch)
        learning_rate: Learning rate for optimizer

    Returns:
        TrainState with initialized params
    """
    # TODO: Implement
    # 1. Create dummy_input = jnp.ones((1,) + input_shape)
    # 2. Initialize variables = model.init(rng, dummy_input, training=True)
    # 3. Create optimizer tx = optax.adam(learning_rate)
    # 4. Create and return train_state.TrainState.create(
    #        apply_fn=model.apply,
    #        params=variables['params'],
    #        tx=tx
    #    ), variables.get('batch_stats', {})
    pass


def compute_loss(params, batch_stats, model, batch, training=True):
    """
    Compute cross-entropy loss.

    Uses optax.softmax_cross_entropy_with_integer_labels
    (from 02_optax/04_losses)
    """
    # TODO: Implement
    # 1. Unpack batch: x, y = batch
    # 2. Create variables dict: variables = {'params': params, 'batch_stats': batch_stats}
    # 3. Apply model:
    #    logits, new_model_state = model.apply(
    #        variables,
    #        x,
    #        training=training,
    #        mutable=['batch_stats'] if training else False
    #    )
    # 4. Compute loss:
    #    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    # 5. Return loss, (logits, new_model_state)
    pass


def compute_metrics(logits, labels):
    """Compute accuracy and loss metrics."""
    # TODO: Implement
    # 1. Compute loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    # 2. Compute accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    # 3. Return {'loss': loss, 'accuracy': accuracy}
    pass


@jax.jit
def train_step(state, batch_stats, model, batch):
    """
    Single training step.

    Pattern from 02_optax/05_training_utils Example 1.
    """
    # TODO: Implement
    # 1. Define loss_fn(params) that calls compute_loss
    # 2. Compute gradients with jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # 3. Update state with state = state.apply_gradients(grads=grads)
    # 4. Compute metrics
    # 5. Extract new_batch_stats from new_model_state
    # 6. Return state, new_batch_stats, metrics
    pass


@jax.jit
def eval_step(state, batch_stats, model, batch):
    """Evaluation step (no gradient computation)."""
    # TODO: Implement
    # 1. Call compute_loss with training=False
    # 2. Return compute_metrics(logits, batch[1])
    pass


def create_synthetic_data(num_samples, input_dim, num_classes, seed=42):
    """Create synthetic classification data."""
    # TODO: Implement
    # 1. Set np.random.seed(seed)
    # 2. Create X = np.random.randn(num_samples, input_dim).astype(np.float32)
    # 3. Create y = np.random.randint(0, num_classes, size=num_samples)
    # 4. Return X, y
    pass


def train_mlp(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    input_dim: int = 784,
    hidden_dims: tuple = (256, 128),
    num_classes: int = 10,
    verbose: bool = True
):
    """
    Train MLP on synthetic data.

    Demonstrates the full training pipeline using Optax.
    """
    # TODO: Implement
    # 1. Create model = MLPClassifier(hidden_dims=hidden_dims, num_classes=num_classes, dropout_rate=0.3)
    # 2. Initialize rng = jax.random.key(0)
    # 3. Create state, batch_stats = create_train_state(rng, model, (input_dim,), learning_rate)
    # 4. Create synthetic data for train and test
    # 5. Initialize history dict
    # 6. Training loop:
    #    - Shuffle training data each epoch
    #    - Loop through batches and call train_step
    #    - Compute average training metrics
    #    - Evaluate on test data with eval_step
    #    - Record metrics in history
    #    - Print progress if verbose
    # 7. Return state, batch_stats, history
    pass


if __name__ == '__main__':
    print("Training MLP")
    print("=" * 50)

    state, batch_stats, history = train_mlp(
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        verbose=True
    )

    print(f"\nFinal Results:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"  Test Accuracy: {history['test_acc'][-1]:.4f}")
