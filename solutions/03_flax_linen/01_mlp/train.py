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
    dummy_input = jnp.ones((1,) + input_shape)
    variables = model.init(rng, dummy_input, training=True)

    # Use Adam optimizer (from 02_optax/01_basic_optimizers)
    tx = optax.adam(learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables.get('batch_stats', {})


def compute_loss(params, batch_stats, model, batch, training=True):
    """
    Compute cross-entropy loss.

    Uses optax.softmax_cross_entropy_with_integer_labels
    (from 02_optax/04_losses)
    """
    x, y = batch
    variables = {'params': params, 'batch_stats': batch_stats}

    logits, new_model_state = model.apply(
        variables,
        x,
        training=training,
        mutable=['batch_stats'] if training else False
    )

    # Cross-entropy loss (from 02_optax/04_losses Example 2)
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, y)
    )

    return loss, (logits, new_model_state)


def compute_metrics(logits, labels):
    """Compute accuracy and loss metrics."""
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step(state, batch_stats, model, batch):
    """
    Single training step.

    Pattern from 02_optax/05_training_utils Example 1.
    """
    def loss_fn(params):
        return compute_loss(params, batch_stats, model, batch, training=True)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch[1])

    new_batch_stats = new_model_state['batch_stats'] if 'batch_stats' in new_model_state else batch_stats

    return state, new_batch_stats, metrics


@jax.jit
def eval_step(state, batch_stats, model, batch):
    """Evaluation step (no gradient computation)."""
    _, (logits, _) = compute_loss(
        state.params, batch_stats, model, batch, training=False
    )
    return compute_metrics(logits, batch[1])


def create_synthetic_data(num_samples, input_dim, num_classes, seed=42):
    """Create synthetic classification data."""
    np.random.seed(seed)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y


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
    # Create model
    model = MLPClassifier(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=0.3
    )

    # Initialize
    rng = jax.random.key(0)
    state, batch_stats = create_train_state(
        rng, model, (input_dim,), learning_rate
    )

    # Create synthetic data
    train_X, train_y = create_synthetic_data(1000, input_dim, num_classes)
    test_X, test_y = create_synthetic_data(200, input_dim, num_classes, seed=123)

    # Training loop (from 02_optax/05_training_utils Example 1)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # Shuffle training data
        perm = np.random.permutation(len(train_X))
        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]

        # Training
        epoch_metrics = []
        for i in range(0, len(train_X), batch_size):
            batch = (
                jnp.array(train_X_shuffled[i:i+batch_size]),
                jnp.array(train_y_shuffled[i:i+batch_size])
            )
            state, batch_stats, metrics = train_step(state, batch_stats, model, batch)
            epoch_metrics.append(metrics)

        # Average training metrics
        train_loss = np.mean([m['loss'] for m in epoch_metrics])
        train_acc = np.mean([m['accuracy'] for m in epoch_metrics])

        # Evaluation
        test_batch = (jnp.array(test_X), jnp.array(test_y))
        test_metrics = eval_step(state, batch_stats, model, test_batch)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['test_loss'].append(float(test_metrics['loss']))
        history['test_acc'].append(float(test_metrics['accuracy']))

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"test_loss={test_metrics['loss']:.4f}, test_acc={test_metrics['accuracy']:.4f}")

    return state, batch_stats, history


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
