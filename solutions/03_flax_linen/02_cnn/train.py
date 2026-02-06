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
    """Create training state with AdamW and gradient clipping."""
    dummy_input = jnp.ones((1,) + input_shape)
    variables = model.init(rng, dummy_input, training=True)

    # AdamW with weight decay (from 02_optax/01_basic_optimizers Example 4)
    # Exponential decay (from 02_optax/02_lr_schedules Example 2)
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.96
    )

    # Chain with gradient clipping (from 02_optax/03_gradient_clipping)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay)
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables.get('batch_stats', {})


def train_cnn(num_epochs=5, batch_size=32, verbose=True):
    """Train CNN on synthetic data."""
    model = CNNWithBatchNorm(num_classes=10)
    rng = jax.random.key(0)

    state, batch_stats = create_train_state(
        rng, model, (32, 32, 3), learning_rate=0.001
    )

    # Synthetic data
    np.random.seed(42)
    train_X = np.random.randn(500, 32, 32, 3).astype(np.float32)
    train_y = np.random.randint(0, 10, size=500)

    @jax.jit
    def train_step(state, batch_stats, batch):
        def loss_fn(params):
            logits, new_state = model.apply(
                {'params': params, 'batch_stats': batch_stats},
                batch[0], training=True, mutable=['batch_stats']
            )
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, batch[1])
            )
            return loss, (logits, new_state)

        (loss, (logits, new_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        state = state.apply_gradients(grads=grads)
        acc = jnp.mean(jnp.argmax(logits, -1) == batch[1])

        return state, new_state.get('batch_stats', batch_stats), loss, acc

    # Training loop
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(train_X))
        epoch_loss, epoch_acc = [], []

        for i in range(0, len(train_X), batch_size):
            batch = (
                jnp.array(train_X[perm[i:i+batch_size]]),
                jnp.array(train_y[perm[i:i+batch_size]])
            )
            state, batch_stats, loss, acc = train_step(state, batch_stats, batch)
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        if verbose:
            print(f"Epoch {epoch+1}: loss={np.mean(epoch_loss):.4f}, acc={np.mean(epoch_acc):.4f}")

    return state, batch_stats


if __name__ == '__main__':
    print("Training CNN")
    print("=" * 50)
    train_cnn(num_epochs=5)
