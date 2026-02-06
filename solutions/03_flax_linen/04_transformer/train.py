"""
Training Transformer with Optax
================================

Uses warmup + cosine decay schedule (critical for Transformers).
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import TransformerEncoder
import numpy as np


def create_train_state(rng, model, learning_rate=0.0001, warmup_steps=4000, total_steps=100000):
    """Create training state with warmup + cosine decay + gradient clipping."""
    dummy_input = jax.random.randint(rng, (1, 32), 0, model.vocab_size)
    variables = model.init(rng, dummy_input, training=True)

    # Warmup + Cosine decay (from 02_optax/02_lr_schedules Example 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps
    )

    # Chain: clip + Adam (from 02_optax/03_gradient_clipping Example 10)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule)
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )


if __name__ == '__main__':
    print("Transformer Training Setup")
    model = TransformerEncoder(vocab_size=1000, num_classes=10)
    rng = jax.random.key(0)
    state = create_train_state(rng, model)
    print(f"State created, step={state.step}")
