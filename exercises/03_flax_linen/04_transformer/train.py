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
    """Create training state with warmup + cosine decay + gradient clipping.

    Args:
        rng: JAX random key
        model: TransformerEncoder model
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total training steps for cosine decay

    Returns:
        TrainState with optimizer configured
    """
    # TODO: Implement create_train_state
    # 1. Create dummy input for model initialization
    #    Hint: Use jax.random.randint to create token indices
    # 2. Initialize model variables
    # 3. Create learning rate schedule using optax.warmup_cosine_decay_schedule
    #    - init_value=0.0, peak_value=learning_rate
    #    - warmup_steps, decay_steps=total_steps
    # 4. Create optimizer chain with:
    #    - Gradient clipping by global norm (1.0)
    #    - Adam optimizer with the schedule
    # 5. Return TrainState.create with apply_fn, params, and tx
    pass


if __name__ == '__main__':
    print("Transformer Training Setup")
    model = TransformerEncoder(vocab_size=1000, num_classes=10)
    rng = jax.random.key(0)
    state = create_train_state(rng, model)
    print(f"State created, step={state.step}")
