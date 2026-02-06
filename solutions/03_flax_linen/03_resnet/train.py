"""
Training ResNet with Optax
==========================

Uses SGD with momentum and cosine decay (classic ResNet training).
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import ResNet
import numpy as np


def create_train_state(rng, model, input_shape, learning_rate=0.1, momentum=0.9):
    """Create training state with SGD + momentum + cosine decay."""
    variables = model.init(rng, jnp.ones((1,) + input_shape), training=True)

    # Cosine decay schedule (from 02_optax/02_lr_schedules)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=10000
    )

    # SGD with momentum (from 02_optax/01_basic_optimizers)
    tx = optax.sgd(schedule, momentum=momentum)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables.get('batch_stats', {})


if __name__ == '__main__':
    print("ResNet Training Setup")
    model = ResNet(num_classes=10)
    rng = jax.random.key(0)
    state, batch_stats = create_train_state(rng, model, (32, 32, 3))
    print(f"State created, step={state.step}")
