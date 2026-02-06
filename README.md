# Jax-Gym

Hands-on practice exercises for learning JAX, Flax, and Optax. Each topic has exercises with pytest-based tests so you can verify your solutions.

## Structure

```
exercises/          # Fill-in-the-blank exercises
  01_jax_core/      # jnp basics, jit, grad, vmap, pytrees, random, control flow, etc.
  02_optax/         # Optimizers, LR schedules, gradient clipping, losses
  03_flax_linen/    # MLP, CNN, ResNet, Transformer

solutions/          # Reference implementations
  01_jax_core/
  02_optax/
  03_flax_linen/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Work through exercises in order. Each exercise has a corresponding test file — run tests to check your work:

```bash
# Run tests for a specific exercise
pytest exercises/01_jax_core/01_jnp_basics/test_exercises.py

# Run all tests in a section
pytest exercises/01_jax_core/

# Check solutions
pytest solutions/01_jax_core/01_jnp_basics/test_examples.py
```

## Topics Covered

**JAX Core** — jnp basics, JIT compilation, autodiff (grad), vmap, pytrees, PRNG, control flow, stateful computations, lax.scan, einops, AOT compilation

**Optax** — SGD/Adam/AdamW, learning rate schedules, gradient clipping, loss functions, training loop patterns

**Flax Linen** — MLP, CNN, ResNet, Transformer (multi-head attention, positional encoding, encoder blocks)
