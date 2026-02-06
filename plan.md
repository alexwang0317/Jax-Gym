# JAX Practice Gym - Implementation Plan

## Overview
A comprehensive JAX learning repository with hands-on examples and tests covering:
1. **JAX Core** - Following JAX 101 + UvA DLC tutorial topics
2. **Flax Linen** - Neural network architectures with practical tasks
3. **Optax** - Optimization and training loops

## Reference Materials
- [JAX 101 Official Docs](https://docs.jax.dev/en/latest/jax-101.html)
- [UvA DLC JAX Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html)
- [UvA DLC Transformers Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)

## Directory Structure

```
Jax-Gym/
├── 01_jax_core/
│   ├── 01_jnp_basics/
│   │   ├── examples.py          # 10 jnp examples
│   │   └── test_examples.py
│   ├── 02_jit/
│   │   ├── examples.py          # 10 JIT examples
│   │   └── test_examples.py
│   ├── 03_grad/
│   │   ├── examples.py          # 10 autodiff examples
│   │   └── test_examples.py
│   ├── 04_vmap/
│   │   ├── examples.py          # 10 vectorization examples
│   │   └── test_examples.py
│   ├── 05_pytrees/
│   │   ├── examples.py          # 10 pytree examples
│   │   └── test_examples.py
│   ├── 06_random/
│   │   ├── examples.py          # 10 PRNG examples
│   │   └── test_examples.py
│   ├── 07_control_flow/
│   │   ├── examples.py          # 10 control flow examples
│   │   └── test_examples.py
│   ├── 08_stateful/
│   │   ├── examples.py          # 10 stateful computation examples
│   │   └── test_examples.py
│   └── 09_lax_scan/
│       ├── examples.py          # 10 lax.scan examples
│       └── test_examples.py
├── 02_flax_linen/
│   ├── 01_mlp/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test_model.py
│   ├── 02_cnn/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test_model.py
│   ├── 03_resnet/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test_model.py
│   ├── 04_transformer/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test_model.py
│   ├── 05_rnn_lstm/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test_model.py
│   └── 06_autoencoder/
│       ├── model.py
│       ├── train.py
│       └── test_model.py
├── 03_optax/
│   ├── 01_basic_optimizers/
│   │   ├── examples.py          # SGD, Adam, AdamW, RMSProp, etc.
│   │   └── test_examples.py
│   ├── 02_lr_schedules/
│   │   ├── examples.py          # warmup, cosine decay, exponential
│   │   └── test_examples.py
│   ├── 03_gradient_clipping/
│   │   ├── examples.py
│   │   └── test_examples.py
│   ├── 04_losses/
│   │   ├── examples.py
│   │   └── test_examples.py
│   └── 05_training_utils/
│       ├── examples.py          # Training loop patterns
│       └── test_examples.py
├── requirements.txt
└── run_all_tests.py
```

---

## Part 1: JAX Core (90 examples total)

### 1.1 jnp_basics (10 examples)
1. Array creation: `jnp.array`, `jnp.zeros`, `jnp.ones`, `jnp.arange`
2. Array reshaping and slicing
3. Mathematical operations: `jnp.sin`, `jnp.exp`, `jnp.log`
4. Linear algebra: `jnp.dot`, `jnp.matmul`
5. Broadcasting rules
6. Reduction operations: `jnp.sum`, `jnp.mean`, `jnp.max`
7. Boolean indexing and `jnp.where`
8. Immutability & `.at[].set()` pattern (no in-place ops)
9. `jnp.einsum` operations
10. Device placement: `jax.devices()`, `jax.device_put()`

### 1.2 jit (10 examples)
1. Basic `@jit` decorator usage
2. Timing comparison: jitted vs non-jitted (10-15x speedup demo)
3. `static_argnums` for shape-dependent code
4. `donate_argnums` for memory efficiency
5. JIT caching and recompilation behavior
6. Debugging with `jax.disable_jit()`
7. `jax.make_jaxpr()` to visualize computation graphs
8. JIT and side effects (what NOT to do - print statements)
9. Pure functions requirement for JIT
10. Dynamic shapes and recompilation costs

### 1.3 grad (10 examples)
1. Basic `grad` for scalar functions
2. `value_and_grad` for efficiency
3. `grad` with `argnums` for multiple inputs
4. Higher-order derivatives with nested `grad`
5. Jacobian with `jax.jacfwd` and `jax.jacrev`
6. Hessian computation
7. `grad` through control flow
8. Custom gradients with `jax.custom_vjp`
9. Stop gradient with `jax.lax.stop_gradient`
10. Gradient of loss functions (MSE, cross-entropy)

### 1.4 vmap (10 examples)
1. Basic `vmap` for batching
2. `in_axes` and `out_axes` specification
3. `vmap` over multiple arguments
4. Nested `vmap` for higher-rank batching
5. `vmap` combined with `jit`
6. `vmap` combined with `grad` (per-example gradients)
7. `vmap` for matrix-vector products
8. `vmap` for pairwise distances
9. `vmap` with None axes (broadcasting)
10. `vmap` for attention score computation

### 1.5 pytrees (10 examples)
1. Basic pytree structure (nested dicts/lists)
2. `jax.tree.map` for element-wise operations
3. `jax.tree.leaves` and `jax.tree.structure`
4. `jax.tree.reduce` for aggregation
5. Custom pytree nodes with `register_pytree_node`
6. Pytrees for neural network parameters
7. `jax.tree.unflatten` and `jax.tree.flatten`
8. Pytree transformations with `jit` and `grad`
9. Combining multiple pytrees
10. Pytree utilities: `tree_all`, `tree_any`

### 1.6 random (10 examples)
1. Key creation with `jax.random.key`
2. Key splitting: `jax.random.split`
3. `jax.random.uniform` and `jax.random.normal`
4. `jax.random.choice` for sampling
5. `jax.random.permutation` for shuffling
6. Reproducibility patterns
7. Random in `vmap` (unique keys per batch)
8. Random in `jit` (passing keys explicitly)
9. Dropout implementation with random
10. Weight initialization patterns

### 1.7 control_flow (10 examples)
1. `jax.lax.cond` for if-else
2. `jax.lax.switch` for multi-branch
3. `jax.lax.while_loop` basics
4. `jax.lax.fori_loop` for fixed iterations
5. Nested control flow
6. Control flow with `grad`
7. Python control flow vs JAX control flow
8. `jax.lax.select` for element-wise conditionals
9. Early stopping patterns
10. Control flow performance considerations

### 1.8 stateful (10 examples)
1. Functional state pattern (pass state explicitly)
2. State as pytree
3. Accumulator patterns
4. Counter implementation
5. Running statistics (mean, variance)
6. Stateful RNG handling
7. Optimizer state management
8. BatchNorm statistics
9. Memoization patterns
10. State checkpointing

### 1.9 lax_scan (10 examples)
1. Basic `lax.scan` for cumulative sum
2. `lax.scan` vs Python loops (performance)
3. Carrying state through scan
4. `lax.scan` for RNN forward pass
5. Reverse scan
6. `lax.scan` with variable-length sequences
7. `lax.scan` combined with `vmap`
8. Unrolling with `lax.scan`
9. `lax.scan` for time series processing
10. Checkpointed scan for memory efficiency

---

## Part 2: Flax Linen (6 architectures)

### 2.1 MLP (Multi-Layer Perceptron)
- `nn.Dense`, `nn.relu`, `nn.BatchNorm`, `nn.Dropout`
- `@nn.compact` decorator pattern
- `model.init(rng, input_example)` initialization
- `TrainState` for bundling params + optimizer + apply_fn
- **Practical Tasks**:
  - XOR classifier (continuous version from UvA)
  - MNIST classification
- Tests: forward pass shapes, parameter count, training convergence

### 2.2 CNN (Convolutional Neural Network)
- `nn.Conv`, `nn.max_pool`, `nn.avg_pool`
- CIFAR-10 style architecture
- Tests: convolution output shapes, feature maps

### 2.3 ResNet
- Residual blocks with skip connections
- `nn.Sequential`, custom Module composition
- Tests: gradient flow, identity shortcuts

### 2.4 Transformer (detailed from UvA tutorial)
- **Scaled Dot Product Attention**: `softmax(QK^T / √d_k) * V`
- **Multi-Head Attention Module**: Xavier init, parallel heads
- **Encoder Block**: MHA + residual + LayerNorm + FFN (Linear→ReLU→Linear)
- **Positional Encoding**: Sine/cosine positional patterns
- **TransformerEncoder**: Stacked encoder blocks with attention map extraction
- **Practical Tasks**:
  - Sequence reversal (test long-term dependencies)
  - Set anomaly detection (permutation equivariance)
- Tests: attention masks, shape validation, gradient flow

### 2.5 RNN/LSTM
- `nn.RNN`, `nn.LSTMCell`, `nn.GRUCell`
- Sequence-to-sequence patterns
- Tests: hidden state propagation

### 2.6 Autoencoder
- Encoder-decoder with bottleneck
- VAE variant with reparameterization
- Tests: reconstruction, latent space

---

## Part 3: Optax (5 modules)

### 3.1 Basic Optimizers
- `optax.sgd`, `optax.adam`, `optax.adamw`
- `optax.rmsprop`, `optax.adagrad`
- Momentum variants
- Tests: parameter update correctness

### 3.2 Learning Rate Schedules
- `optax.warmup_cosine_decay_schedule` (essential for Transformers)
- `optax.exponential_decay`
- `optax.piecewise_constant_schedule`
- `optax.linear_schedule`, `optax.join_schedules`
- Custom schedule composition
- Tests: schedule values at checkpoints, visualization

### 3.3 Gradient Clipping
- `optax.clip_by_global_norm`
- `optax.clip_by_value`
- Combining clipping with optimizers via `optax.chain`
- Tests: gradient norm bounds

### 3.4 Losses
- `optax.softmax_cross_entropy`
- `optax.sigmoid_binary_cross_entropy`
- `optax.l2_loss`, `optax.huber_loss`
- Tests: loss value verification

### 3.5 Training Utilities
- Complete training loop patterns
- `optax.apply_updates` for parameter updates
- PyTorch DataLoader integration with `numpy_collate`
- Train/eval mode handling (`deterministic`, `use_running_average`)
- Metric tracking and logging

---

## Implementation Order

1. Create directory structure and `requirements.txt`
2. Implement JAX Core modules (01-09) with tests
3. Implement Flax architectures with training scripts
4. Implement Optax modules with tests
5. Create `run_all_tests.py` for full test suite

## Requirements

```
jax[cpu]>=0.4.20
flax>=0.8.0
optax>=0.1.7
pytest>=7.0.0
numpy>=1.24.0
torch>=2.0.0          # For DataLoader integration
torchvision>=0.15.0   # For datasets (MNIST, CIFAR)
matplotlib>=3.7.0     # For visualizations
```

## Verification

- Run `pytest` in each module directory
- Run `python run_all_tests.py` for complete suite
- Each example should be runnable standalone
- Tests verify correctness of shapes, values, and gradients
