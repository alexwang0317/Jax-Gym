"""
Optax Training Utilities - 10 Examples
=======================================

Patterns and utilities for training neural networks with JAX and Optax.
These patterns are used throughout the Flax examples.

Reference: https://flax.readthedocs.io/
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any, Callable
import numpy as np


# =============================================================================
# Example 1: Basic Training Loop Pattern
# =============================================================================
def example_basic_training_loop():
    """
    The fundamental training loop pattern in JAX.
    """
    # Simple model
    def model(params, x):
        return params['w'] @ x + params['b']

    # Loss function
    def loss_fn(params, x, y):
        pred = model(params, x)
        return jnp.mean((pred - y) ** 2)

    # Initialize
    params = {
        'w': jnp.ones((2, 3)),
        'b': jnp.zeros(2)
    }

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    # Training step function
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    losses = []
    for step in range(100):
        # Dummy data
        x = jnp.ones(3)
        y = jnp.array([1.0, 2.0])

        params, opt_state, loss = train_step(params, opt_state, x, y)
        losses.append(float(loss))

    return {
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'loss_decreased': losses[-1] < losses[0]
    }


# =============================================================================
# Example 2: TrainState from Flax
# =============================================================================
def example_train_state():
    """
    Flax's TrainState bundles params, optimizer state, and step count.
    Much cleaner than managing them separately.
    """
    # Simple params
    params = {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)}

    # Create optimizer with schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.001,
        warmup_steps=100,
        decay_steps=1000
    )
    tx = optax.adam(schedule)

    # Create TrainState
    state = train_state.TrainState.create(
        apply_fn=lambda p, x: p['w'] @ x + p['b'],  # Model forward function
        params=params,
        tx=tx
    )

    # Access components
    current_step = state.step
    current_params = state.params

    # Update state (returns new state)
    grads = {'w': jnp.ones((3, 4)) * 0.1, 'b': jnp.ones(4) * 0.01}
    new_state = state.apply_gradients(grads=grads)

    return {
        'initial_step': current_step,
        'after_update_step': new_state.step,
        'params_updated': not jnp.allclose(
            state.params['w'], new_state.params['w']
        )
    }


# =============================================================================
# Example 3: apply_updates Workflow
# =============================================================================
def example_apply_updates_workflow():
    """
    Detailed workflow of computing and applying updates.
    """
    params = {'layer1': jnp.ones((10, 20)), 'layer2': jnp.ones((20, 5))}

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(0.001, weight_decay=0.01)
    )
    opt_state = optimizer.init(params)

    # Compute gradients (from loss function)
    grads = {
        'layer1': jnp.ones((10, 20)) * 0.5,
        'layer2': jnp.ones((20, 5)) * 0.5
    }

    # Get updates from optimizer
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # Updates are the actual changes to apply
    # They may be scaled, momentum-adjusted, etc.

    # Apply updates: new_params = params + updates
    new_params = optax.apply_updates(params, updates)

    # Verify structure preserved
    same_structure = (
        set(params.keys()) == set(new_params.keys()) and
        params['layer1'].shape == new_params['layer1'].shape
    )

    return {
        'updates_layer1_norm': float(jnp.linalg.norm(updates['layer1'])),
        'grads_layer1_norm': float(jnp.linalg.norm(grads['layer1'])),
        'structure_preserved': same_structure,
        'params_changed': not jnp.allclose(params['layer1'], new_params['layer1'])
    }


# =============================================================================
# Example 4: PyTorch DataLoader Integration
# =============================================================================
def example_dataloader_integration():
    """
    Using PyTorch DataLoader with JAX.
    Common pattern for dataset handling.
    """
    # Numpy collate function for JAX
    def numpy_collate(batch):
        """Convert batch to numpy arrays (JAX-compatible)."""
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    # Simulated DataLoader behavior
    class SimpleDataLoader:
        def __init__(self, X, y, batch_size, shuffle=True):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            n = len(self.X)
            indices = np.arange(n)
            if self.shuffle:
                np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]
                yield self.X[batch_idx], self.y[batch_idx]

        def __len__(self):
            return (len(self.X) + self.batch_size - 1) // self.batch_size

    # Create dataset
    X = np.random.randn(1000, 10).astype(np.float32)
    y = np.random.randint(0, 5, size=1000)

    loader = SimpleDataLoader(X, y, batch_size=32)

    # Training loop with DataLoader
    batches_seen = 0
    for batch_X, batch_y in loader:
        # Convert to JAX arrays
        batch_X = jnp.array(batch_X)
        batch_y = jnp.array(batch_y)

        batches_seen += 1
        if batches_seen >= 5:
            break

    return {
        'total_samples': len(X),
        'batch_size': 32,
        'num_batches': len(loader),
        'batches_seen': batches_seen
    }


# =============================================================================
# Example 5: Train/Eval Mode Handling
# =============================================================================
def example_train_eval_mode():
    """
    Handling training vs evaluation mode.
    Affects Dropout, BatchNorm, etc.
    """
    # Simulated model with dropout and batchnorm
    def model_with_dropout(params, x, *, training: bool, rng=None):
        """
        Model that behaves differently in train vs eval.

        Args:
            training: If True, apply dropout
            rng: Random key for dropout (required if training=True)
        """
        h = x @ params['w1'] + params['b1']
        h = jax.nn.relu(h)

        if training:
            # Apply dropout during training
            assert rng is not None
            mask = jax.random.bernoulli(rng, 0.9, h.shape)
            h = h * mask / 0.9

        h = h @ params['w2'] + params['b2']
        return h

    # Initialize
    params = {
        'w1': jnp.ones((10, 20)) * 0.1,
        'b1': jnp.zeros(20),
        'w2': jnp.ones((20, 5)) * 0.1,
        'b2': jnp.zeros(5)
    }

    x = jnp.ones((1, 10))

    # Training mode
    rng = jax.random.key(42)
    train_output = model_with_dropout(params, x, training=True, rng=rng)

    # Eval mode (deterministic)
    eval_output = model_with_dropout(params, x, training=False)

    # Multiple eval calls should be identical
    eval_output2 = model_with_dropout(params, x, training=False)

    return {
        'train_output_shape': train_output.shape,
        'eval_output_shape': eval_output.shape,
        'eval_deterministic': jnp.allclose(eval_output, eval_output2)
    }


# =============================================================================
# Example 6: Metric Tracking and Logging
# =============================================================================
def example_metric_tracking():
    """
    Track and log training metrics.
    """
    class MetricTracker:
        def __init__(self):
            self.metrics = {}

        def update(self, name, value):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))

        def get_mean(self, name, window=None):
            values = self.metrics.get(name, [])
            if window:
                values = values[-window:]
            return np.mean(values) if values else 0.0

        def get_last(self, name):
            values = self.metrics.get(name, [])
            return values[-1] if values else 0.0

    # Training with metric tracking
    tracker = MetricTracker()

    for step in range(100):
        # Simulated metrics
        loss = 1.0 / (step + 1) + np.random.randn() * 0.1
        accuracy = min(0.95, 0.5 + step * 0.005 + np.random.randn() * 0.02)

        tracker.update('loss', loss)
        tracker.update('accuracy', accuracy)

        if (step + 1) % 20 == 0:
            avg_loss = tracker.get_mean('loss', window=20)
            avg_acc = tracker.get_mean('accuracy', window=20)
            # print(f"Step {step+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return {
        'final_loss': tracker.get_last('loss'),
        'final_accuracy': tracker.get_last('accuracy'),
        'avg_loss_last_20': tracker.get_mean('loss', window=20),
        'total_steps': len(tracker.metrics['loss'])
    }


# =============================================================================
# Example 7: Checkpointing (Simplified)
# =============================================================================
def example_checkpointing():
    """
    Save and restore training state.
    In practice, use orbax-checkpoint for robust checkpointing.
    """
    # Simulated checkpoint save/load
    def save_checkpoint(path, state):
        """Save state to checkpoint (simplified)."""
        return {
            'step': int(state.step),
            'params': jax.tree.map(lambda x: x.tolist(), state.params),
        }

    def load_checkpoint(checkpoint, tx):
        """Load state from checkpoint."""
        params = jax.tree.map(jnp.array, checkpoint['params'])
        state = train_state.TrainState.create(
            apply_fn=None,
            params=params,
            tx=tx
        )
        # Manually set step (TrainState doesn't allow this directly)
        return state.replace(step=checkpoint['step'])

    # Create initial state
    params = {'w': jnp.ones((3, 4))}
    tx = optax.adam(0.001)
    state = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

    # Simulate training
    for _ in range(10):
        grads = {'w': jnp.ones((3, 4)) * 0.1}
        state = state.apply_gradients(grads=grads)

    # Save checkpoint
    checkpoint = save_checkpoint('/tmp/ckpt', state)

    # Restore from checkpoint
    restored_state = load_checkpoint(checkpoint, tx)

    return {
        'saved_step': checkpoint['step'],
        'restored_step': restored_state.step,
        'params_match': jnp.allclose(
            state.params['w'],
            restored_state.params['w']
        )
    }


# =============================================================================
# Example 8: Early Stopping
# =============================================================================
def example_early_stopping():
    """
    Stop training when validation loss stops improving.
    """
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.counter = 0
            self.should_stop = False

        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            return self.should_stop

    # Simulate training
    early_stop = EarlyStopping(patience=3)

    # Simulated validation losses
    val_losses = [1.0, 0.8, 0.7, 0.65, 0.64, 0.64, 0.65, 0.66, 0.67, 0.68]

    stopped_at = None
    for epoch, val_loss in enumerate(val_losses):
        if early_stop(val_loss):
            stopped_at = epoch
            break

    return {
        'val_losses': val_losses,
        'stopped_at_epoch': stopped_at,
        'best_loss': early_stop.best_loss,
        'early_stopping_triggered': early_stop.should_stop
    }


# =============================================================================
# Example 9: Gradient Accumulation
# =============================================================================
def example_gradient_accumulation():
    """
    Accumulate gradients over multiple steps for effective larger batch.
    Useful when memory limited.
    """
    def create_accumulated_optimizer(tx, accumulation_steps):
        """Wrap optimizer with gradient accumulation."""
        return optax.MultiSteps(tx, every_k_schedule=accumulation_steps)

    # Effective batch size = actual_batch * accumulation_steps
    accumulation_steps = 4
    actual_batch_size = 8
    effective_batch_size = actual_batch_size * accumulation_steps

    optimizer = create_accumulated_optimizer(
        optax.adam(0.001),
        accumulation_steps
    )

    params = {'w': jnp.ones((10, 10))}
    opt_state = optimizer.init(params)

    # Simulate accumulation
    steps_before_update = 0
    params_changed = False

    original_params = params['w'].copy()
    for step in range(accumulation_steps + 1):
        grads = {'w': jnp.ones((10, 10)) * 0.1}
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        if not jnp.allclose(params['w'], original_params):
            params_changed = True
            steps_before_update = step
            break

        params = new_params

    return {
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': effective_batch_size,
        'steps_before_update': steps_before_update,
        'params_changed': params_changed
    }


# =============================================================================
# Example 10: Multi-GPU Training Basics (Conceptual)
# =============================================================================
def example_multi_gpu_basics():
    """
    Concepts for multi-GPU training with JAX.
    Uses pmap for data parallelism.
    """
    # Check available devices
    devices = jax.devices()
    num_devices = len(devices)

    # For multi-GPU: use pmap
    # This example shows the pattern (may not run on single GPU)

    def train_step(params, batch):
        x, y = batch
        def loss_fn(p):
            pred = p['w'] @ x.T
            return jnp.mean((pred - y) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    # For pmap: replicate params across devices
    params = {'w': jnp.ones((2, 3))}

    if num_devices > 1:
        # Replicate params
        replicated_params = jax.device_put_replicated(params, devices)

        # pmap the training step
        pmapped_train_step = jax.pmap(train_step)

        # Would use pmapped version for multi-GPU
        strategy = 'pmap for data parallelism'
    else:
        strategy = 'single device (no pmap needed)'

    return {
        'num_devices': num_devices,
        'devices': [str(d) for d in devices],
        'strategy': strategy,
        'note': 'Use jax.pmap for data parallelism across devices'
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Training Utilities Examples")
    print("=" * 60)

    examples = [
        ("1. Basic Training Loop", example_basic_training_loop),
        ("2. TrainState", example_train_state),
        ("3. apply_updates Workflow", example_apply_updates_workflow),
        ("4. DataLoader Integration", example_dataloader_integration),
        ("5. Train/Eval Mode", example_train_eval_mode),
        ("6. Metric Tracking", example_metric_tracking),
        ("7. Checkpointing", example_checkpointing),
        ("8. Early Stopping", example_early_stopping),
        ("9. Gradient Accumulation", example_gradient_accumulation),
        ("10. Multi-GPU Basics", example_multi_gpu_basics),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
