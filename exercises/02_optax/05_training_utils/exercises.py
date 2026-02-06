"""
Optax Training Utilities - 10 Exercises
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
# Exercise 1: Basic Training Loop Pattern
# =============================================================================
def exercise_basic_training_loop():
    """
    The fundamental training loop pattern in JAX.

    Instructions:
    1. Create a simple model function that computes params['w'] @ x + params['b']
    2. Create a loss function that computes MSE between predictions and targets
    3. Initialize params with w as ones((2, 3)) and b as zeros(2)
    4. Create an Adam optimizer with learning rate 0.01
    5. Implement a JIT-compiled train_step that:
       - Computes loss and gradients using jax.value_and_grad
       - Updates optimizer state
       - Applies updates to params
    6. Run 100 training steps with dummy data (x=ones(3), y=[1.0, 2.0])
    7. Track losses and verify the loss decreased

    Returns:
        dict with keys:
        - 'initial_loss': First loss value
        - 'final_loss': Last loss value
        - 'loss_decreased': Boolean, True if final < initial
    """
    # TODO: Implement this function

    initial_loss = None
    final_loss = None
    loss_decreased = None

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_decreased': loss_decreased
    }


# =============================================================================
# Exercise 2: TrainState from Flax
# =============================================================================
def exercise_train_state():
    """
    Flax's TrainState bundles params, optimizer state, and step count.
    Much cleaner than managing them separately.

    Instructions:
    1. Create params dict with w as ones((3, 4)) and b as zeros(4)
    2. Create a warmup_cosine_decay_schedule with:
       - init_value=0.0, peak_value=0.001
       - warmup_steps=100, decay_steps=1000
    3. Create an Adam optimizer with that schedule
    4. Create a TrainState with apply_fn that computes p['w'] @ x + p['b']
    5. Record the initial step count
    6. Create gradients (w: ones((3, 4)) * 0.1, b: ones(4) * 0.01)
    7. Apply gradients to get new state
    8. Verify step incremented and params changed

    Returns:
        dict with keys:
        - 'initial_step': Step count before update
        - 'after_update_step': Step count after update
        - 'params_updated': Boolean, True if params changed
    """
    # TODO: Implement this function

    initial_step = None
    after_update_step = None
    params_updated = None

    return {
        'initial_step': initial_step,
        'after_update_step': after_update_step,
        'params_updated': params_updated
    }


# =============================================================================
# Exercise 3: apply_updates Workflow
# =============================================================================
def exercise_apply_updates_workflow():
    """
    Detailed workflow of computing and applying updates.

    Instructions:
    1. Create params with layer1: ones((10, 20)) and layer2: ones((20, 5))
    2. Create a chained optimizer with:
       - clip_by_global_norm(1.0)
       - adamw(0.001, weight_decay=0.01)
    3. Initialize optimizer state
    4. Create gradients (both layers: ones * 0.5)
    5. Get updates from optimizer.update()
    6. Apply updates using optax.apply_updates()
    7. Verify structure is preserved and params changed

    Returns:
        dict with keys:
        - 'updates_layer1_norm': L2 norm of updates for layer1
        - 'grads_layer1_norm': L2 norm of gradients for layer1
        - 'structure_preserved': Boolean, True if keys and shapes match
        - 'params_changed': Boolean, True if params differ
    """
    # TODO: Implement this function

    updates_layer1_norm = None
    grads_layer1_norm = None
    structure_preserved = None
    params_changed = None

    return {
        'updates_layer1_norm': updates_layer1_norm,
        'grads_layer1_norm': grads_layer1_norm,
        'structure_preserved': structure_preserved,
        'params_changed': params_changed
    }


# =============================================================================
# Exercise 4: PyTorch DataLoader Integration
# =============================================================================
def exercise_dataloader_integration():
    """
    Using PyTorch DataLoader with JAX.
    Common pattern for dataset handling.

    Instructions:
    1. Implement a SimpleDataLoader class with:
       - __init__(self, X, y, batch_size, shuffle=True)
       - __iter__ that yields batches (shuffled if shuffle=True)
       - __len__ that returns number of batches
    2. Create random data: X (1000, 10) float32, y (1000,) integers 0-4
    3. Create loader with batch_size=32
    4. Iterate through batches, converting to JAX arrays
    5. Count batches seen (stop after 5)

    Returns:
        dict with keys:
        - 'total_samples': Total number of samples
        - 'batch_size': Batch size used
        - 'num_batches': Total batches in loader
        - 'batches_seen': Number of batches iterated
    """
    # TODO: Implement this function

    total_samples = None
    batch_size = None
    num_batches = None
    batches_seen = None

    return {
        'total_samples': total_samples,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'batches_seen': batches_seen
    }


# =============================================================================
# Exercise 5: Train/Eval Mode Handling
# =============================================================================
def exercise_train_eval_mode():
    """
    Handling training vs evaluation mode.
    Affects Dropout, BatchNorm, etc.

    Instructions:
    1. Create a model function with dropout that takes:
       - params, x, training (bool), rng (optional)
    2. The model should:
       - Compute h = x @ w1 + b1, then relu
       - If training, apply dropout (keep_prob=0.9)
       - Compute output = h @ w2 + b2
    3. Initialize params (w1: (10,20)*0.1, b1: zeros(20),
       w2: (20,5)*0.1, b2: zeros(5))
    4. Run in training mode with rng
    5. Run in eval mode twice (should be identical)

    Returns:
        dict with keys:
        - 'train_output_shape': Shape of training output
        - 'eval_output_shape': Shape of eval output
        - 'eval_deterministic': Boolean, True if eval outputs match
    """
    # TODO: Implement this function

    train_output_shape = None
    eval_output_shape = None
    eval_deterministic = None

    return {
        'train_output_shape': train_output_shape,
        'eval_output_shape': eval_output_shape,
        'eval_deterministic': eval_deterministic
    }


# =============================================================================
# Exercise 6: Metric Tracking and Logging
# =============================================================================
def exercise_metric_tracking():
    """
    Track and log training metrics.

    Instructions:
    1. Create a MetricTracker class with:
       - update(name, value): Add value to metric history
       - get_mean(name, window=None): Get mean of last 'window' values
       - get_last(name): Get most recent value
    2. Simulate 100 training steps with:
       - loss = 1.0 / (step + 1) + random noise * 0.1
       - accuracy = min(0.95, 0.5 + step * 0.005 + noise * 0.02)
    3. Track both metrics each step

    Returns:
        dict with keys:
        - 'final_loss': Last recorded loss
        - 'final_accuracy': Last recorded accuracy
        - 'avg_loss_last_20': Mean loss over last 20 steps
        - 'total_steps': Total number of recorded steps
    """
    # TODO: Implement this function

    final_loss = None
    final_accuracy = None
    avg_loss_last_20 = None
    total_steps = None

    return {
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'avg_loss_last_20': avg_loss_last_20,
        'total_steps': total_steps
    }


# =============================================================================
# Exercise 7: Checkpointing (Simplified)
# =============================================================================
def exercise_checkpointing():
    """
    Save and restore training state.
    In practice, use orbax-checkpoint for robust checkpointing.

    Instructions:
    1. Implement save_checkpoint(path, state) that returns dict with:
       - 'step': Current step as int
       - 'params': Params converted to Python lists
    2. Implement load_checkpoint(checkpoint, tx) that:
       - Converts params back to JAX arrays
       - Creates new TrainState
       - Restores the step count
    3. Create TrainState with w: ones((3, 4)), Adam optimizer
    4. Simulate 10 training steps
    5. Save checkpoint
    6. Restore from checkpoint
    7. Verify params and step match

    Returns:
        dict with keys:
        - 'saved_step': Step count in checkpoint
        - 'restored_step': Step count after restore
        - 'params_match': Boolean, True if params match
    """
    # TODO: Implement this function

    saved_step = None
    restored_step = None
    params_match = None

    return {
        'saved_step': saved_step,
        'restored_step': restored_step,
        'params_match': params_match
    }


# =============================================================================
# Exercise 8: Early Stopping
# =============================================================================
def exercise_early_stopping():
    """
    Stop training when validation loss stops improving.

    Instructions:
    1. Create an EarlyStopping class with:
       - __init__(patience, min_delta=0.0)
       - __call__(val_loss) that returns True if should stop
       - Tracks best_loss, counter, should_stop
    2. Create instance with patience=3
    3. Test with losses: [1.0, 0.8, 0.7, 0.65, 0.64, 0.64, 0.65, 0.66, 0.67, 0.68]
    4. Find which epoch triggers early stopping

    Returns:
        dict with keys:
        - 'val_losses': The list of validation losses
        - 'stopped_at_epoch': Epoch index where stopped (or None)
        - 'best_loss': Best loss seen
        - 'early_stopping_triggered': Boolean, True if stopped early
    """
    # TODO: Implement this function

    val_losses = None
    stopped_at_epoch = None
    best_loss = None
    early_stopping_triggered = None

    return {
        'val_losses': val_losses,
        'stopped_at_epoch': stopped_at_epoch,
        'best_loss': best_loss,
        'early_stopping_triggered': early_stopping_triggered
    }


# =============================================================================
# Exercise 9: Gradient Accumulation
# =============================================================================
def exercise_gradient_accumulation():
    """
    Accumulate gradients over multiple steps for effective larger batch.
    Useful when memory limited.

    Instructions:
    1. Create a function that wraps an optimizer with optax.MultiSteps
    2. Set accumulation_steps=4, actual_batch_size=8
    3. Calculate effective_batch_size
    4. Create params with w: ones((10, 10))
    5. Run accumulation_steps + 1 iterations
    6. Track when params actually change from original

    Returns:
        dict with keys:
        - 'accumulation_steps': Number of accumulation steps
        - 'effective_batch_size': actual_batch * accumulation_steps
        - 'steps_before_update': Steps before params first changed
        - 'params_changed': Boolean, True if params changed
    """
    # TODO: Implement this function

    accumulation_steps = None
    effective_batch_size = None
    steps_before_update = None
    params_changed = None

    return {
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': effective_batch_size,
        'steps_before_update': steps_before_update,
        'params_changed': params_changed
    }


# =============================================================================
# Exercise 10: Multi-GPU Training Basics (Conceptual)
# =============================================================================
def exercise_multi_gpu_basics():
    """
    Concepts for multi-GPU training with JAX.
    Uses pmap for data parallelism.

    Instructions:
    1. Get available devices using jax.devices()
    2. Count number of devices
    3. Create a simple train_step function
    4. Create params with w: ones((2, 3))
    5. If multiple devices:
       - Use jax.device_put_replicated to replicate params
       - Strategy is 'pmap for data parallelism'
    6. Otherwise: strategy is 'single device (no pmap needed)'

    Returns:
        dict with keys:
        - 'num_devices': Number of available devices
        - 'devices': List of device strings
        - 'strategy': Strategy description string
        - 'note': 'Use jax.pmap for data parallelism across devices'
    """
    # TODO: Implement this function

    num_devices = None
    devices = None
    strategy = None
    note = None

    return {
        'num_devices': num_devices,
        'devices': devices,
        'strategy': strategy,
        'note': note
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Training Utilities Exercises")
    print("=" * 60)

    exercises = [
        ("1. Basic Training Loop", exercise_basic_training_loop),
        ("2. TrainState", exercise_train_state),
        ("3. apply_updates Workflow", exercise_apply_updates_workflow),
        ("4. DataLoader Integration", exercise_dataloader_integration),
        ("5. Train/Eval Mode", exercise_train_eval_mode),
        ("6. Metric Tracking", exercise_metric_tracking),
        ("7. Checkpointing", exercise_checkpointing),
        ("8. Early Stopping", exercise_early_stopping),
        ("9. Gradient Accumulation", exercise_gradient_accumulation),
        ("10. Multi-GPU Basics", exercise_multi_gpu_basics),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
