"""
Optax Gradient Clipping - 10 Exercises
======================================

Gradient clipping prevents exploding gradients.
Essential for RNNs, LSTMs, and Transformers.

Key methods:
- clip_by_global_norm: Clip if global norm exceeds threshold
- clip_by_value: Clip element-wise to range
- clip: Simple element-wise clipping

Reference: https://optax.readthedocs.io/en/latest/api/transformations.html
"""

import jax
import jax.numpy as jnp
import optax


# =============================================================================
# Exercise 1: clip_by_global_norm
# =============================================================================
def exercise_clip_by_global_norm():
    """
    Clip gradients by their global norm.
    If ||g|| > max_norm, scale g to have norm = max_norm.

    Most common clipping method for neural networks.
    """
    max_norm = 1.0

    # TODO: Implement this function
    # 1. Create gradient transformation using optax.clip_by_global_norm
    # 2. Initialize state with dummy params
    # 3. Test with small gradients (should be unchanged)
    # 4. Test with large gradients (should be clipped to max_norm)

    clip_transform = None
    params = None
    state = None

    small_grads = None
    small_norm = None
    clipped_small = None
    clipped_small_norm = None

    large_grads = None
    large_norm = None
    clipped_large = None
    clipped_large_norm = None

    return {
        'small_norm_before': None,
        'small_norm_after': None,
        'large_norm_before': None,
        'large_norm_after': None,
        'large_clipped_to_max': None
    }


# =============================================================================
# Exercise 2: clip_by_value
# =============================================================================
def exercise_clip_by_value():
    """
    Clip gradients element-wise to a range [min_val, max_val].
    Useful when you want hard limits on gradient values.
    """
    min_val, max_val = -1.0, 1.0

    # TODO: Implement this function
    # 1. Create clip_by_value transformation with min_val and max_val
    # 2. Initialize state
    # 3. Apply to gradients with values outside the range
    # 4. Verify the clipped values match expected

    clip_transform = None
    params = None
    state = None
    grads = None
    clipped = None
    expected = None

    return {
        'original': None,
        'clipped': None,
        'expected': None,
        'matches': None
    }


# =============================================================================
# Exercise 3: clip (Simple Clipping)
# =============================================================================
def exercise_simple_clip():
    """
    Simple clipping: same as clip_by_value with symmetric range.
    clip(max_val) is equivalent to clip_by_value(-max_val, max_val).
    """
    max_val = 1.0

    # TODO: Implement this function
    # 1. Create optax.clip transformation with max_val
    # 2. Initialize state and apply to gradients
    # 3. Verify all clipped values are within [-max_val, max_val]

    clip_transform = None
    params = None
    state = None
    grads = None
    clipped = None

    return {
        'original': None,
        'clipped': None,
        'in_range': None
    }


# =============================================================================
# Exercise 4: Combining Clipping with Optimizers
# =============================================================================
def exercise_clipping_with_optimizer():
    """
    Chain clipping with an optimizer using optax.chain.
    Clipping should come BEFORE the optimizer.
    """
    # TODO: Implement this function
    # 1. Create optimizer with optax.chain: clip_by_global_norm(1.0) + adam(0.001)
    # 2. Compare updates from clipped vs unclipped optimizer
    # 3. Verify that clipping reduces the update norm

    optimizer = None
    params = None
    opt_state = None
    large_grads = None
    updates = None

    adam_only = None
    adam_state = None
    adam_updates = None

    return {
        'clipped_update_norm': None,
        'unclipped_update_norm': None,
        'clipping_reduces_norm': None
    }


# =============================================================================
# Exercise 5: Adaptive Gradient Clipping
# =============================================================================
def exercise_adaptive_clipping():
    """
    Adaptive gradient clipping (AGC) from NFNet paper.
    Clips based on ratio of gradient norm to parameter norm.
    """
    # TODO: Implement this function
    # 1. Implement adaptive_clip function that clips if ||g|| > max_ratio * ||w||
    # 2. Test with large params + small grads (no clipping)
    # 3. Test with small params + large grads (clipping occurs)

    def adaptive_clip(max_ratio=0.01, eps=1e-3):
        """
        Clip if ||g|| > max_ratio * ||w||.
        """
        def clip_fn(grad, param):
            # TODO: Implement adaptive clipping logic
            return None

        return clip_fn

    clip_fn = adaptive_clip(max_ratio=0.01)

    large_param = None
    small_grad = None
    clipped1 = None

    small_param = None
    large_grad = None
    clipped2 = None

    return {
        'small_grad_unchanged': None,
        'large_grad_clipped': None,
        'adaptive_clipping': True
    }


# =============================================================================
# Exercise 6: Per-Layer Gradient Clipping
# =============================================================================
def exercise_per_layer_clipping():
    """
    Apply different clipping thresholds to different layers.
    """
    # TODO: Implement this function
    # 1. Create per_layer_clip function that applies different max_norms per layer
    # 2. Test with different limits for embedding, encoder, decoder, output
    # 3. Verify each layer respects its limit

    def per_layer_clip(layer_max_norms):
        """Create per-layer clipping transformation."""
        def clip_fn(updates, state, params=None):
            # TODO: Implement per-layer clipping
            return None, state

        def init_fn(params):
            return optax.EmptyState()

        return optax.GradientTransformation(init_fn, clip_fn)

    layer_limits = {
        'embedding': 0.5,
        'encoder': 1.0,
        'decoder': 1.0,
        'output': 2.0
    }

    clip_transform = None
    params = None
    state = None
    grads = None
    clipped = None
    norms = None

    return {
        'embedding_norm': None,
        'encoder_norm': None,
        'output_norm': None,
        'embedding_within_limit': None
    }


# =============================================================================
# Exercise 7: Gradient Norm Monitoring
# =============================================================================
def exercise_gradient_monitoring():
    """
    Monitor gradient norms during training to decide on clipping threshold.
    """
    # TODO: Implement this function
    # 1. Implement monitor_gradient_norms to analyze gradient history
    # 2. Compute mean, max, min, std of norms
    # 3. Suggest clip threshold as mean + 2*std

    def monitor_gradient_norms(grads_history):
        """Analyze gradient norms from training history."""
        # TODO: Implement gradient norm monitoring
        return {
            'mean_norm': None,
            'max_norm': None,
            'min_norm': None,
            'std_norm': None,
            'suggested_clip': None
        }

    key = jax.random.key(42)
    grads_history = []

    # TODO: Simulate gradient history with occasional spikes
    # for i in range(100):
    #     ...

    stats = monitor_gradient_norms(grads_history)

    return stats


# =============================================================================
# Exercise 8: When to Use Which Clipping Strategy
# =============================================================================
def exercise_clipping_strategies():
    """
    Guidelines for choosing clipping strategy.
    """
    # TODO: Implement this function
    # 1. Create original_grad with known norm
    # 2. Apply global norm clipping (preserves direction)
    # 3. Apply value clipping (may change direction)
    # 4. Verify which method preserves gradient direction

    strategies = {
        'global_norm': {
            'use_for': ['RNNs', 'LSTMs', 'Transformers'],
            'typical_values': [1.0, 5.0, 10.0],
            'pros': 'Preserves gradient direction',
            'cons': 'Requires tuning threshold'
        },
        'value_clip': {
            'use_for': ['When you need hard limits'],
            'typical_values': [1.0],
            'pros': 'Simple, no surprises',
            'cons': 'Changes gradient direction'
        },
        'adaptive': {
            'use_for': ['Very deep networks', 'NFNet-style training'],
            'typical_values': [0.01],
            'pros': 'Automatically scales with params',
            'cons': 'More complex'
        }
    }

    original_grad = None
    norm = None
    global_clipped = None
    value_clipped = None

    original_direction = None
    global_direction = None
    value_direction = None

    return {
        'strategies': list(strategies.keys()),
        'original_direction': None,
        'global_preserves_direction': None,
        'value_preserves_direction': None
    }


# =============================================================================
# Exercise 9: Clipping for RNNs/LSTMs
# =============================================================================
def exercise_rnn_clipping():
    """
    RNNs are prone to exploding gradients.
    Gradient clipping is essential.
    """
    # TODO: Implement this function
    # 1. Create RNN optimizer with clip_by_global_norm(5.0) + adam
    # 2. Simulate RNN gradients that grow with sequence length
    # 3. Apply clipping and verify norm is reduced

    rnn_optimizer = None

    key = jax.random.key(42)

    def simulate_rnn_gradients(key, seq_length=50):
        """Longer sequences can have exponentially growing gradients."""
        # TODO: Implement gradient simulation
        return {}

    grads = None
    total_norm = None

    clip_transform = None
    state = None
    clipped_grads = None
    clipped_norm = None

    return {
        'original_norm': None,
        'clipped_norm': None,
        'was_clipped': None,
        'rnn_clip_value': 5.0
    }


# =============================================================================
# Exercise 10: Clipping for Transformers
# =============================================================================
def exercise_transformer_clipping():
    """
    Transformers also benefit from gradient clipping,
    especially during warmup and for attention layers.
    """
    # TODO: Implement this function
    # 1. Create warmup_cosine_decay_schedule
    # 2. Create optimizer: clip_by_global_norm(1.0) + adamw with schedule
    # 3. Initialize with transformer-like params (embedding, attention, ffn)
    # 4. Compute gradient norms

    schedule = None
    transformer_optimizer = None

    params = None
    opt_state = None

    key = jax.random.key(42)
    grads = None
    original_norm = None
    updates = None

    return {
        'original_grad_norm': None,
        'transformer_clip_value': 1.0,
        'warmup_steps': 4000,
        'architecture': 'Transformer uses clip + warmup + weight_decay'
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Gradient Clipping Exercises")
    print("=" * 60)

    exercises = [
        ("1. clip_by_global_norm", exercise_clip_by_global_norm),
        ("2. clip_by_value", exercise_clip_by_value),
        ("3. Simple clip", exercise_simple_clip),
        ("4. With Optimizer", exercise_clipping_with_optimizer),
        ("5. Adaptive Clipping", exercise_adaptive_clipping),
        ("6. Per-Layer Clipping", exercise_per_layer_clipping),
        ("7. Gradient Monitoring", exercise_gradient_monitoring),
        ("8. Clipping Strategies", exercise_clipping_strategies),
        ("9. RNN Clipping", exercise_rnn_clipping),
        ("10. Transformer Clipping", exercise_transformer_clipping),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
