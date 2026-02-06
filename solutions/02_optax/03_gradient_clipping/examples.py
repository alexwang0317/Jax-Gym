"""
Optax Gradient Clipping - 10 Examples
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
# Example 1: clip_by_global_norm
# =============================================================================
def example_clip_by_global_norm():
    """
    Clip gradients by their global norm.
    If ||g|| > max_norm, scale g to have norm = max_norm.

    Most common clipping method for neural networks.
    """
    max_norm = 1.0

    # Create gradient transformation
    clip_transform = optax.clip_by_global_norm(max_norm)

    # Dummy params and state
    params = {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)}
    state = clip_transform.init(params)

    # Normal gradients (small norm)
    small_grads = {'w': jnp.ones((3, 4)) * 0.1, 'b': jnp.ones(4) * 0.1}
    small_norm = optax.global_norm(small_grads)

    clipped_small, _ = clip_transform.update(small_grads, state, params)
    clipped_small_norm = optax.global_norm(clipped_small)

    # Large gradients (should be clipped)
    large_grads = {'w': jnp.ones((3, 4)) * 10, 'b': jnp.ones(4) * 10}
    large_norm = optax.global_norm(large_grads)

    clipped_large, _ = clip_transform.update(large_grads, state, params)
    clipped_large_norm = optax.global_norm(clipped_large)

    return {
        'small_norm_before': float(small_norm),
        'small_norm_after': float(clipped_small_norm),
        'large_norm_before': float(large_norm),
        'large_norm_after': float(clipped_large_norm),
        'large_clipped_to_max': jnp.allclose(clipped_large_norm, max_norm, atol=1e-5)
    }


# =============================================================================
# Example 2: clip_by_value
# =============================================================================
def example_clip_by_value():
    """
    Clip gradients element-wise to a range [min_val, max_val].
    Useful when you want hard limits on gradient values.
    """
    min_val, max_val = -1.0, 1.0

    clip_transform = optax.clip_by_value(min_val, max_val)

    params = {'w': jnp.ones(5)}
    state = clip_transform.init(params)

    # Gradients with values outside range
    grads = {'w': jnp.array([-5.0, -0.5, 0.0, 0.5, 5.0])}

    clipped, _ = clip_transform.update(grads, state, params)

    expected = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    return {
        'original': grads['w'],
        'clipped': clipped['w'],
        'expected': expected,
        'matches': jnp.allclose(clipped['w'], expected)
    }


# =============================================================================
# Example 3: clip (Simple Clipping)
# =============================================================================
def example_simple_clip():
    """
    Simple clipping: same as clip_by_value with symmetric range.
    clip(max_val) is equivalent to clip_by_value(-max_val, max_val).
    """
    max_val = 1.0
    clip_transform = optax.clip(max_val)

    params = {'w': jnp.array([-3.0, -0.5, 0.0, 0.5, 3.0])}
    state = clip_transform.init(params)
    grads = params  # Use params as grads for this example

    clipped, _ = clip_transform.update(grads, state, params)

    return {
        'original': grads['w'],
        'clipped': clipped['w'],
        'in_range': jnp.all(jnp.abs(clipped['w']) <= max_val)
    }


# =============================================================================
# Example 4: Combining Clipping with Optimizers
# =============================================================================
def example_clipping_with_optimizer():
    """
    Chain clipping with an optimizer using optax.chain.
    Clipping should come BEFORE the optimizer.
    """
    # Clip gradients, then apply Adam
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(0.001)
    )

    params = {'w': jnp.ones((10, 10))}
    opt_state = optimizer.init(params)

    # Large gradients
    large_grads = {'w': jnp.ones((10, 10)) * 100}

    # The update should have clipped gradients
    updates, _ = optimizer.update(large_grads, opt_state, params)

    # Compare with unclipped Adam
    adam_only = optax.adam(0.001)
    adam_state = adam_only.init(params)
    adam_updates, _ = adam_only.update(large_grads, adam_state, params)

    return {
        'clipped_update_norm': float(jnp.linalg.norm(updates['w'])),
        'unclipped_update_norm': float(jnp.linalg.norm(adam_updates['w'])),
        'clipping_reduces_norm': jnp.linalg.norm(updates['w']) < jnp.linalg.norm(adam_updates['w'])
    }


# =============================================================================
# Example 5: Adaptive Gradient Clipping
# =============================================================================
def example_adaptive_clipping():
    """
    Adaptive gradient clipping (AGC) from NFNet paper.
    Clips based on ratio of gradient norm to parameter norm.
    """
    def adaptive_clip(max_ratio=0.01, eps=1e-3):
        """
        Clip if ||g|| > max_ratio * ||w||.
        """
        def clip_fn(grad, param):
            param_norm = jnp.linalg.norm(param)
            grad_norm = jnp.linalg.norm(grad)

            # Clip if grad_norm > max_ratio * param_norm
            max_norm = jnp.maximum(param_norm * max_ratio, eps)
            scale = jnp.minimum(1.0, max_norm / (grad_norm + eps))

            return grad * scale

        return clip_fn

    clip_fn = adaptive_clip(max_ratio=0.01)

    # Large params, small grads - no clipping
    large_param = jnp.ones((10, 10)) * 100
    small_grad = jnp.ones((10, 10)) * 0.1
    clipped1 = clip_fn(small_grad, large_param)

    # Small params, large grads - clipping
    small_param = jnp.ones((10, 10)) * 0.1
    large_grad = jnp.ones((10, 10)) * 10
    clipped2 = clip_fn(large_grad, small_param)

    return {
        'small_grad_unchanged': jnp.allclose(clipped1, small_grad),
        'large_grad_clipped': jnp.linalg.norm(clipped2) < jnp.linalg.norm(large_grad),
        'adaptive_clipping': True
    }


# =============================================================================
# Example 6: Per-Layer Gradient Clipping
# =============================================================================
def example_per_layer_clipping():
    """
    Apply different clipping thresholds to different layers.
    """
    def per_layer_clip(layer_max_norms):
        """Create per-layer clipping transformation."""
        def clip_fn(updates, state, params=None):
            clipped = {}
            for key, grad in updates.items():
                max_norm = layer_max_norms.get(key, float('inf'))
                grad_norm = jnp.linalg.norm(grad)
                scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))
                clipped[key] = grad * scale
            return clipped, state

        def init_fn(params):
            return optax.EmptyState()

        return optax.GradientTransformation(init_fn, clip_fn)

    # Different limits for different layers
    layer_limits = {
        'embedding': 0.5,   # Embeddings need smaller gradients
        'encoder': 1.0,
        'decoder': 1.0,
        'output': 2.0       # Output layer can have larger gradients
    }

    clip_transform = per_layer_clip(layer_limits)

    params = {
        'embedding': jnp.ones((1000, 128)),
        'encoder': jnp.ones((128, 256)),
        'decoder': jnp.ones((256, 128)),
        'output': jnp.ones((128, 10))
    }
    state = clip_transform.init(params)

    # All large gradients
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 10, params)

    clipped, _ = clip_transform.update(grads, state, params)

    # Check each layer's norm
    norms = {k: float(jnp.linalg.norm(v)) for k, v in clipped.items()}

    return {
        'embedding_norm': norms['embedding'],
        'encoder_norm': norms['encoder'],
        'output_norm': norms['output'],
        'embedding_within_limit': norms['embedding'] <= 0.5 + 1e-5
    }


# =============================================================================
# Example 7: Gradient Norm Monitoring
# =============================================================================
def example_gradient_monitoring():
    """
    Monitor gradient norms during training to decide on clipping threshold.
    """
    def monitor_gradient_norms(grads_history):
        """Analyze gradient norms from training history."""
        norms = [optax.global_norm(g) for g in grads_history]

        return {
            'mean_norm': float(jnp.mean(jnp.array(norms))),
            'max_norm': float(jnp.max(jnp.array(norms))),
            'min_norm': float(jnp.min(jnp.array(norms))),
            'std_norm': float(jnp.std(jnp.array(norms))),
            'suggested_clip': float(jnp.mean(jnp.array(norms)) + 2 * jnp.std(jnp.array(norms)))
        }

    # Simulate gradient history
    key = jax.random.key(42)
    grads_history = []
    for i in range(100):
        key, subkey = jax.random.split(key)
        # Simulate varying gradient magnitudes (some spikes)
        scale = 1.0 if i % 20 != 0 else 10.0  # Occasional spikes
        grads = {'w': jax.random.normal(subkey, (100, 100)) * scale}
        grads_history.append(grads)

    stats = monitor_gradient_norms(grads_history)

    return stats


# =============================================================================
# Example 8: When to Use Which Clipping Strategy
# =============================================================================
def example_clipping_strategies():
    """
    Guidelines for choosing clipping strategy.
    """
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

    # Demonstrate effect of clipping on direction
    original_grad = jnp.array([3.0, 4.0])  # norm = 5
    norm = jnp.linalg.norm(original_grad)

    # Global norm: preserves direction
    global_clipped = original_grad * (1.0 / norm)  # norm = 1

    # Value clip: may change direction
    value_clipped = jnp.clip(original_grad, -1.0, 1.0)

    # Check if directions preserved
    original_direction = original_grad / norm
    global_direction = global_clipped / jnp.linalg.norm(global_clipped)
    value_direction = value_clipped / jnp.linalg.norm(value_clipped)

    return {
        'strategies': list(strategies.keys()),
        'original_direction': original_direction,
        'global_preserves_direction': jnp.allclose(original_direction, global_direction),
        'value_preserves_direction': jnp.allclose(original_direction, value_direction)
    }


# =============================================================================
# Example 9: Clipping for RNNs/LSTMs
# =============================================================================
def example_rnn_clipping():
    """
    RNNs are prone to exploding gradients.
    Gradient clipping is essential.
    """
    # Typical RNN training setup
    rnn_optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),  # RNNs often use 5.0
        optax.adam(0.001)
    )

    # Simulate RNN gradients (can have high variance)
    key = jax.random.key(42)

    # Simulate gradients through time (BPTT)
    def simulate_rnn_gradients(key, seq_length=50):
        """Longer sequences can have exponentially growing gradients."""
        # Simplified: gradients can grow exponentially
        grads = {}
        for t in range(seq_length):
            key, subkey = jax.random.split(key)
            # Gradients may grow with sequence position
            scale = jnp.exp(0.1 * t) if t < 30 else jnp.exp(3.0)  # Clip effect
            grads[f'time_{t}'] = jax.random.normal(subkey, (64, 64)) * scale * 0.01
        return grads

    grads = simulate_rnn_gradients(key)
    total_norm = optax.global_norm(grads)

    # With clipping
    clip_transform = optax.clip_by_global_norm(5.0)
    state = clip_transform.init(grads)
    clipped_grads, _ = clip_transform.update(grads, state)
    clipped_norm = optax.global_norm(clipped_grads)

    return {
        'original_norm': float(total_norm),
        'clipped_norm': float(clipped_norm),
        'was_clipped': total_norm > 5.0,
        'rnn_clip_value': 5.0
    }


# =============================================================================
# Example 10: Clipping for Transformers
# =============================================================================
def example_transformer_clipping():
    """
    Transformers also benefit from gradient clipping,
    especially during warmup and for attention layers.
    """
    # Typical Transformer training setup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.0001,
        warmup_steps=4000,
        decay_steps=100000
    )

    transformer_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Transformers often use 1.0
        optax.adamw(schedule, weight_decay=0.01)
    )

    # Simulate transformer gradients
    params = {
        'embedding': jnp.ones((32000, 512)),  # Vocab x dim
        'attention': {
            'q': jnp.ones((512, 512)),
            'k': jnp.ones((512, 512)),
            'v': jnp.ones((512, 512)),
            'o': jnp.ones((512, 512)),
        },
        'ffn': {
            'w1': jnp.ones((512, 2048)),
            'w2': jnp.ones((2048, 512)),
        }
    }

    opt_state = transformer_optimizer.init(params)

    # Simulate gradients (attention can have high variance)
    key = jax.random.key(42)
    grads = jax.tree.map(
        lambda x: jax.random.normal(jax.random.fold_in(key, id(x)), x.shape) * 0.1,
        params
    )

    # Check gradient norms before/after
    original_norm = optax.global_norm(grads)

    updates, _ = transformer_optimizer.update(grads, opt_state, params)

    return {
        'original_grad_norm': float(original_norm),
        'transformer_clip_value': 1.0,
        'warmup_steps': 4000,
        'architecture': 'Transformer uses clip + warmup + weight_decay'
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Gradient Clipping Examples")
    print("=" * 60)

    examples = [
        ("1. clip_by_global_norm", example_clip_by_global_norm),
        ("2. clip_by_value", example_clip_by_value),
        ("3. Simple clip", example_simple_clip),
        ("4. With Optimizer", example_clipping_with_optimizer),
        ("5. Adaptive Clipping", example_adaptive_clipping),
        ("6. Per-Layer Clipping", example_per_layer_clipping),
        ("7. Gradient Monitoring", example_gradient_monitoring),
        ("8. Clipping Strategies", example_clipping_strategies),
        ("9. RNN Clipping", example_rnn_clipping),
        ("10. Transformer Clipping", example_transformer_clipping),
    ]

    for name, func in examples:
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
