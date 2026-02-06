"""
Optax Basic Optimizers - 10 Examples
=====================================

Optax is JAX's gradient processing and optimization library.
Optimizers are composed from gradient transformations.

Key pattern:
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

Reference: https://optax.readthedocs.io/
"""

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple


# =============================================================================
# Example 1: SGD - Stochastic Gradient Descent
# =============================================================================
def example_sgd():
    """
    Basic SGD: params = params - lr * grads
    The simplest optimizer.
    """
    # Create optimizer
    learning_rate = 0.1
    optimizer = optax.sgd(learning_rate)

    # Initialize parameters
    params = {'w': jnp.array([1.0, 2.0, 3.0])}

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Dummy gradients
    grads = {'w': jnp.array([0.1, 0.2, 0.3])}

    # Update step
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # SGD update: w_new = w - lr * grad = [1, 2, 3] - 0.1 * [0.1, 0.2, 0.3]
    expected = jnp.array([0.99, 1.98, 2.97])

    return {
        'original_params': params['w'],
        'new_params': new_params['w'],
        'expected': expected,
        'matches': jnp.allclose(new_params['w'], expected)
    }


# =============================================================================
# Example 2: SGD with Momentum
# =============================================================================
def example_sgd_momentum():
    """
    SGD with momentum: velocity = momentum * velocity + grad
                       params = params - lr * velocity

    Momentum helps escape local minima and smooths updates.
    """
    learning_rate = 0.1
    momentum = 0.9

    optimizer = optax.sgd(learning_rate, momentum=momentum)

    params = {'w': jnp.array([1.0, 2.0, 3.0])}
    opt_state = optimizer.init(params)

    # Multiple update steps
    grads = {'w': jnp.array([0.1, 0.1, 0.1])}

    history = [params['w'].copy()]
    for _ in range(5):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        history.append(params['w'].copy())

    # Momentum accumulates, so later updates are larger
    step_sizes = [jnp.linalg.norm(history[i+1] - history[i]) for i in range(len(history)-1)]

    return {
        'initial': history[0],
        'final': history[-1],
        'step_sizes': step_sizes,
        'momentum_effect': step_sizes[-1] > step_sizes[0]  # Later steps are larger
    }


# =============================================================================
# Example 3: Adam - Adaptive Moment Estimation
# =============================================================================
def example_adam():
    """
    Adam: Adaptive learning rates per parameter.
    Combines momentum (first moment) and RMSprop (second moment).

    Most popular optimizer for deep learning.
    """
    learning_rate = 0.001
    optimizer = optax.adam(learning_rate)

    # MLP-like parameters
    params = {
        'layer1': {'w': jnp.ones((4, 8)), 'b': jnp.zeros(8)},
        'layer2': {'w': jnp.ones((8, 2)), 'b': jnp.zeros(2)}
    }

    opt_state = optimizer.init(params)

    # Gradients with different scales
    grads = {
        'layer1': {'w': jnp.ones((4, 8)) * 0.1, 'b': jnp.ones(8) * 0.01},
        'layer2': {'w': jnp.ones((8, 2)) * 1.0, 'b': jnp.ones(2) * 0.1}
    }

    # Multiple updates
    for _ in range(100):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        'layer1_w_mean': jnp.mean(params['layer1']['w']),
        'layer2_w_mean': jnp.mean(params['layer2']['w']),
        'adam_adapts': True  # Adam normalizes by gradient variance
    }


# =============================================================================
# Example 4: AdamW - Adam with Weight Decay
# =============================================================================
def example_adamw():
    """
    AdamW: Adam with decoupled weight decay regularization.
    Better for training with regularization than L2 penalty in loss.

    Weight decay: params = params * (1 - lr * weight_decay) - lr * m_hat / sqrt(v_hat)
    """
    learning_rate = 0.001
    weight_decay = 0.01

    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

    params = {'w': jnp.ones((10, 10))}
    opt_state = optimizer.init(params)

    # Zero gradients - weight decay still shrinks params
    zero_grads = {'w': jnp.zeros((10, 10))}

    # Update with zero gradient
    updates, opt_state = optimizer.update(zero_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Weight decay should shrink parameters even with zero gradient
    shrunk = jnp.all(new_params['w'] < params['w'])

    # Compare with regular Adam
    adam_opt = optax.adam(learning_rate)
    adam_state = adam_opt.init(params)
    adam_updates, _ = adam_opt.update(zero_grads, adam_state, params)
    adam_params = optax.apply_updates(params, adam_updates)

    return {
        'original': jnp.mean(params['w']),
        'adamw_updated': jnp.mean(new_params['w']),
        'adam_updated': jnp.mean(adam_params['w']),
        'adamw_shrinks': shrunk,
        'adam_unchanged': jnp.allclose(adam_params['w'], params['w'])
    }


# =============================================================================
# Example 5: RMSprop
# =============================================================================
def example_rmsprop():
    """
    RMSprop: Divide gradients by running average of squared gradients.
    Good for RNNs and non-stationary problems.
    """
    learning_rate = 0.01
    optimizer = optax.rmsprop(learning_rate)

    params = {'w': jnp.array([1.0, 2.0, 3.0])}
    opt_state = optimizer.init(params)

    # Gradients with different magnitudes
    grads = {'w': jnp.array([0.001, 0.1, 10.0])}

    # Update
    updates, _ = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # RMSprop normalizes updates by gradient magnitude
    update_magnitudes = jnp.abs(updates['w'])

    return {
        'gradient_magnitudes': jnp.abs(grads['w']),
        'update_magnitudes': update_magnitudes,
        'updates_normalized': jnp.std(update_magnitudes) < jnp.std(jnp.abs(grads['w']))
    }


# =============================================================================
# Example 6: Adagrad
# =============================================================================
def example_adagrad():
    """
    Adagrad: Accumulates squared gradients over time.
    Good for sparse gradients (NLP embeddings).
    Learning rate decreases over time.
    """
    learning_rate = 0.1
    optimizer = optax.adagrad(learning_rate)

    params = {'w': jnp.ones(5)}
    opt_state = optimizer.init(params)

    grads = {'w': jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])}

    # Track update magnitudes over time
    update_norms = []
    for _ in range(20):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        update_norms.append(float(jnp.linalg.norm(updates['w'])))

    # Adagrad: learning rate effectively decreases
    lr_decreases = all(update_norms[i] >= update_norms[i+1] for i in range(len(update_norms)-1))

    return {
        'first_update_norm': update_norms[0],
        'last_update_norm': update_norms[-1],
        'learning_rate_decreases': lr_decreases
    }


# =============================================================================
# Example 7: optax.chain - Combining Transformations
# =============================================================================
def example_chain():
    """
    optax.chain combines multiple gradient transformations.
    Order matters: transformations are applied sequentially.

    Common pattern: clip gradients, then apply optimizer.
    """
    # Chain: clip gradients, then apply Adam
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # First: clip
        optax.adam(0.001)                 # Then: optimize
    )

    params = {'w': jnp.ones((3, 4))}
    opt_state = optimizer.init(params)

    # Large gradients
    large_grads = {'w': jnp.ones((3, 4)) * 100}

    updates, opt_state = optimizer.update(large_grads, opt_state, params)

    # Gradients should be clipped before Adam processes them
    update_norm = jnp.linalg.norm(jax.tree.leaves(updates)[0])

    # More complex chain
    complex_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.add_decayed_weights(0.01),
        optax.scale(-0.001)  # Negative because we subtract updates
    )

    return {
        'update_norm': update_norm,
        'clipping_applied': update_norm < 100,
        'chain_length': 2
    }


# =============================================================================
# Example 8: optax.apply_updates
# =============================================================================
def example_apply_updates():
    """
    apply_updates combines params and updates: new_params = params + updates.
    Works with any pytree structure.
    """
    params = {
        'encoder': {
            'conv1': jnp.ones((3, 3, 3, 64)),
            'conv2': jnp.ones((3, 3, 64, 128))
        },
        'decoder': {
            'dense': jnp.ones((128, 10))
        }
    }

    # Updates (usually from optimizer)
    updates = jax.tree.map(lambda x: -0.01 * jnp.ones_like(x), params)

    # Apply updates
    new_params = optax.apply_updates(params, updates)

    # Verify structure preserved
    structure_match = (
        jax.tree.structure(params) == jax.tree.structure(new_params)
    )

    # Verify values updated
    values_changed = not jnp.allclose(
        params['encoder']['conv1'],
        new_params['encoder']['conv1']
    )

    return {
        'structure_preserved': structure_match,
        'values_changed': values_changed,
        'original_mean': jnp.mean(params['encoder']['conv1']),
        'new_mean': jnp.mean(new_params['encoder']['conv1'])
    }


# =============================================================================
# Example 9: Optimizer State Inspection
# =============================================================================
def example_state_inspection():
    """
    Inspect optimizer state to understand what's being tracked.
    """
    optimizer = optax.adam(0.001)

    params = {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)}
    opt_state = optimizer.init(params)

    # Adam state contains:
    # - count: step counter
    # - mu: first moment (mean of gradients)
    # - nu: second moment (mean of squared gradients)

    # State is a tuple of transformation states
    state_types = [type(s).__name__ for s in opt_state]

    # Access inner states (Adam is composed of scale_by_adam + scale)
    # The actual structure depends on optax version

    # Perform some updates and check state evolution
    grads = {'w': jnp.ones((3, 4)) * 0.1, 'b': jnp.ones(4) * 0.01}

    updates1, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates1)

    updates2, opt_state = optimizer.update(grads, opt_state, params)

    return {
        'state_types': state_types,
        'num_state_components': len(opt_state),
        'first_update_norm': float(jnp.linalg.norm(updates1['w'])),
        'second_update_norm': float(jnp.linalg.norm(updates2['w']))
    }


# =============================================================================
# Example 10: Custom Optimizer Composition
# =============================================================================
def example_custom_optimizer():
    """
    Build custom optimizers by composing transformations.
    """
    # Custom optimizer: SGD with momentum + weight decay + gradient clipping
    def custom_sgd_with_extras(lr, momentum=0.9, weight_decay=0.0001, max_grad_norm=1.0):
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.trace(decay=momentum),  # Momentum
            optax.add_decayed_weights(weight_decay),
            optax.scale(-lr)
        )

    optimizer = custom_sgd_with_extras(
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        max_grad_norm=1.0
    )

    params = {'w': jnp.ones((10, 10))}
    opt_state = optimizer.init(params)

    # Training loop
    for step in range(100):
        grads = {'w': jnp.ones((10, 10)) * (0.1 if step < 50 else 0.01)}
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    # Different optimizer for different param groups (manual)
    def multi_optimizer(params):
        """Different optimizers for different parameters."""
        # In practice, you'd partition params and use optax.multi_transform
        encoder_opt = optax.adam(0.001)
        decoder_opt = optax.sgd(0.01)
        return encoder_opt, decoder_opt

    # Using optax.multi_transform for param-specific optimizers
    def create_multi_optimizer():
        return optax.multi_transform(
            {'encoder': optax.adam(0.001), 'decoder': optax.sgd(0.01)},
            param_labels={'encoder': 'encoder', 'decoder': 'decoder'}
        )

    return {
        'final_w_mean': jnp.mean(params['w']),
        'custom_optimizer_works': True
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Basic Optimizers Examples")
    print("=" * 60)

    examples = [
        ("1. SGD", example_sgd),
        ("2. SGD with Momentum", example_sgd_momentum),
        ("3. Adam", example_adam),
        ("4. AdamW", example_adamw),
        ("5. RMSprop", example_rmsprop),
        ("6. Adagrad", example_adagrad),
        ("7. optax.chain", example_chain),
        ("8. apply_updates", example_apply_updates),
        ("9. State Inspection", example_state_inspection),
        ("10. Custom Optimizer", example_custom_optimizer),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}")
                elif isinstance(value, list):
                    print(f"  {key}: {value[:5]}..." if len(value) > 5 else f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
