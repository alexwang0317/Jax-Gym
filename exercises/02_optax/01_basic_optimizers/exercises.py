"""
Optax Basic Optimizers - 10 Exercises
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
# Exercise 1: SGD - Stochastic Gradient Descent
# =============================================================================
def exercise_sgd():
    """
    Basic SGD: params = params - lr * grads
    The simplest optimizer.

    TODO:
    - Create an SGD optimizer with learning_rate = 0.1
    - Initialize parameters: {'w': jnp.array([1.0, 2.0, 3.0])}
    - Initialize optimizer state
    - Create gradients: {'w': jnp.array([0.1, 0.2, 0.3])}
    - Perform one update step
    - Calculate expected result: w_new = w - lr * grad
    """
    # TODO: Implement this function
    learning_rate = None
    optimizer = None

    params = None
    opt_state = None
    grads = None

    updates = None
    new_opt_state = None
    new_params = None

    # SGD update: w_new = w - lr * grad = [1, 2, 3] - 0.1 * [0.1, 0.2, 0.3]
    expected = None

    return {
        'original_params': params['w'] if params else None,
        'new_params': new_params['w'] if new_params else None,
        'expected': expected,
        'matches': jnp.allclose(new_params['w'], expected) if new_params and expected is not None else False
    }


# =============================================================================
# Exercise 2: SGD with Momentum
# =============================================================================
def exercise_sgd_momentum():
    """
    SGD with momentum: velocity = momentum * velocity + grad
                       params = params - lr * velocity

    Momentum helps escape local minima and smooths updates.

    TODO:
    - Create SGD optimizer with learning_rate=0.1 and momentum=0.9
    - Initialize params: {'w': jnp.array([1.0, 2.0, 3.0])}
    - Use gradients: {'w': jnp.array([0.1, 0.1, 0.1])}
    - Run 5 update steps, tracking parameter history
    - Calculate step sizes to show momentum effect
    """
    # TODO: Implement this function
    learning_rate = None
    momentum = None

    optimizer = None

    params = None
    opt_state = None

    grads = None

    history = None
    step_sizes = None

    return {
        'initial': history[0] if history else None,
        'final': history[-1] if history else None,
        'step_sizes': step_sizes,
        'momentum_effect': step_sizes[-1] > step_sizes[0] if step_sizes else False
    }


# =============================================================================
# Exercise 3: Adam - Adaptive Moment Estimation
# =============================================================================
def exercise_adam():
    """
    Adam: Adaptive learning rates per parameter.
    Combines momentum (first moment) and RMSprop (second moment).

    Most popular optimizer for deep learning.

    TODO:
    - Create Adam optimizer with learning_rate=0.001
    - Initialize MLP-like params with two layers
    - Create gradients with different scales per layer
    - Run 100 update steps
    """
    # TODO: Implement this function
    learning_rate = None
    optimizer = None

    params = None
    opt_state = None
    grads = None

    return {
        'layer1_w_mean': jnp.mean(params['layer1']['w']) if params else None,
        'layer2_w_mean': jnp.mean(params['layer2']['w']) if params else None,
        'adam_adapts': True
    }


# =============================================================================
# Exercise 4: AdamW - Adam with Weight Decay
# =============================================================================
def exercise_adamw():
    """
    AdamW: Adam with decoupled weight decay regularization.
    Better for training with regularization than L2 penalty in loss.

    Weight decay: params = params * (1 - lr * weight_decay) - lr * m_hat / sqrt(v_hat)

    TODO:
    - Create AdamW optimizer with learning_rate=0.001 and weight_decay=0.01
    - Initialize params: {'w': jnp.ones((10, 10))}
    - Use zero gradients to show weight decay effect
    - Compare with regular Adam to show the difference
    """
    # TODO: Implement this function
    learning_rate = None
    weight_decay = None

    optimizer = None

    params = None
    opt_state = None
    zero_grads = None

    updates = None
    new_params = None
    shrunk = None

    # Compare with regular Adam
    adam_opt = None
    adam_state = None
    adam_updates = None
    adam_params = None

    return {
        'original': jnp.mean(params['w']) if params else None,
        'adamw_updated': jnp.mean(new_params['w']) if new_params else None,
        'adam_updated': jnp.mean(adam_params['w']) if adam_params else None,
        'adamw_shrinks': shrunk if shrunk is not None else False,
        'adam_unchanged': jnp.allclose(adam_params['w'], params['w']) if adam_params and params else False
    }


# =============================================================================
# Exercise 5: RMSprop
# =============================================================================
def exercise_rmsprop():
    """
    RMSprop: Divide gradients by running average of squared gradients.
    Good for RNNs and non-stationary problems.

    TODO:
    - Create RMSprop optimizer with learning_rate=0.01
    - Initialize params: {'w': jnp.array([1.0, 2.0, 3.0])}
    - Use gradients with different magnitudes: {'w': jnp.array([0.001, 0.1, 10.0])}
    - Show that RMSprop normalizes updates by gradient magnitude
    """
    # TODO: Implement this function
    learning_rate = None
    optimizer = None

    params = None
    opt_state = None
    grads = None

    updates = None
    new_params = None

    update_magnitudes = None

    return {
        'gradient_magnitudes': jnp.abs(grads['w']) if grads else None,
        'update_magnitudes': update_magnitudes,
        'updates_normalized': jnp.std(update_magnitudes) < jnp.std(jnp.abs(grads['w'])) if update_magnitudes is not None and grads else False
    }


# =============================================================================
# Exercise 6: Adagrad
# =============================================================================
def exercise_adagrad():
    """
    Adagrad: Accumulates squared gradients over time.
    Good for sparse gradients (NLP embeddings).
    Learning rate decreases over time.

    TODO:
    - Create Adagrad optimizer with learning_rate=0.1
    - Initialize params: {'w': jnp.ones(5)}
    - Use constant gradients: {'w': jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])}
    - Run 20 update steps, tracking update norms
    - Verify that learning rate effectively decreases over time
    """
    # TODO: Implement this function
    learning_rate = None
    optimizer = None

    params = None
    opt_state = None
    grads = None

    update_norms = None
    lr_decreases = None

    return {
        'first_update_norm': update_norms[0] if update_norms else None,
        'last_update_norm': update_norms[-1] if update_norms else None,
        'learning_rate_decreases': lr_decreases if lr_decreases is not None else False
    }


# =============================================================================
# Exercise 7: optax.chain - Combining Transformations
# =============================================================================
def exercise_chain():
    """
    optax.chain combines multiple gradient transformations.
    Order matters: transformations are applied sequentially.

    Common pattern: clip gradients, then apply optimizer.

    TODO:
    - Create a chained optimizer: clip_by_global_norm(1.0), then adam(0.001)
    - Initialize params: {'w': jnp.ones((3, 4))}
    - Use large gradients: {'w': jnp.ones((3, 4)) * 100}
    - Verify that clipping is applied before Adam processes gradients
    """
    # TODO: Implement this function
    optimizer = None

    params = None
    opt_state = None
    large_grads = None

    updates = None
    update_norm = None

    return {
        'update_norm': update_norm,
        'clipping_applied': update_norm < 100 if update_norm is not None else False,
        'chain_length': 2
    }


# =============================================================================
# Exercise 8: optax.apply_updates
# =============================================================================
def exercise_apply_updates():
    """
    apply_updates combines params and updates: new_params = params + updates.
    Works with any pytree structure.

    TODO:
    - Create nested params structure (encoder/decoder with conv/dense layers)
    - Create updates by scaling params by -0.01
    - Apply updates and verify structure is preserved
    """
    # TODO: Implement this function
    params = None
    updates = None
    new_params = None

    structure_match = None
    values_changed = None

    return {
        'structure_preserved': structure_match if structure_match is not None else False,
        'values_changed': values_changed if values_changed is not None else False,
        'original_mean': jnp.mean(params['encoder']['conv1']) if params else None,
        'new_mean': jnp.mean(new_params['encoder']['conv1']) if new_params else None
    }


# =============================================================================
# Exercise 9: Optimizer State Inspection
# =============================================================================
def exercise_state_inspection():
    """
    Inspect optimizer state to understand what's being tracked.

    TODO:
    - Create Adam optimizer with learning_rate=0.001
    - Initialize params: {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)}
    - Inspect optimizer state structure
    - Perform two updates and observe state evolution
    """
    # TODO: Implement this function
    optimizer = None

    params = None
    opt_state = None

    state_types = None
    grads = None

    updates1 = None
    updates2 = None

    return {
        'state_types': state_types,
        'num_state_components': len(opt_state) if opt_state else 0,
        'first_update_norm': float(jnp.linalg.norm(updates1['w'])) if updates1 else None,
        'second_update_norm': float(jnp.linalg.norm(updates2['w'])) if updates2 else None
    }


# =============================================================================
# Exercise 10: Custom Optimizer Composition
# =============================================================================
def exercise_custom_optimizer():
    """
    Build custom optimizers by composing transformations.

    TODO:
    - Create a custom optimizer combining:
      - clip_by_global_norm(max_grad_norm)
      - trace(decay=momentum) for momentum
      - add_decayed_weights(weight_decay)
      - scale(-lr)
    - Initialize params: {'w': jnp.ones((10, 10))}
    - Run 100 training steps with varying gradient magnitudes
    """
    # TODO: Implement this function
    def custom_sgd_with_extras(lr, momentum=0.9, weight_decay=0.0001, max_grad_norm=1.0):
        # TODO: Return a chained optimizer
        return None

    optimizer = None

    params = None
    opt_state = None

    return {
        'final_w_mean': jnp.mean(params['w']) if params else None,
        'custom_optimizer_works': True
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Basic Optimizers Exercises")
    print("=" * 60)

    exercises = [
        ("1. SGD", exercise_sgd),
        ("2. SGD with Momentum", exercise_sgd_momentum),
        ("3. Adam", exercise_adam),
        ("4. AdamW", exercise_adamw),
        ("5. RMSprop", exercise_rmsprop),
        ("6. Adagrad", exercise_adagrad),
        ("7. optax.chain", exercise_chain),
        ("8. apply_updates", exercise_apply_updates),
        ("9. State Inspection", exercise_state_inspection),
        ("10. Custom Optimizer", exercise_custom_optimizer),
    ]

    for name, func in exercises:
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
