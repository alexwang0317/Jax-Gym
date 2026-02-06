"""
JAX Stateful Computation - 10 Examples
=======================================

JAX functions must be pure (no side effects).
State must be passed explicitly as input and returned as output.

Key pattern:
  new_state, output = f(state, input)

This is how Flax, Optax, and other JAX libraries handle state.

Reference: https://jax.readthedocs.io/en/latest/jax-101/07-control-flow.html
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import NamedTuple, Any


# =============================================================================
# Example 1: Functional State Pattern
# =============================================================================
def example_functional_state():
    """
    The core pattern: pass state in, return state out.
    No global variables, no mutation.
    """
    # BAD: Global mutable state (doesn't work with JIT)
    # global_counter = 0
    # def bad_increment():
    #     global global_counter
    #     global_counter += 1
    #     return global_counter

    # GOOD: Explicit state passing
    def increment(state):
        """Pure function: state in, new state out."""
        return state + 1

    state = 0
    state = increment(state)  # 1
    state = increment(state)  # 2
    state = increment(state)  # 3

    # JIT compatible
    jit_increment = jax.jit(increment)
    state = jit_increment(state)  # 4
    state = jit_increment(state)  # 5

    # Function with input and output
    def process(state, x):
        """Process x, update state, return both."""
        new_state = state + x
        output = x * 2
        return new_state, output

    state = 0.0
    state, out1 = process(state, 1.0)  # state=1, out=2
    state, out2 = process(state, 2.0)  # state=3, out=4
    state, out3 = process(state, 3.0)  # state=6, out=6

    return {
        'counter_final': 5,
        'process_final_state': state,
        'outputs': [out1, out2, out3]
    }


# =============================================================================
# Example 2: State as Pytree
# =============================================================================
def example_state_pytree():
    """
    Complex state is stored as pytrees (nested dicts/tuples).
    This is how optimizers, models, etc. manage state.
    """
    # Define state structure
    def init_state():
        return {
            'count': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        }

    def update_state(state, x):
        """Update statistics with new value."""
        return {
            'count': state['count'] + 1,
            'sum': state['sum'] + x,
            'min': jnp.minimum(state['min'], x),
            'max': jnp.maximum(state['max'], x)
        }

    def get_mean(state):
        return state['sum'] / state['count']

    # Process data
    state = init_state()
    data = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

    for x in data:
        state = update_state(state, x)

    # Using NamedTuple for typed state
    class OptimizerState(NamedTuple):
        step: int
        momentum: jnp.ndarray
        velocity: jnp.ndarray

    def init_optimizer(params_shape):
        return OptimizerState(
            step=0,
            momentum=jnp.zeros(params_shape),
            velocity=jnp.zeros(params_shape)
        )

    opt_state = init_optimizer((10,))

    return {
        'final_count': state['count'],
        'final_sum': state['sum'],
        'mean': get_mean(state),
        'min': state['min'],
        'max': state['max'],
        'opt_state_step': opt_state.step,
        'opt_state_momentum_shape': opt_state.momentum.shape
    }


# =============================================================================
# Example 3: Accumulator Patterns
# =============================================================================
def example_accumulator():
    """
    Common pattern: accumulate values over iterations.
    """
    # Simple accumulator
    def accumulate(state, values):
        """Sum values into state."""
        for v in values:
            state = state + v
        return state

    total = accumulate(0.0, [1.0, 2.0, 3.0, 4.0, 5.0])

    # Accumulate with lax.fori_loop (JIT compatible)
    def accumulate_jit(arr):
        def body(i, acc):
            return acc + arr[i]
        return lax.fori_loop(0, len(arr), body, 0.0)

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    total_jit = jax.jit(accumulate_jit)(arr)

    # Accumulate multiple values
    def multi_accumulate(state, x):
        """Accumulate sum and sum of squares."""
        sum_acc, sq_acc, count = state
        return (sum_acc + x, sq_acc + x**2, count + 1)

    state = (0.0, 0.0, 0)
    for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        state = multi_accumulate(state, x)

    sum_val, sq_sum, count = state
    mean = sum_val / count
    variance = sq_sum / count - mean ** 2

    return {
        'simple_total': total,
        'jit_total': total_jit,
        'mean': mean,
        'variance': variance
    }


# =============================================================================
# Example 4: Counter Implementation
# =============================================================================
def example_counter():
    """
    Implement a counter that works with JAX transformations.
    """
    class Counter(NamedTuple):
        value: int

    def make_counter(initial=0):
        return Counter(value=initial)

    def increment(counter, amount=1):
        return Counter(value=counter.value + amount)

    def decrement(counter, amount=1):
        return Counter(value=counter.value - amount)

    def reset(counter):
        return Counter(value=0)

    # Usage
    c = make_counter(0)
    c = increment(c)      # 1
    c = increment(c)      # 2
    c = increment(c, 5)   # 7
    c = decrement(c, 2)   # 5

    # Works with JIT
    @jax.jit
    def batch_increment(counter, n):
        def body(i, c):
            return increment(c)
        return lax.fori_loop(0, n, body, counter)

    c = make_counter(0)
    c = batch_increment(c, 100)

    return {
        'manual_count': 5,
        'batch_count': c.value
    }


# =============================================================================
# Example 5: Running Statistics (Mean, Variance)
# =============================================================================
def example_running_stats():
    """
    Compute running mean and variance with Welford's algorithm.
    """
    class RunningStats(NamedTuple):
        count: int
        mean: float
        M2: float  # Sum of squared differences

    def init_stats():
        return RunningStats(count=0, mean=0.0, M2=0.0)

    def update_stats(stats, x):
        """Welford's online algorithm."""
        count = stats.count + 1
        delta = x - stats.mean
        mean = stats.mean + delta / count
        delta2 = x - mean
        M2 = stats.M2 + delta * delta2
        return RunningStats(count=count, mean=mean, M2=M2)

    def get_variance(stats):
        if stats.count < 2:
            return 0.0
        return stats.M2 / (stats.count - 1)

    def get_std(stats):
        return jnp.sqrt(get_variance(stats))

    # Process data
    data = jnp.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    stats = init_stats()

    for x in data:
        stats = update_stats(stats, x)

    # Verify against numpy
    np_mean = jnp.mean(data)
    np_var = jnp.var(data, ddof=1)  # Sample variance

    return {
        'running_mean': stats.mean,
        'numpy_mean': np_mean,
        'running_var': get_variance(stats),
        'numpy_var': np_var,
        'running_std': get_std(stats),
        'mean_matches': jnp.allclose(stats.mean, np_mean),
        'var_matches': jnp.allclose(get_variance(stats), np_var)
    }


# =============================================================================
# Example 6: Stateful RNG Handling
# =============================================================================
def example_stateful_rng():
    """
    Managing random state in a stateful manner.
    """
    class RNGState(NamedTuple):
        key: Any  # JAX PRNG key

    def init_rng(seed):
        return RNGState(key=random.key(seed))

    def next_key(state):
        """Get next key and update state."""
        key, subkey = random.split(state.key)
        return RNGState(key=key), subkey

    def random_normal(state, shape):
        """Sample normal and update state."""
        new_state, subkey = next_key(state)
        sample = random.normal(subkey, shape)
        return new_state, sample

    def random_uniform(state, shape, minval=0.0, maxval=1.0):
        """Sample uniform and update state."""
        new_state, subkey = next_key(state)
        sample = random.uniform(subkey, shape, minval=minval, maxval=maxval)
        return new_state, sample

    # Usage
    rng = init_rng(42)
    rng, sample1 = random_normal(rng, (3,))
    rng, sample2 = random_normal(rng, (3,))
    rng, sample3 = random_uniform(rng, (3,), minval=-1, maxval=1)

    # Samples should be different
    samples_different = not jnp.allclose(sample1, sample2)

    # Reproducibility
    rng1 = init_rng(42)
    rng1, s1 = random_normal(rng1, (3,))

    rng2 = init_rng(42)
    rng2, s2 = random_normal(rng2, (3,))

    return {
        'sample1': sample1,
        'sample2': sample2,
        'samples_different': samples_different,
        'reproducible': jnp.allclose(s1, s2)
    }


# =============================================================================
# Example 7: Optimizer State Management
# =============================================================================
def example_optimizer_state():
    """
    How optimizers manage state (simplified Adam).
    """
    class AdamState(NamedTuple):
        step: int
        m: Any  # First moment (pytree)
        v: Any  # Second moment (pytree)

    def init_adam(params):
        """Initialize Adam state."""
        return AdamState(
            step=0,
            m=jax.tree.map(jnp.zeros_like, params),
            v=jax.tree.map(jnp.zeros_like, params)
        )

    def adam_update(state, params, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Perform Adam update."""
        step = state.step + 1

        # Update moments
        m = jax.tree.map(
            lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i,
            state.m, grads
        )
        v = jax.tree.map(
            lambda v_i, g_i: beta2 * v_i + (1 - beta2) * g_i ** 2,
            state.v, grads
        )

        # Bias correction
        m_hat = jax.tree.map(lambda m_i: m_i / (1 - beta1 ** step), m)
        v_hat = jax.tree.map(lambda v_i: v_i / (1 - beta2 ** step), v)

        # Update params
        new_params = jax.tree.map(
            lambda p, m_i, v_i: p - lr * m_i / (jnp.sqrt(v_i) + eps),
            params, m_hat, v_hat
        )

        new_state = AdamState(step=step, m=m, v=v)
        return new_state, new_params

    # Example usage
    params = {
        'w': jnp.ones((3, 4)),
        'b': jnp.zeros(4)
    }
    grads = {
        'w': jnp.ones((3, 4)) * 0.1,
        'b': jnp.ones(4) * 0.01
    }

    state = init_adam(params)

    # Multiple updates
    for _ in range(10):
        state, params = adam_update(state, params, grads)

    return {
        'final_step': state.step,
        'params_w_mean': jnp.mean(params['w']),
        'params_b_mean': jnp.mean(params['b']),
        'm_w_shape': state.m['w'].shape,
        'v_b_shape': state.v['b'].shape
    }


# =============================================================================
# Example 8: BatchNorm Statistics
# =============================================================================
def example_batchnorm_stats():
    """
    BatchNorm maintains running mean/var statistics.
    Different behavior in training vs inference.
    """
    class BatchNormState(NamedTuple):
        running_mean: jnp.ndarray
        running_var: jnp.ndarray
        num_updates: int

    def init_batchnorm(num_features):
        return BatchNormState(
            running_mean=jnp.zeros(num_features),
            running_var=jnp.ones(num_features),
            num_updates=0
        )

    def batchnorm(state, x, training=True, momentum=0.1, eps=1e-5):
        """
        Apply batch normalization.
        x: (batch, features)
        """
        if training:
            # Compute batch statistics
            batch_mean = jnp.mean(x, axis=0)
            batch_var = jnp.var(x, axis=0)

            # Update running statistics
            running_mean = (1 - momentum) * state.running_mean + momentum * batch_mean
            running_var = (1 - momentum) * state.running_var + momentum * batch_var

            new_state = BatchNormState(
                running_mean=running_mean,
                running_var=running_var,
                num_updates=state.num_updates + 1
            )

            # Normalize using batch statistics
            x_norm = (x - batch_mean) / jnp.sqrt(batch_var + eps)
        else:
            # Use running statistics
            new_state = state
            x_norm = (x - state.running_mean) / jnp.sqrt(state.running_var + eps)

        return new_state, x_norm

    # Training
    num_features = 5
    state = init_batchnorm(num_features)

    key = random.key(42)
    for i in range(10):
        key, subkey = random.split(key)
        x = random.normal(subkey, (32, num_features)) * (i + 1) + i  # Varying stats

        state, x_norm = batchnorm(state, x, training=True)

    # Inference
    key, subkey = random.split(key)
    x_test = random.normal(subkey, (16, num_features)) * 5 + 5
    _, x_test_norm = batchnorm(state, x_test, training=False)

    return {
        'running_mean': state.running_mean,
        'running_var': state.running_var,
        'num_updates': state.num_updates,
        'test_output_shape': x_test_norm.shape
    }


# =============================================================================
# Example 9: Memoization Patterns
# =============================================================================
def example_memoization():
    """
    Caching computed values in state.
    """
    class CacheState(NamedTuple):
        cache: dict

    def init_cache():
        return CacheState(cache={})

    def memoized_expensive_fn(state, x):
        """Cache expensive computation results."""
        x_key = float(x)  # Convert to hashable

        if x_key in state.cache:
            return state, state.cache[x_key]

        # Expensive computation
        result = jnp.sum(jnp.sin(jnp.arange(1000) * x))

        # Update cache
        new_cache = state.cache.copy()
        new_cache[x_key] = result
        new_state = CacheState(cache=new_cache)

        return new_state, result

    # Note: This pattern doesn't work well with JIT
    # For JIT, use functional caching with explicit lookup

    # JIT-compatible pattern: precompute lookup table
    def create_lookup_table(values):
        """Precompute results for known values."""
        results = jnp.array([jnp.sum(jnp.sin(jnp.arange(1000) * v)) for v in values])
        return results

    lookup_values = jnp.linspace(0, 1, 100)
    lookup_table = create_lookup_table(lookup_values)

    @jax.jit
    def fast_lookup(x, table, values):
        """Find nearest value in precomputed table."""
        idx = jnp.argmin(jnp.abs(values - x))
        return table[idx]

    result = fast_lookup(0.5, lookup_table, lookup_values)

    return {
        'lookup_table_size': len(lookup_table),
        'lookup_result': result,
        'note': 'Use precomputed tables for JIT compatibility'
    }


# =============================================================================
# Example 10: State Checkpointing
# =============================================================================
def example_checkpointing():
    """
    Saving and restoring state for training resumption.
    """
    class TrainingState(NamedTuple):
        step: int
        params: Any
        opt_state: Any
        best_loss: float

    def init_training_state(params):
        return TrainingState(
            step=0,
            params=params,
            opt_state={'m': jax.tree.map(jnp.zeros_like, params),
                       'v': jax.tree.map(jnp.zeros_like, params)},
            best_loss=float('inf')
        )

    def save_checkpoint(state):
        """Convert state to serializable format."""
        return {
            'step': int(state.step),
            'params': jax.tree.map(lambda x: x.tolist(), state.params),
            'opt_state': {
                'm': jax.tree.map(lambda x: x.tolist(), state.opt_state['m']),
                'v': jax.tree.map(lambda x: x.tolist(), state.opt_state['v'])
            },
            'best_loss': float(state.best_loss)
        }

    def load_checkpoint(checkpoint):
        """Restore state from checkpoint."""
        return TrainingState(
            step=checkpoint['step'],
            params=jax.tree.map(jnp.array, checkpoint['params']),
            opt_state={
                'm': jax.tree.map(jnp.array, checkpoint['opt_state']['m']),
                'v': jax.tree.map(jnp.array, checkpoint['opt_state']['v'])
            },
            best_loss=checkpoint['best_loss']
        )

    # Example usage
    params = {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)}
    state = init_training_state(params)

    # Simulate training
    state = TrainingState(
        step=100,
        params={'w': params['w'] - 0.1, 'b': params['b'] + 0.01},
        opt_state=state.opt_state,
        best_loss=0.5
    )

    # Save and load
    checkpoint = save_checkpoint(state)
    restored_state = load_checkpoint(checkpoint)

    return {
        'original_step': state.step,
        'restored_step': restored_state.step,
        'params_match': jnp.allclose(state.params['w'], restored_state.params['w']),
        'loss_match': state.best_loss == restored_state.best_loss
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Stateful Computation Examples")
    print("=" * 60)

    examples = [
        ("1. Functional State", example_functional_state),
        ("2. State as Pytree", example_state_pytree),
        ("3. Accumulator", example_accumulator),
        ("4. Counter", example_counter),
        ("5. Running Statistics", example_running_stats),
        ("6. Stateful RNG", example_stateful_rng),
        ("7. Optimizer State", example_optimizer_state),
        ("8. BatchNorm Stats", example_batchnorm_stats),
        ("9. Memoization", example_memoization),
        ("10. Checkpointing", example_checkpointing),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
