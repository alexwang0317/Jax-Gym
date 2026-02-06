"""
JAX Stateful Computation - 10 Exercises
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
# Exercise 1: Functional State Pattern
# =============================================================================
def exercise_functional_state():
    """
    The core pattern: pass state in, return state out.
    No global variables, no mutation.

    TODO:
    - Implement increment(state) that returns state + 1
    - Start with state = 0, call increment 3 times, then JIT compile and call 2 more times
    - Implement process(state, x) that returns (state + x, x * 2)
    - Start with state = 0.0, process values 1.0, 2.0, 3.0

    Returns dict with:
    - 'counter_final': final counter value (should be 5)
    - 'process_final_state': final state from process (should be 6.0)
    - 'outputs': list of outputs from process (should be [2.0, 4.0, 6.0])
    """
    # TODO: Implement this function

    def increment(state):
        """Pure function: state in, new state out."""
        # TODO: Implement this function
        pass

    state = None
    # TODO: Call increment 3 times

    # TODO: JIT compile increment and call 2 more times
    jit_increment = None

    def process(state, x):
        """Process x, update state, return both."""
        # TODO: Implement this function
        pass

    state = None
    out1 = None
    out2 = None
    out3 = None

    return {
        'counter_final': None,
        'process_final_state': None,
        'outputs': None
    }


# =============================================================================
# Exercise 2: State as Pytree
# =============================================================================
def exercise_state_pytree():
    """
    Complex state is stored as pytrees (nested dicts/tuples).
    This is how optimizers, models, etc. manage state.

    TODO:
    - Implement init_state() returning dict with count=0, sum=0.0, min=inf, max=-inf
    - Implement update_state(state, x) that updates all statistics
    - Implement get_mean(state) that returns sum/count
    - Process data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    - Create OptimizerState NamedTuple with step, momentum, velocity
    - Initialize optimizer state with shape (10,)

    Returns dict with:
    - 'final_count': count after processing all data (should be 8)
    - 'final_sum': sum after processing all data (should be 31.0)
    - 'mean': mean of all data (should be 3.875)
    - 'min': min value (should be 1.0)
    - 'max': max value (should be 9.0)
    - 'opt_state_step': initial step (should be 0)
    - 'opt_state_momentum_shape': shape of momentum (should be (10,))
    """
    # TODO: Implement this function

    def init_state():
        # TODO: Implement this function
        pass

    def update_state(state, x):
        """Update statistics with new value."""
        # TODO: Implement this function
        pass

    def get_mean(state):
        # TODO: Implement this function
        pass

    # Process data
    state = None
    data = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

    # TODO: Process all data through update_state

    # TODO: Create OptimizerState NamedTuple
    class OptimizerState(NamedTuple):
        step: int
        momentum: jnp.ndarray
        velocity: jnp.ndarray

    def init_optimizer(params_shape):
        # TODO: Implement this function
        pass

    opt_state = None

    return {
        'final_count': None,
        'final_sum': None,
        'mean': None,
        'min': None,
        'max': None,
        'opt_state_step': None,
        'opt_state_momentum_shape': None
    }


# =============================================================================
# Exercise 3: Accumulator Patterns
# =============================================================================
def exercise_accumulator():
    """
    Common pattern: accumulate values over iterations.

    TODO:
    - Implement accumulate(state, values) that sums values into state using a loop
    - Implement accumulate_jit(arr) using lax.fori_loop for JIT compatibility
    - Implement multi_accumulate(state, x) that tracks sum, sum of squares, and count
    - Calculate mean and variance from the accumulated values

    Returns dict with:
    - 'simple_total': total from simple accumulate (should be 15.0)
    - 'jit_total': total from JIT accumulate (should be 15.0)
    - 'mean': mean from multi accumulate (should be 3.0)
    - 'variance': variance from multi accumulate (should be 2.0)
    """
    # TODO: Implement this function

    def accumulate(state, values):
        """Sum values into state."""
        # TODO: Implement this function
        pass

    total = None

    def accumulate_jit(arr):
        # TODO: Implement using lax.fori_loop
        pass

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    total_jit = None

    def multi_accumulate(state, x):
        """Accumulate sum and sum of squares."""
        # TODO: Implement this function
        pass

    state = None
    # TODO: Process values [1.0, 2.0, 3.0, 4.0, 5.0]

    sum_val = None
    sq_sum = None
    count = None
    mean = None
    variance = None

    return {
        'simple_total': None,
        'jit_total': None,
        'mean': None,
        'variance': None
    }


# =============================================================================
# Exercise 4: Counter Implementation
# =============================================================================
def exercise_counter():
    """
    Implement a counter that works with JAX transformations.

    TODO:
    - Create Counter NamedTuple with value field
    - Implement make_counter(initial=0)
    - Implement increment(counter, amount=1)
    - Implement decrement(counter, amount=1)
    - Use the counter: 0 -> +1 -> +1 -> +5 -> -2 = 5
    - Implement batch_increment using lax.fori_loop

    Returns dict with:
    - 'manual_count': final count after manual operations (should be 5)
    - 'batch_count': count after batch_increment 100 times (should be 100)
    """
    # TODO: Implement this function

    class Counter(NamedTuple):
        value: int

    def make_counter(initial=0):
        # TODO: Implement this function
        pass

    def increment(counter, amount=1):
        # TODO: Implement this function
        pass

    def decrement(counter, amount=1):
        # TODO: Implement this function
        pass

    def reset(counter):
        # TODO: Implement this function
        pass

    # Usage
    c = None
    # TODO: increment twice, increment by 5, decrement by 2

    # TODO: Implement batch_increment with JIT and lax.fori_loop
    @jax.jit
    def batch_increment(counter, n):
        # TODO: Implement this function
        pass

    c_batch = None

    return {
        'manual_count': None,
        'batch_count': None
    }


# =============================================================================
# Exercise 5: Running Statistics (Mean, Variance)
# =============================================================================
def exercise_running_stats():
    """
    Compute running mean and variance with Welford's algorithm.

    TODO:
    - Create RunningStats NamedTuple with count, mean, M2 (sum of squared differences)
    - Implement init_stats() returning initial state
    - Implement update_stats(stats, x) using Welford's online algorithm:
      - count = stats.count + 1
      - delta = x - stats.mean
      - mean = stats.mean + delta / count
      - delta2 = x - mean
      - M2 = stats.M2 + delta * delta2
    - Implement get_variance(stats) returning M2 / (count - 1) for sample variance
    - Process data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    - Compare with numpy mean and variance

    Returns dict with:
    - 'running_mean': mean from running stats
    - 'numpy_mean': mean from jnp.mean
    - 'running_var': variance from running stats
    - 'numpy_var': variance from jnp.var with ddof=1
    - 'running_std': standard deviation from running stats
    - 'mean_matches': bool, True if means match
    - 'var_matches': bool, True if variances match
    """
    # TODO: Implement this function

    class RunningStats(NamedTuple):
        count: int
        mean: float
        M2: float

    def init_stats():
        # TODO: Implement this function
        pass

    def update_stats(stats, x):
        """Welford's online algorithm."""
        # TODO: Implement this function
        pass

    def get_variance(stats):
        # TODO: Implement this function
        pass

    def get_std(stats):
        # TODO: Implement this function
        pass

    # Process data
    data = jnp.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    stats = None
    # TODO: Process all data

    # Verify against numpy
    np_mean = None
    np_var = None

    return {
        'running_mean': None,
        'numpy_mean': None,
        'running_var': None,
        'numpy_var': None,
        'running_std': None,
        'mean_matches': None,
        'var_matches': None
    }


# =============================================================================
# Exercise 6: Stateful RNG Handling
# =============================================================================
def exercise_stateful_rng():
    """
    Managing random state in a stateful manner.

    TODO:
    - Create RNGState NamedTuple with key field
    - Implement init_rng(seed) to create initial state
    - Implement next_key(state) that splits key and returns (new_state, subkey)
    - Implement random_normal(state, shape) that samples normal and updates state
    - Implement random_uniform(state, shape, minval, maxval) similarly
    - Generate samples and verify they're different
    - Verify reproducibility with same seed

    Returns dict with:
    - 'sample1': first normal sample (shape (3,))
    - 'sample2': second normal sample (shape (3,))
    - 'samples_different': bool, True if sample1 != sample2
    - 'reproducible': bool, True if same seed gives same first sample
    """
    # TODO: Implement this function

    class RNGState(NamedTuple):
        key: Any

    def init_rng(seed):
        # TODO: Implement this function
        pass

    def next_key(state):
        """Get next key and update state."""
        # TODO: Implement this function
        pass

    def random_normal(state, shape):
        """Sample normal and update state."""
        # TODO: Implement this function
        pass

    def random_uniform(state, shape, minval=0.0, maxval=1.0):
        """Sample uniform and update state."""
        # TODO: Implement this function
        pass

    # Usage
    rng = None
    sample1 = None
    sample2 = None
    sample3 = None

    samples_different = None

    # Reproducibility test
    s1 = None
    s2 = None

    return {
        'sample1': None,
        'sample2': None,
        'samples_different': None,
        'reproducible': None
    }


# =============================================================================
# Exercise 7: Optimizer State Management
# =============================================================================
def exercise_optimizer_state():
    """
    How optimizers manage state (simplified Adam).

    TODO:
    - Create AdamState NamedTuple with step, m (first moment), v (second moment)
    - Implement init_adam(params) that initializes state with zeros
    - Implement adam_update(state, params, grads, lr, beta1, beta2, eps) that:
      - Updates step
      - Updates moments: m = beta1*m + (1-beta1)*g, v = beta2*v + (1-beta2)*g^2
      - Applies bias correction
      - Updates params: p = p - lr * m_hat / (sqrt(v_hat) + eps)
    - Use params = {'w': ones((3,4)), 'b': zeros(4)}
    - Run 10 updates with grads = {'w': 0.1*ones, 'b': 0.01*ones}

    Returns dict with:
    - 'final_step': step after 10 updates (should be 10)
    - 'params_w_mean': mean of w after updates (should be < 1.0)
    - 'params_b_mean': mean of b after updates (should be < 0.0)
    - 'm_w_shape': shape of m['w'] (should be (3, 4))
    - 'v_b_shape': shape of v['b'] (should be (4,))
    """
    # TODO: Implement this function

    class AdamState(NamedTuple):
        step: int
        m: Any
        v: Any

    def init_adam(params):
        """Initialize Adam state."""
        # TODO: Implement this function
        pass

    def adam_update(state, params, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Perform Adam update."""
        # TODO: Implement this function
        pass

    # Example usage
    params = {
        'w': jnp.ones((3, 4)),
        'b': jnp.zeros(4)
    }
    grads = {
        'w': jnp.ones((3, 4)) * 0.1,
        'b': jnp.ones(4) * 0.01
    }

    state = None
    # TODO: Run 10 updates

    return {
        'final_step': None,
        'params_w_mean': None,
        'params_b_mean': None,
        'm_w_shape': None,
        'v_b_shape': None
    }


# =============================================================================
# Exercise 8: BatchNorm Statistics
# =============================================================================
def exercise_batchnorm_stats():
    """
    BatchNorm maintains running mean/var statistics.
    Different behavior in training vs inference.

    TODO:
    - Create BatchNormState NamedTuple with running_mean, running_var, num_updates
    - Implement init_batchnorm(num_features)
    - Implement batchnorm(state, x, training, momentum, eps) that:
      - In training: computes batch stats, updates running stats, normalizes with batch stats
      - In inference: normalizes with running stats
    - Train for 10 batches with varying statistics
    - Test inference mode

    Returns dict with:
    - 'running_mean': final running mean (shape (5,))
    - 'running_var': final running variance (shape (5,))
    - 'num_updates': number of training updates (should be 10)
    - 'test_output_shape': shape of test output (should be (16, 5))
    """
    # TODO: Implement this function

    class BatchNormState(NamedTuple):
        running_mean: jnp.ndarray
        running_var: jnp.ndarray
        num_updates: int

    def init_batchnorm(num_features):
        # TODO: Implement this function
        pass

    def batchnorm(state, x, training=True, momentum=0.1, eps=1e-5):
        """
        Apply batch normalization.
        x: (batch, features)
        """
        # TODO: Implement this function
        pass

    # Training
    num_features = 5
    state = None

    key = random.key(42)
    # TODO: Train for 10 batches

    # Inference
    x_test_norm = None

    return {
        'running_mean': None,
        'running_var': None,
        'num_updates': None,
        'test_output_shape': None
    }


# =============================================================================
# Exercise 9: Memoization Patterns
# =============================================================================
def exercise_memoization():
    """
    Caching computed values in state.

    TODO:
    - Create CacheState NamedTuple with cache dict
    - Implement init_cache()
    - Implement memoized_expensive_fn that caches results
    - For JIT compatibility, implement lookup table pattern:
      - create_lookup_table(values) precomputes results
      - fast_lookup(x, table, values) finds nearest precomputed value
    - Create lookup table for 100 values in [0, 1]

    Returns dict with:
    - 'lookup_table_size': size of lookup table (should be 100)
    - 'lookup_result': result of looking up 0.5
    - 'note': 'Use precomputed tables for JIT compatibility'
    """
    # TODO: Implement this function

    class CacheState(NamedTuple):
        cache: dict

    def init_cache():
        # TODO: Implement this function
        pass

    def memoized_expensive_fn(state, x):
        """Cache expensive computation results."""
        # TODO: Implement this function
        pass

    # JIT-compatible pattern: precompute lookup table
    def create_lookup_table(values):
        """Precompute results for known values."""
        # TODO: Implement this function
        pass

    lookup_values = jnp.linspace(0, 1, 100)
    lookup_table = None

    @jax.jit
    def fast_lookup(x, table, values):
        """Find nearest value in precomputed table."""
        # TODO: Implement this function
        pass

    result = None

    return {
        'lookup_table_size': None,
        'lookup_result': None,
        'note': 'Use precomputed tables for JIT compatibility'
    }


# =============================================================================
# Exercise 10: State Checkpointing
# =============================================================================
def exercise_checkpointing():
    """
    Saving and restoring state for training resumption.

    TODO:
    - Create TrainingState NamedTuple with step, params, opt_state, best_loss
    - Implement init_training_state(params) with step=0, best_loss=inf
    - Implement save_checkpoint(state) converting to serializable format (lists, ints, floats)
    - Implement load_checkpoint(checkpoint) restoring to JAX arrays
    - Test save/load roundtrip

    Returns dict with:
    - 'original_step': step before save (should be 100)
    - 'restored_step': step after load (should be 100)
    - 'params_match': bool, True if params match after roundtrip
    - 'loss_match': bool, True if best_loss matches after roundtrip
    """
    # TODO: Implement this function

    class TrainingState(NamedTuple):
        step: int
        params: Any
        opt_state: Any
        best_loss: float

    def init_training_state(params):
        # TODO: Implement this function
        pass

    def save_checkpoint(state):
        """Convert state to serializable format."""
        # TODO: Implement this function
        pass

    def load_checkpoint(checkpoint):
        """Restore state from checkpoint."""
        # TODO: Implement this function
        pass

    # Example usage
    params = {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)}
    state = None

    # Simulate training
    # TODO: Update state to step=100, modify params, set best_loss=0.5

    # Save and load
    checkpoint = None
    restored_state = None

    return {
        'original_step': None,
        'restored_step': None,
        'params_match': None,
        'loss_match': None
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Stateful Computation Exercises")
    print("=" * 60)

    exercises = [
        ("1. Functional State", exercise_functional_state),
        ("2. State as Pytree", exercise_state_pytree),
        ("3. Accumulator", exercise_accumulator),
        ("4. Counter", exercise_counter),
        ("5. Running Statistics", exercise_running_stats),
        ("6. Stateful RNG", exercise_stateful_rng),
        ("7. Optimizer State", exercise_optimizer_state),
        ("8. BatchNorm Stats", exercise_batchnorm_stats),
        ("9. Memoization", exercise_memoization),
        ("10. Checkpointing", exercise_checkpointing),
    ]

    for name, func in exercises:
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
