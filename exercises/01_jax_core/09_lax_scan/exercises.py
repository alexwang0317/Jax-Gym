"""
JAX lax.scan - 10 Exercises
===========================

lax.scan is JAX's efficient way to write sequential operations.
It compiles to a single fused loop, much faster than Python loops.

Signature: lax.scan(f, init, xs) -> (final_carry, stacked_outputs)
- f(carry, x) -> (new_carry, output)
- init: initial carry value
- xs: sequence to scan over

Reference: https://jax.readthedocs.io/en/latest/jax-101/05-control-flow.html
"""

import jax
import jax.numpy as jnp
from jax import lax, random
import time


# =============================================================================
# Exercise 1: Basic lax.scan for Cumulative Sum
# =============================================================================
def exercise_basic_scan():
    """
    Basic scan: carry state through a sequence.

    Implement a cumulative sum using lax.scan.
    The scan function should:
    - Take a running total (carry) and current element (x)
    - Return the new running total as both new_carry and output

    Expected: cumsum = [1, 3, 6, 10, 15] for input [1, 2, 3, 4, 5]
    """
    # TODO: Implement the scan function
    def scan_fn(carry, x):
        """
        carry: running total
        x: current element
        returns: (new_carry, output)
        """
        # TODO: Implement this function
        new_carry = None
        output = None
        return new_carry, output

    # Input sequence
    xs = jnp.array([1, 2, 3, 4, 5])

    # TODO: Run scan with scan_fn, init=0, and xs
    final_carry = None
    cumsum = None

    # Compare with jnp.cumsum
    expected = jnp.cumsum(xs)

    return {
        'cumsum': cumsum,
        'final_carry': final_carry,
        'expected': expected,
        'matches': jnp.allclose(cumsum, expected) if cumsum is not None else False
    }


# =============================================================================
# Exercise 2: lax.scan vs Python Loops (Performance)
# =============================================================================
def exercise_performance():
    """
    Demonstrate performance advantage of scan over Python loops.

    Implement three versions of array sum:
    1. Python loop (inside JIT)
    2. lax.fori_loop
    3. lax.scan

    All should return the same result (sum of array elements).
    """
    n = 1000

    # Python loop version (inside JIT - unrolled at compile time)
    @jax.jit
    def python_loop_sum(arr):
        total = 0.0
        for i in range(len(arr)):
            total = total + arr[i]
        return total

    # TODO: Implement lax.fori_loop version
    @jax.jit
    def fori_loop_sum(arr):
        # TODO: Implement this function
        # Use lax.fori_loop(0, len(arr), body_fn, init_val)
        # body_fn takes (i, total) and returns new total
        return None

    # TODO: Implement lax.scan version
    @jax.jit
    def scan_sum(arr):
        # TODO: Implement this function
        # Use lax.scan with a function that accumulates sum
        # Output can be None since we only need final carry
        return None

    arr = jnp.ones(n)

    # Warm up
    _ = python_loop_sum(arr)
    result_fori = fori_loop_sum(arr) if fori_loop_sum(arr) is not None else None
    result_scan = scan_sum(arr) if scan_sum(arr) is not None else None

    # Results should match
    result_python = python_loop_sum(arr)
    result_fori = fori_loop_sum(arr)
    result_scan = scan_sum(arr)

    return {
        'python_result': result_python,
        'fori_result': result_fori,
        'scan_result': result_scan,
        'all_match': (jnp.allclose(result_python, result_fori) and jnp.allclose(result_fori, result_scan)) if result_fori is not None and result_scan is not None else False
    }


# =============================================================================
# Exercise 3: Carrying State Through Scan
# =============================================================================
def exercise_carry_state():
    """
    Carry complex state (pytree) through scan.

    Track running statistics (count, sum, sum_squared) through a sequence.
    At each step, output the running mean and variance.

    carry = (count, sum_val, sum_sq)
    output = (running_mean, running_variance)
    """
    # TODO: Implement scan function that tracks multiple running statistics
    def scan_fn(carry, x):
        """Track multiple running statistics."""
        count, sum_val, sum_sq = carry
        # TODO: Implement this function
        # Update count, sum, and sum of squares
        # Output running mean and variance at each step
        new_count = None
        new_sum = None
        new_sum_sq = None
        output = (None, None)  # (running_mean, running_variance)
        return (new_count, new_sum, new_sum_sq), output

    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # TODO: Run scan with initial state (0, 0.0, 0.0)
    init = (0, 0.0, 0.0)  # count, sum, sum_sq
    final_carry = None
    outputs = None

    # Extract results (handle None case)
    if final_carry is not None and outputs is not None:
        running_means, running_vars = outputs
        count, total, total_sq = final_carry
        final_mean = total / count
        final_var = total_sq / count - final_mean ** 2
    else:
        running_means, running_vars = None, None
        count, final_mean, final_var = None, None, None

    return {
        'final_count': count,
        'final_mean': final_mean,
        'expected_mean': jnp.mean(xs),
        'running_means': running_means,
        'running_vars': running_vars
    }


# =============================================================================
# Exercise 4: lax.scan for RNN Forward Pass
# =============================================================================
def exercise_rnn_scan():
    """
    Implement RNN forward pass using scan.
    Much more efficient than Python loops.

    RNN cell: h_new = tanh(W_h @ h + W_x @ x + b)
    Use scan to process entire sequence.
    """
    def rnn_cell(params, h, x):
        """Simple RNN cell: h_new = tanh(W_h @ h + W_x @ x + b)."""
        W_h, W_x, b = params['W_h'], params['W_x'], params['b']
        return jnp.tanh(W_h @ h + W_x @ x + b)

    def rnn_forward(params, h0, inputs):
        """
        Forward pass through RNN using scan.
        inputs: (seq_len, input_dim)
        returns: (final_h, all_hidden_states)
        """
        # TODO: Implement scan function for RNN
        def scan_fn(h, x):
            # TODO: Implement this function
            # Use rnn_cell to compute new hidden state
            # Return (new_h, new_h) since both carry and output are h_new
            h_new = None
            return h_new, h_new

        # TODO: Run scan
        final_h = None
        hidden_states = None
        return final_h, hidden_states

    # Initialize
    key = random.key(42)
    hidden_dim, input_dim, seq_len = 32, 16, 50

    keys = random.split(key, 3)
    params = {
        'W_h': random.normal(keys[0], (hidden_dim, hidden_dim)) * 0.1,
        'W_x': random.normal(keys[1], (hidden_dim, input_dim)) * 0.1,
        'b': jnp.zeros(hidden_dim)
    }

    h0 = jnp.zeros(hidden_dim)
    inputs = random.normal(keys[2], (seq_len, input_dim))

    # Run forward pass
    final_h, hidden_states = rnn_forward(params, h0, inputs)

    # JIT compile
    jit_rnn = jax.jit(lambda h, x: rnn_forward(params, h, x))
    final_h_jit, hidden_states_jit = jit_rnn(h0, inputs)

    return {
        'hidden_states_shape': hidden_states.shape if hidden_states is not None else None,
        'final_h_shape': final_h.shape if final_h is not None else None,
        'jit_matches': jnp.allclose(hidden_states, hidden_states_jit) if hidden_states is not None and hidden_states_jit is not None else False
    }


# =============================================================================
# Exercise 5: Reverse Scan
# =============================================================================
def exercise_reverse_scan():
    """
    Scan in reverse order (for backward RNN, etc.).

    Use the reverse=True parameter in lax.scan.
    Also implement bidirectional scan (forward + reverse).
    """
    def scan_fn(carry, x):
        return carry + x, carry + x

    xs = jnp.array([1, 2, 3, 4, 5])

    # TODO: Forward scan
    forward_cumsum = None

    # TODO: Reverse scan (use reverse=True parameter)
    reverse_cumsum = None

    # TODO: Manually reverse for comparison
    manual_reverse = None

    # TODO: Bidirectional pattern (forward + reverse stacked)
    def bidirectional_scan(xs):
        # TODO: Implement this function
        # Return shape should be (5, 2) with forward and reverse stacked
        return None

    combined = bidirectional_scan(xs)

    return {
        'forward': forward_cumsum,
        'reverse': reverse_cumsum,
        'manual_reverse': manual_reverse,
        'bidirectional_shape': combined.shape if combined is not None else None
    }


# =============================================================================
# Exercise 6: lax.scan with Variable-Length Sequences
# =============================================================================
def exercise_variable_length():
    """
    Handle variable-length sequences with masking.

    Use jnp.where to conditionally update the carry based on a mask.
    """
    # TODO: Implement masked scan function
    def masked_scan_fn(carry, inputs):
        """Process only if mask is True."""
        x, mask = inputs
        # TODO: Implement this function
        # Update carry only if mask is True, otherwise keep old carry
        new_carry = None
        return new_carry, new_carry

    # Padded sequence with mask
    xs = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0])  # Actual length is 3
    mask = jnp.array([True, True, True, False, False])

    # TODO: Run scan with masked function
    final = None
    outputs = None

    # Alternative: use jnp.where inside scan
    def length_aware_sum(xs, length):
        def scan_fn(carry, inputs):
            x, idx = inputs
            # TODO: Implement this function
            # Only add if idx < length
            new_carry = None
            return new_carry, new_carry

        indices = jnp.arange(len(xs))
        return lax.scan(scan_fn, 0.0, (xs, indices))

    final2, _ = length_aware_sum(xs, 3)

    return {
        'masked_final': final,
        'masked_outputs': outputs,
        'length_aware_final': final2,
        'expected': jnp.sum(xs[:3])
    }


# =============================================================================
# Exercise 7: lax.scan Combined with vmap
# =============================================================================
def exercise_scan_vmap():
    """
    Batch multiple sequences with vmap over scan.

    First implement cumsum for single sequence, then vmap over batch.
    """
    # TODO: Implement cumsum for a single sequence
    def cumsum_single(xs):
        """Cumsum for a single sequence."""
        def scan_fn(carry, x):
            # TODO: Implement this function
            new_carry = None
            return new_carry, new_carry
        # TODO: Run scan
        cumsum = None
        return cumsum

    # Batch of sequences
    batch = jnp.array([
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [1, 1, 1, 1, 1],
    ], dtype=jnp.float32)

    # TODO: vmap over batch dimension
    batched_cumsum = None
    result = None

    # Verify
    expected = jnp.cumsum(batch, axis=1)

    # RNN example: batch of sequences
    def rnn_single(params, h0, inputs):
        def cell(h, x):
            return jnp.tanh(params['W'] @ h + params['U'] @ x), jnp.tanh(params['W'] @ h + params['U'] @ x)
        return lax.scan(cell, h0, inputs)

    hidden_dim, input_dim, seq_len, batch_size = 8, 4, 10, 3

    key = random.key(42)
    keys = random.split(key, 3)
    params = {
        'W': random.normal(keys[0], (hidden_dim, hidden_dim)) * 0.1,
        'U': random.normal(keys[1], (hidden_dim, input_dim)) * 0.1,
    }

    h0_batch = jnp.zeros((batch_size, hidden_dim))
    inputs_batch = random.normal(keys[2], (batch_size, seq_len, input_dim))

    # TODO: vmap over batch for RNN
    batched_rnn = None
    final_h_batch = None
    hidden_batch = None

    return {
        'cumsum_result': result,
        'expected_cumsum': expected,
        'cumsum_match': jnp.allclose(result, expected) if result is not None else False,
        'rnn_hidden_shape': hidden_batch.shape if hidden_batch is not None else None,
        'rnn_final_shape': final_h_batch.shape if final_h_batch is not None else None
    }


# =============================================================================
# Exercise 8: Unrolling with lax.scan
# =============================================================================
def exercise_unrolling():
    """
    Control loop unrolling with scan's unroll parameter.

    The unroll parameter tells JAX to process multiple iterations
    per compiled loop body, which can improve performance for short sequences.
    """
    def scan_fn(carry, x):
        return carry + x, carry + x

    xs = jnp.arange(16, dtype=jnp.float32)

    # TODO: Default scan (no unrolling)
    final1 = None

    # TODO: Unroll by 4
    final2 = None

    # TODO: Unroll by 8
    final4 = None

    # TODO: Full unroll (length of xs)
    final_full = None

    # All should give same result
    results_match = all([
        jnp.allclose(final1, final2),
        jnp.allclose(final2, final4),
        jnp.allclose(final4, final_full)
    ]) if all(x is not None for x in [final1, final2, final4, final_full]) else False

    return {
        'no_unroll': final1,
        'unroll_4': final2,
        'unroll_8': final4,
        'full_unroll': final_full,
        'all_match': results_match,
        'note': 'Unrolling can improve performance for short sequences'
    }


# =============================================================================
# Exercise 9: lax.scan for Time Series Processing
# =============================================================================
def exercise_time_series():
    """
    Time series operations: exponential moving average, etc.

    Implement EMA, differencing, and cumulative product using scan.
    """
    # TODO: Implement exponential moving average
    def ema(xs, alpha=0.1):
        """Exponential moving average using scan."""
        def scan_fn(ema_val, x):
            # TODO: Implement this function
            # new_ema = alpha * x + (1 - alpha) * ema_val
            new_ema = None
            return new_ema, new_ema

        # TODO: Run scan starting from xs[0]
        ema_values = None
        # Prepend the first value
        return jnp.concatenate([xs[:1], ema_values]) if ema_values is not None else None

    # Generate time series with noise
    key = random.key(42)
    t = jnp.linspace(0, 10, 100)
    signal = jnp.sin(t)
    noise = random.normal(key, (100,)) * 0.3
    noisy_signal = signal + noise

    # Compute EMA
    smoothed = ema(noisy_signal, alpha=0.2)

    # TODO: Implement ARIMA-style difference
    def difference(xs, order=1):
        """Compute differences using scan."""
        def diff_fn(prev, curr):
            # TODO: Implement this function
            return None, None
        # TODO: Run scan
        diffs = None
        return diffs

    diffs = difference(noisy_signal)

    # TODO: Implement cumulative product
    def cumulative_product(xs):
        def scan_fn(prod, x):
            # TODO: Implement this function
            new_prod = None
            return new_prod, new_prod
        # TODO: Run scan
        cumprod = None
        return cumprod

    returns = jnp.array([1.01, 0.99, 1.02, 1.01, 0.98])  # Daily returns
    prices = cumulative_product(returns)  # Price evolution from $1

    return {
        'original_shape': noisy_signal.shape,
        'smoothed_shape': smoothed.shape if smoothed is not None else None,
        'diffs_shape': diffs.shape if diffs is not None else None,
        'prices': prices,
        'final_price': prices[-1] if prices is not None else None
    }


# =============================================================================
# Exercise 10: Checkpointed Scan for Memory Efficiency
# =============================================================================
def exercise_checkpointed_scan():
    """
    Use gradient checkpointing with scan for memory efficiency.
    Recompute activations during backward pass instead of storing.
    """
    def heavy_computation(carry, x):
        """Simulates memory-intensive computation."""
        # In real case, this might be a transformer layer
        result = jnp.tanh(carry + x)
        return result, result

    xs = jnp.ones(100)

    # TODO: Implement normal scan forward pass
    def normal_forward(xs):
        # TODO: Run scan and return sum of outputs
        outputs = None
        return jnp.sum(outputs) if outputs is not None else None

    # TODO: Implement checkpointed version using @jax.checkpoint
    @jax.checkpoint
    def checkpointed_forward(xs):
        # TODO: Run scan and return sum of outputs
        outputs = None
        return jnp.sum(outputs) if outputs is not None else None

    # Both should give same gradients
    grad_normal = jax.grad(normal_forward)(xs) if normal_forward(xs) is not None else None
    grad_checkpointed = jax.grad(checkpointed_forward)(xs) if checkpointed_forward(xs) is not None else None

    # TODO: Implement custom chunked checkpointing strategy
    def scan_with_checkpoints(xs, checkpoint_every=10):
        """
        Manual checkpointing strategy.
        Process in chunks, checkpoint between chunks.
        """
        n = len(xs)
        num_chunks = (n + checkpoint_every - 1) // checkpoint_every

        def process_chunk(carry, chunk):
            # TODO: Implement chunk processing with scan
            final = None
            outputs = None
            return final, outputs

        # Reshape into chunks (pad if necessary)
        padded_len = num_chunks * checkpoint_every
        xs_padded = jnp.pad(xs, (0, padded_len - n), constant_values=0.0)
        chunks = xs_padded.reshape(num_chunks, checkpoint_every)

        # TODO: Scan over chunks
        all_outputs = None

        # Flatten and trim
        return all_outputs.flatten()[:n] if all_outputs is not None else None

    chunked_output = scan_with_checkpoints(xs)
    _, normal_output = lax.scan(heavy_computation, 0.0, xs)

    return {
        'gradients_match': jnp.allclose(grad_normal, grad_checkpointed) if grad_normal is not None and grad_checkpointed is not None else False,
        'chunked_matches_normal': jnp.allclose(chunked_output, normal_output) if chunked_output is not None else False,
        'memory_note': 'Checkpointing trades compute for memory'
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX lax.scan Exercises")
    print("=" * 60)

    exercises = [
        ("1. Basic Cumsum", exercise_basic_scan),
        ("2. Performance", exercise_performance),
        ("3. Carry State", exercise_carry_state),
        ("4. RNN Forward", exercise_rnn_scan),
        ("5. Reverse Scan", exercise_reverse_scan),
        ("6. Variable Length", exercise_variable_length),
        ("7. Scan + vmap", exercise_scan_vmap),
        ("8. Unrolling", exercise_unrolling),
        ("9. Time Series", exercise_time_series),
        ("10. Checkpointing", exercise_checkpointed_scan),
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
