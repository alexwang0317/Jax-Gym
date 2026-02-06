"""
JAX lax.scan - 10 Examples
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
# Example 1: Basic lax.scan for Cumulative Sum
# =============================================================================
def example_basic_scan():
    """
    Basic scan: carry state through a sequence.
    """
    def scan_fn(carry, x):
        """
        carry: running total
        x: current element
        returns: (new_carry, output)
        """
        new_carry = carry + x
        output = new_carry  # Output the running total
        return new_carry, output

    # Input sequence
    xs = jnp.array([1, 2, 3, 4, 5])

    # Run scan
    final_carry, cumsum = lax.scan(scan_fn, init=0, xs=xs)

    # cumsum = [1, 3, 6, 10, 15]
    # final_carry = 15

    # Compare with jnp.cumsum
    expected = jnp.cumsum(xs)

    return {
        'cumsum': cumsum,
        'final_carry': final_carry,
        'expected': expected,
        'matches': jnp.allclose(cumsum, expected)
    }


# =============================================================================
# Example 2: lax.scan vs Python Loops (Performance)
# =============================================================================
def example_performance():
    """
    Demonstrate performance advantage of scan over Python loops.
    """
    n = 1000

    # Python loop version (inside JIT - unrolled at compile time)
    @jax.jit
    def python_loop_sum(arr):
        total = 0.0
        for i in range(len(arr)):
            total = total + arr[i]
        return total

    # lax.fori_loop version
    @jax.jit
    def fori_loop_sum(arr):
        return lax.fori_loop(0, len(arr), lambda i, t: t + arr[i], 0.0)

    # lax.scan version
    @jax.jit
    def scan_sum(arr):
        final, _ = lax.scan(lambda c, x: (c + x, None), 0.0, arr)
        return final

    arr = jnp.ones(n)

    # Warm up
    _ = python_loop_sum(arr)
    _ = fori_loop_sum(arr)
    _ = scan_sum(arr)

    # Results should match
    result_python = python_loop_sum(arr)
    result_fori = fori_loop_sum(arr)
    result_scan = scan_sum(arr)

    return {
        'python_result': result_python,
        'fori_result': result_fori,
        'scan_result': result_scan,
        'all_match': jnp.allclose(result_python, result_fori) and jnp.allclose(result_fori, result_scan)
    }


# =============================================================================
# Example 3: Carrying State Through Scan
# =============================================================================
def example_carry_state():
    """
    Carry complex state (pytree) through scan.
    """
    def scan_fn(carry, x):
        """Track multiple running statistics."""
        count, sum_val, sum_sq = carry
        new_count = count + 1
        new_sum = sum_val + x
        new_sum_sq = sum_sq + x ** 2
        output = (new_sum / new_count, (new_sum_sq / new_count) - (new_sum / new_count) ** 2)
        return (new_count, new_sum, new_sum_sq), output

    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    init = (0, 0.0, 0.0)  # count, sum, sum_sq
    final_carry, outputs = lax.scan(scan_fn, init, xs)

    running_means, running_vars = outputs

    # Verify final values
    count, total, total_sq = final_carry
    final_mean = total / count
    final_var = total_sq / count - final_mean ** 2

    return {
        'final_count': count,
        'final_mean': final_mean,
        'expected_mean': jnp.mean(xs),
        'running_means': running_means,
        'running_vars': running_vars
    }


# =============================================================================
# Example 4: lax.scan for RNN Forward Pass
# =============================================================================
def example_rnn_scan():
    """
    Implement RNN forward pass using scan.
    Much more efficient than Python loops.
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
        def scan_fn(h, x):
            h_new = rnn_cell(params, h, x)
            return h_new, h_new  # carry and output are both h_new

        final_h, hidden_states = lax.scan(scan_fn, h0, inputs)
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
        'hidden_states_shape': hidden_states.shape,
        'final_h_shape': final_h.shape,
        'jit_matches': jnp.allclose(hidden_states, hidden_states_jit)
    }


# =============================================================================
# Example 5: Reverse Scan
# =============================================================================
def example_reverse_scan():
    """
    Scan in reverse order (for backward RNN, etc.).
    """
    def scan_fn(carry, x):
        return carry + x, carry + x

    xs = jnp.array([1, 2, 3, 4, 5])

    # Forward scan
    _, forward_cumsum = lax.scan(scan_fn, 0, xs)

    # Reverse scan
    _, reverse_cumsum = lax.scan(scan_fn, 0, xs, reverse=True)

    # Manually reverse for comparison
    _, manual_reverse = lax.scan(scan_fn, 0, xs[::-1])
    manual_reverse = manual_reverse[::-1]

    # Bidirectional pattern (forward + reverse)
    def bidirectional_scan(xs):
        _, fwd = lax.scan(scan_fn, 0, xs)
        _, bwd = lax.scan(scan_fn, 0, xs, reverse=True)
        return jnp.stack([fwd, bwd], axis=-1)

    combined = bidirectional_scan(xs)

    return {
        'forward': forward_cumsum,
        'reverse': reverse_cumsum,
        'manual_reverse': manual_reverse,
        'bidirectional_shape': combined.shape
    }


# =============================================================================
# Example 6: lax.scan with Variable-Length Sequences
# =============================================================================
def example_variable_length():
    """
    Handle variable-length sequences with masking.
    """
    def masked_scan_fn(carry, inputs):
        """Process only if mask is True."""
        x, mask = inputs
        # Update carry only if mask is True
        new_carry = jnp.where(mask, carry + x, carry)
        return new_carry, new_carry

    # Padded sequence with mask
    xs = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0])  # Actual length is 3
    mask = jnp.array([True, True, True, False, False])

    final, outputs = lax.scan(masked_scan_fn, 0.0, (xs, mask))

    # Alternative: use jnp.where inside scan
    def length_aware_sum(xs, length):
        def scan_fn(carry, inputs):
            x, idx = inputs
            # Only add if idx < length
            new_carry = jnp.where(idx < length, carry + x, carry)
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
# Example 7: lax.scan Combined with vmap
# =============================================================================
def example_scan_vmap():
    """
    Batch multiple sequences with vmap over scan.
    """
    def cumsum_single(xs):
        """Cumsum for a single sequence."""
        def scan_fn(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        _, cumsum = lax.scan(scan_fn, 0.0, xs)
        return cumsum

    # Batch of sequences
    batch = jnp.array([
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [1, 1, 1, 1, 1],
    ], dtype=jnp.float32)

    # vmap over batch dimension
    batched_cumsum = jax.vmap(cumsum_single)
    result = batched_cumsum(batch)

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

    # vmap over batch
    batched_rnn = jax.vmap(lambda h, x: rnn_single(params, h, x))
    final_h_batch, hidden_batch = batched_rnn(h0_batch, inputs_batch)

    return {
        'cumsum_result': result,
        'expected_cumsum': expected,
        'cumsum_match': jnp.allclose(result, expected),
        'rnn_hidden_shape': hidden_batch.shape,
        'rnn_final_shape': final_h_batch.shape
    }


# =============================================================================
# Example 8: Unrolling with lax.scan
# =============================================================================
def example_unrolling():
    """
    Control loop unrolling with scan's unroll parameter.
    """
    def scan_fn(carry, x):
        return carry + x, carry + x

    xs = jnp.arange(16, dtype=jnp.float32)

    # Default: no unrolling
    final1, _ = lax.scan(scan_fn, 0.0, xs)

    # Unroll by 4: processes 4 iterations per compiled loop body
    final2, _ = lax.scan(scan_fn, 0.0, xs, unroll=4)

    # Unroll by 8
    final4, _ = lax.scan(scan_fn, 0.0, xs, unroll=8)

    # Full unroll (length of xs) - equivalent to Python loop unrolling
    final_full, _ = lax.scan(scan_fn, 0.0, xs, unroll=len(xs))

    # All should give same result
    results_match = all([
        jnp.allclose(final1, final2),
        jnp.allclose(final2, final4),
        jnp.allclose(final4, final_full)
    ])

    return {
        'no_unroll': final1,
        'unroll_4': final2,
        'unroll_8': final4,
        'full_unroll': final_full,
        'all_match': results_match,
        'note': 'Unrolling can improve performance for short sequences'
    }


# =============================================================================
# Example 9: lax.scan for Time Series Processing
# =============================================================================
def example_time_series():
    """
    Time series operations: exponential moving average, etc.
    """
    def ema(xs, alpha=0.1):
        """Exponential moving average using scan."""
        def scan_fn(ema_val, x):
            new_ema = alpha * x + (1 - alpha) * ema_val
            return new_ema, new_ema

        _, ema_values = lax.scan(scan_fn, xs[0], xs[1:])
        # Prepend the first value
        return jnp.concatenate([xs[:1], ema_values])

    # Generate time series with noise
    key = random.key(42)
    t = jnp.linspace(0, 10, 100)
    signal = jnp.sin(t)
    noise = random.normal(key, (100,)) * 0.3
    noisy_signal = signal + noise

    # Compute EMA
    smoothed = ema(noisy_signal, alpha=0.2)

    # ARIMA-style difference
    def difference(xs, order=1):
        """Compute differences using scan."""
        def diff_fn(prev, curr):
            return curr, curr - prev
        _, diffs = lax.scan(diff_fn, xs[0], xs[1:])
        return diffs

    diffs = difference(noisy_signal)

    # Cumulative product (for returns -> prices)
    def cumulative_product(xs):
        def scan_fn(prod, x):
            new_prod = prod * x
            return new_prod, new_prod
        _, cumprod = lax.scan(scan_fn, 1.0, xs)
        return cumprod

    returns = jnp.array([1.01, 0.99, 1.02, 1.01, 0.98])  # Daily returns
    prices = cumulative_product(returns)  # Price evolution from $1

    return {
        'original_shape': noisy_signal.shape,
        'smoothed_shape': smoothed.shape,
        'diffs_shape': diffs.shape,
        'prices': prices,
        'final_price': prices[-1]
    }


# =============================================================================
# Example 10: Checkpointed Scan for Memory Efficiency
# =============================================================================
def example_checkpointed_scan():
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

    # Normal scan (stores all intermediate activations)
    def normal_forward(xs):
        _, outputs = lax.scan(heavy_computation, 0.0, xs)
        return jnp.sum(outputs)

    # Checkpointed version (recomputes during backward)
    @jax.checkpoint
    def checkpointed_forward(xs):
        _, outputs = lax.scan(heavy_computation, 0.0, xs)
        return jnp.sum(outputs)

    # Both should give same gradients
    grad_normal = jax.grad(normal_forward)(xs)
    grad_checkpointed = jax.grad(checkpointed_forward)(xs)

    # Custom policy: checkpoint every N steps
    def scan_with_checkpoints(xs, checkpoint_every=10):
        """
        Manual checkpointing strategy.
        Process in chunks, checkpoint between chunks.
        """
        n = len(xs)
        num_chunks = (n + checkpoint_every - 1) // checkpoint_every

        def process_chunk(carry, chunk):
            final, outputs = lax.scan(heavy_computation, carry, chunk)
            return final, outputs

        # Reshape into chunks (pad if necessary)
        padded_len = num_chunks * checkpoint_every
        xs_padded = jnp.pad(xs, (0, padded_len - n), constant_values=0.0)
        chunks = xs_padded.reshape(num_chunks, checkpoint_every)

        # Scan over chunks (outer loop can be checkpointed)
        final, all_outputs = lax.scan(process_chunk, 0.0, chunks)

        # Flatten and trim
        return all_outputs.flatten()[:n]

    chunked_output = scan_with_checkpoints(xs)
    _, normal_output = lax.scan(heavy_computation, 0.0, xs)

    return {
        'gradients_match': jnp.allclose(grad_normal, grad_checkpointed),
        'chunked_matches_normal': jnp.allclose(chunked_output, normal_output),
        'memory_note': 'Checkpointing trades compute for memory'
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX lax.scan Examples")
    print("=" * 60)

    examples = [
        ("1. Basic Cumsum", example_basic_scan),
        ("2. Performance", example_performance),
        ("3. Carry State", example_carry_state),
        ("4. RNN Forward", example_rnn_scan),
        ("5. Reverse Scan", example_reverse_scan),
        ("6. Variable Length", example_variable_length),
        ("7. Scan + vmap", example_scan_vmap),
        ("8. Unrolling", example_unrolling),
        ("9. Time Series", example_time_series),
        ("10. Checkpointing", example_checkpointed_scan),
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
