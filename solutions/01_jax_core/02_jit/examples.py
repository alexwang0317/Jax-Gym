"""
JAX JIT (Just-In-Time Compilation) - 10 Examples
=================================================

JIT compilation is the primary way to make JAX fast.
It compiles Python functions to optimized XLA code.

Key concepts:
- First call traces the function and compiles it
- Subsequent calls use the cached compiled version
- Functions must be pure (no side effects)
- Shape/dtype changes trigger recompilation

Reference: https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
"""

import jax
import jax.numpy as jnp
from jax import jit
import time
from functools import partial


# =============================================================================
# Example 1: Basic @jit Decorator Usage
# =============================================================================
def example_basic_jit():
    """
    The simplest way to use JIT: decorate your function.
    The first call compiles; subsequent calls are fast.
    """
    # Without JIT
    def slow_fn(x):
        return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

    # With JIT
    @jit
    def fast_fn(x):
        return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

    x = jnp.linspace(0, 10, 1000)

    # Both produce the same result
    result_slow = slow_fn(x)
    result_fast = fast_fn(x)

    # Can also use jit as a function
    also_fast = jit(slow_fn)
    result_also = also_fast(x)

    return {
        'slow_result': result_slow,
        'fast_result': result_fast,
        'results_match': jnp.allclose(result_slow, result_fast)
    }


# =============================================================================
# Example 2: Timing Comparison - JIT vs Non-JIT
# =============================================================================
def example_timing_comparison():
    """
    Demonstrate the speedup from JIT compilation.
    First call includes compilation time; subsequent calls are fast.
    """
    def matmul_chain(x):
        """Chain of matrix multiplications - benefits greatly from JIT."""
        for _ in range(10):
            x = x @ x.T @ x
        return x

    fast_matmul = jit(matmul_chain)

    x = jnp.ones((100, 100))

    # Warm up JIT (first call compiles)
    _ = fast_matmul(x).block_until_ready()

    # Time without JIT
    start = time.perf_counter()
    for _ in range(10):
        result_slow = matmul_chain(x).block_until_ready()
    slow_time = time.perf_counter() - start

    # Time with JIT (already compiled)
    start = time.perf_counter()
    for _ in range(10):
        result_fast = fast_matmul(x).block_until_ready()
    fast_time = time.perf_counter() - start

    speedup = slow_time / fast_time if fast_time > 0 else float('inf')

    return {
        'slow_time': slow_time,
        'fast_time': fast_time,
        'speedup': speedup,
        'results_match': jnp.allclose(result_slow, result_fast)
    }


# =============================================================================
# Example 3: static_argnums for Shape-Dependent Code
# =============================================================================
def example_static_argnums():
    """
    Use static_argnums when argument values affect the computation graph shape.
    These arguments are treated as compile-time constants.

    Common use cases:
    - Array sizes/shapes
    - Number of iterations
    - Boolean flags
    """
    @partial(jit, static_argnums=(1,))  # Second argument (n) is static
    def repeat_sum(x, n):
        """Sum x with itself n times. n affects the graph structure."""
        result = x
        for _ in range(n):
            result = result + x
        return result

    x = jnp.array([1.0, 2.0, 3.0])

    # Different n values cause recompilation
    result_3 = repeat_sum(x, 3)   # Compiles for n=3
    result_5 = repeat_sum(x, 5)   # Compiles for n=5
    result_3_again = repeat_sum(x, 3)  # Uses cached n=3 version

    # static_argnames alternative (more readable)
    @partial(jit, static_argnames=['axis'])
    def dynamic_reduce(x, axis):
        return jnp.sum(x, axis=axis)

    matrix = jnp.ones((3, 4))
    sum_rows = dynamic_reduce(matrix, axis=0)
    sum_cols = dynamic_reduce(matrix, axis=1)

    return {
        'repeat_3': result_3,
        'repeat_5': result_5,
        'sum_rows': sum_rows,
        'sum_cols': sum_cols
    }


# =============================================================================
# Example 4: donate_argnums for Memory Efficiency
# =============================================================================
def example_donate_argnums():
    """
    donate_argnums tells JAX that an input buffer can be reused for output.
    This avoids memory allocation when updating arrays in-place conceptually.

    Use when:
    - You won't need the input after the function call
    - Input and output have the same shape/dtype
    """
    @partial(jit, donate_argnums=(0,))  # Donate first argument
    def update_in_place(x, y):
        """Add y to x. The x buffer can be reused."""
        return x + y

    # Note: After calling with donate, the input becomes invalid
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000)) * 2

    # This is memory efficient - x's buffer is reused
    result = update_in_place(x, y)
    # x is now invalid! Don't use it.

    # Common pattern: state updates in training loops
    @partial(jit, donate_argnums=(0,))
    def update_params(params, grads, lr):
        return params - lr * grads

    params = jnp.ones(1000)
    grads = jnp.ones(1000) * 0.1
    new_params = update_params(params, grads, 0.01)

    return {
        'result_shape': result.shape,
        'new_params_shape': new_params.shape,
        'memory_efficient': True
    }


# =============================================================================
# Example 5: JIT Caching and Recompilation Behavior
# =============================================================================
def example_jit_caching():
    """
    JIT caches compiled functions based on input shapes and dtypes.
    Changes in shape or dtype trigger recompilation.
    """
    compile_count = [0]  # Using list to track across calls

    @jit
    def traced_fn(x):
        # This runs during tracing (compilation)
        compile_count[0] += 1
        return x * 2

    # First call with shape (3,) - compiles
    _ = traced_fn(jnp.array([1, 2, 3]))
    count_after_first = compile_count[0]

    # Same shape - uses cache
    _ = traced_fn(jnp.array([4, 5, 6]))
    count_after_same = compile_count[0]

    # Different shape - recompiles!
    _ = traced_fn(jnp.array([1, 2, 3, 4]))
    count_after_different = compile_count[0]

    # Different dtype - recompiles!
    _ = traced_fn(jnp.array([1.0, 2.0, 3.0]))
    count_after_dtype = compile_count[0]

    # Clear caches if needed (rarely necessary)
    traced_fn.clear_cache()

    return {
        'compiles_for_first': count_after_first,
        'compiles_after_same_shape': count_after_same,
        'compiles_after_diff_shape': count_after_different,
        'compiles_after_diff_dtype': count_after_dtype
    }


# =============================================================================
# Example 6: Debugging with jax.disable_jit()
# =============================================================================
def example_debugging():
    """
    jax.disable_jit() context manager disables JIT for debugging.
    Useful when you need to:
    - Print intermediate values
    - Use Python debugger (pdb)
    - Understand control flow
    """
    @jit
    def complex_fn(x):
        a = x * 2
        b = jnp.sin(a)
        c = b + 1
        return c

    x = jnp.array([1.0, 2.0, 3.0])

    # Normal execution - JIT compiled
    result_jitted = complex_fn(x)

    # Debugging mode - no JIT
    with jax.disable_jit():
        # Now you can add print statements, use debugger, etc.
        result_debug = complex_fn(x)

    # Check JIT is re-enabled after context
    result_after = complex_fn(x)

    return {
        'jitted_result': result_jitted,
        'debug_result': result_debug,
        'results_match': jnp.allclose(result_jitted, result_debug)
    }


# =============================================================================
# Example 7: Visualizing Computation Graphs with make_jaxpr
# =============================================================================
def example_make_jaxpr():
    """
    jax.make_jaxpr() shows the intermediate representation (jaxpr)
    that JAX compiles. Useful for understanding what JIT sees.

    jaxpr = JAX Program Representation
    """
    def simple_fn(x, y):
        return jnp.sin(x) + y * 2

    # Get the jaxpr for specific input shapes
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    jaxpr = jax.make_jaxpr(simple_fn)(x, y)
    jaxpr_str = str(jaxpr)

    # More complex example
    def matmul_relu(W, x, b):
        return jnp.maximum(0, W @ x + b)

    W = jnp.ones((3, 4))
    x_vec = jnp.ones(4)
    b = jnp.zeros(3)

    jaxpr_mlp = jax.make_jaxpr(matmul_relu)(W, x_vec, b)

    return {
        'simple_jaxpr': jaxpr_str[:200] + '...',  # Truncate for display
        'mlp_jaxpr_lines': len(str(jaxpr_mlp).split('\n'))
    }


# =============================================================================
# Example 8: JIT and Side Effects (What NOT to Do)
# =============================================================================
def example_side_effects():
    """
    JIT traces functions, so side effects only happen during tracing.
    This is a common source of bugs!

    Side effects to avoid:
    - print() statements
    - Modifying global variables
    - I/O operations
    - Random number generation without explicit keys
    """
    trace_count = [0]

    @jit
    def bad_example(x):
        # This print only happens during tracing!
        trace_count[0] += 1
        # print("This only prints once!")  # Uncomment to see
        return x + 1

    # Call multiple times
    _ = bad_example(jnp.array([1.0]))  # Traces and "prints"
    _ = bad_example(jnp.array([2.0]))  # Uses cache - no print!
    _ = bad_example(jnp.array([3.0]))  # Uses cache - no print!

    # For debugging, use jax.debug.print (works inside JIT)
    @jit
    def good_debug(x):
        y = x * 2
        jax.debug.print("y = {}", y)  # This prints every call
        return y + 1

    # Note: jax.debug.print has overhead, use sparingly

    return {
        'trace_count': trace_count[0],
        'expected_traces': 1,
        'explanation': "Function traced once, cached for same shape"
    }


# =============================================================================
# Example 9: Pure Functions Requirement for JIT
# =============================================================================
def example_pure_functions():
    """
    JIT requires pure functions: same inputs -> same outputs.
    Functions must not depend on or modify external state.

    Pure function rules:
    1. No global variable reads (that change)
    2. No global variable writes
    3. No I/O
    4. No random without explicit keys
    """
    # BAD: Depends on global state
    global_counter = [0]

    def impure_fn(x):
        global_counter[0] += 1  # Side effect!
        return x + global_counter[0]

    # This will give wrong results with JIT because
    # the counter increment is traced once

    # GOOD: All state is explicit
    def pure_fn(x, counter):
        new_counter = counter + 1
        return x + new_counter, new_counter

    x = jnp.array(1.0)
    counter = 0

    # Explicit state threading
    result1, counter = pure_fn(x, counter)
    result2, counter = pure_fn(x, counter)
    result3, counter = pure_fn(x, counter)

    # This can be safely JIT compiled
    pure_fn_jit = jit(pure_fn)
    result_jit, _ = pure_fn_jit(x, 0)

    return {
        'pure_results': [float(result1), float(result2), float(result3)],
        'jit_result': float(result_jit),
        'final_counter': counter
    }


# =============================================================================
# Example 10: Dynamic Shapes and Recompilation Costs
# =============================================================================
def example_dynamic_shapes():
    """
    Dynamic shapes cause recompilation, which can be expensive.
    Strategies to handle this:
    1. Pad to fixed sizes
    2. Use static_argnums for dimensions
    3. Accept the recompilation cost for rare shapes
    """
    @jit
    def process_batch(x):
        return jnp.mean(x, axis=0)

    # Different batch sizes cause recompilation
    batch_16 = jnp.ones((16, 10))
    batch_32 = jnp.ones((32, 10))
    batch_64 = jnp.ones((64, 10))

    # Each new batch size triggers compilation
    _ = process_batch(batch_16)  # Compiles for (16, 10)
    _ = process_batch(batch_32)  # Compiles for (32, 10)
    _ = process_batch(batch_64)  # Compiles for (64, 10)

    # Strategy 1: Pad to fixed size
    def pad_to_size(x, target_size):
        current_size = x.shape[0]
        if current_size < target_size:
            padding = jnp.zeros((target_size - current_size,) + x.shape[1:])
            return jnp.concatenate([x, padding], axis=0)
        return x

    # Strategy 2: Use masking
    @jit
    def masked_mean(x, mask):
        """Compute mean only over valid elements."""
        masked = x * mask[:, None]
        return jnp.sum(masked, axis=0) / jnp.sum(mask)

    # Fixed size with variable valid elements
    x_padded = jnp.ones((64, 10))
    mask = jnp.array([1.0] * 32 + [0.0] * 32)  # Only first 32 valid
    result_masked = masked_mean(x_padded, mask)

    return {
        'padded_shape': x_padded.shape,
        'masked_mean_shape': result_masked.shape,
        'strategy': 'Pad to fixed size and use masks to handle variable sizes'
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX JIT Examples")
    print("=" * 60)

    examples = [
        ("1. Basic JIT", example_basic_jit),
        ("2. Timing Comparison", example_timing_comparison),
        ("3. static_argnums", example_static_argnums),
        ("4. donate_argnums", example_donate_argnums),
        ("5. JIT Caching", example_jit_caching),
        ("6. Debugging", example_debugging),
        ("7. make_jaxpr", example_make_jaxpr),
        ("8. Side Effects", example_side_effects),
        ("9. Pure Functions", example_pure_functions),
        ("10. Dynamic Shapes", example_dynamic_shapes),
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
