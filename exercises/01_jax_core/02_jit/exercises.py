"""
JAX JIT (Just-In-Time Compilation) - 10 Exercises
==================================================

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
# Exercise 1: Basic @jit Decorator Usage
# =============================================================================
def exercise_basic_jit():
    """
    The simplest way to use JIT: decorate your function.
    The first call compiles; subsequent calls are fast.

    Your task:
    1. Create a function `slow_fn(x)` that computes sin(x)^2 + cos(x)^2 (without JIT)
    2. Create a function `fast_fn(x)` that does the same but uses the @jit decorator
    3. Create an input array x using jnp.linspace(0, 10, 1000)
    4. Call both functions and verify they produce the same results

    Hints:
    - Use @jit decorator on fast_fn
    - Use jnp.sin() and jnp.cos() for trig functions
    - Use jnp.allclose() to compare results
    - You can also use jit() as a function: jit(slow_fn)
    """
    # TODO: Implement this function

    # Without JIT
    def slow_fn(x):
        # TODO: return sin(x)^2 + cos(x)^2
        return jnp.sin(x)**2 + jnp.cos(x)**2

    # With JIT - add the @jit decorator
    @jit
    def fast_fn(x):
        # TODO: return sin(x)^2 + cos(x)^2
        return jnp.sin(x)**2 + jnp.cos(x)**2

    # TODO: Create input array x with jnp.linspace(0, 10, 1000)
    x = jnp.linspace(0, 10, 1000)

    # TODO: Call both functions
    result_slow = slow_fn(x)
    result_fast = fast_fn(x)

    # TODO: Check if results match using jnp.allclose
    results_match = jnp.allclose(result_slow, result_fast)

    return {
        'slow_result': result_slow,
        'fast_result': result_fast,
        'results_match': results_match
    }


# =============================================================================
# Exercise 2: Timing Comparison - JIT vs Non-JIT
# =============================================================================
def exercise_timing_comparison():
    """
    Demonstrate the speedup from JIT compilation.
    First call includes compilation time; subsequent calls are fast.

    Your task:
    1. Create a function matmul_chain(x) that does 10 iterations of x @ x.T @ x
    2. Create a JIT-compiled version using jit()
    3. Warm up the JIT version (first call compiles)
    4. Time both versions over 10 iterations
    5. Calculate the speedup ratio

    Hints:
    - Use .block_until_ready() to ensure computation completes before timing
    - Use time.perf_counter() for timing
    - x.T gives the transpose of x
    - @ is matrix multiplication
    - speedup = slow_time / fast_time
    """
    # TODO: Implement this function

    def matmul_chain(x):
        """Chain of matrix multiplications - benefits greatly from JIT."""
        # TODO: Do 10 iterations of x = x @ x.T @ x

        for i in range(10):
            x = x @ x.T @ x
        return x

    # TODO: Create JIT version
    fast_matmul = jit(matmul_chain)

    # TODO: Create input matrix x of shape (100, 100) filled with ones
    x = jnp.ones((100, 100))

    # TODO: Warm up JIT (call once and use .block_until_ready())
    fast_matmul(x).block_until_ready()

    # TODO: Time without JIT (10 iterations)
    start_slow = time.perf_counter()
    y_1 = matmul_chain(x)
    y_1.block_until_ready()
    slow_time = time.perf_counter() - start_slow

    # TODO: Time with JIT (10 iterations)
    start_fast = time.perf_counter()
    y_2 = fast_matmul(x)
    y_2.block_until_ready()
    fast_time = time.perf_counter() - start_fast


    # TODO: Calculate speedup
    speedup = slow_time / fast_time

    # TODO: Check results match
    results_match =  jnp.allclose(y_1, y_2)

    return {
        'slow_time': slow_time,
        'fast_time': fast_time,
        'speedup': speedup,
        'results_match': results_match
    }


# =============================================================================
# Exercise 3: static_argnums for Shape-Dependent Code
# =============================================================================
def exercise_static_argnums():
    """
    Use static_argnums when argument values affect the computation graph shape.
    These arguments are treated as compile-time constants.

    Your task:
    1. Create repeat_sum(x, n) that adds x to itself n times (x + x + x + ...)
       Use @partial(jit, static_argnums=(1,)) to mark n as static
    2. Create dynamic_reduce(x, axis) using static_argnames=['axis']
       that sums x along the given axis

    Hints:
    - static_argnums=(1,) marks the second argument (index 1) as static
    - static_argnames=['axis'] is more readable for keyword arguments
    - For repeat_sum: start with result = x, then loop n times adding x
    - Different static argument values cause recompilation
    """
    # TODO: Implement this function

    # TODO: Use @partial(jit, static_argnums=(1,))   if it was (1,2,3), then the 2nd through 4th argument would be static instead, essentially a list that is read. 
    @partial(jit, static_argnums=(1,))
    def repeat_sum(x, n):
        """Sum x with itself n times. n affects the graph structure."""
        # TODO: Implement - start with result = x, loop n times adding x
        total = x
        for i in range(n):
            total += x 
        return total

    # TODO: Create input array x = [1.0, 2.0, 3.0]
    x = jnp.array([1.0, 2.0, 3.0])

    # TODO: Call repeat_sum with n=3 and n=5
    result_3 = repeat_sum(x, n=3)
    result_5 = repeat_sum(x, n=5)

    # TODO: Use @partial(jit, static_argnames=['axis'])
    @partial(jit, static_argnames=['axis'])
    def dynamic_reduce(x, axis):
        # TODO: Return jnp.sum(x, axis=axis)
        return jnp.sum(x, axis=axis)

    # TODO: Create matrix of shape (3, 4) filled with ones
    matrix = jnp.ones((3,4))

    # TODO: Sum along axis 0 (sum rows) and axis 1 (sum columns)
    sum_rows = dynamic_reduce(matrix, axis=0)
    sum_cols = dynamic_reduce(matrix, axis=1)

    return {
        'repeat_3': result_3,
        'repeat_5': result_5,
        'sum_rows': sum_rows,
        'sum_cols': sum_cols
    }


# =============================================================================
# Exercise 4: donate_argnums for Memory Efficiency
# =============================================================================
def exercise_donate_argnums():
    """
    donate_argnums tells JAX that an input buffer can be reused for output.
    This avoids memory allocation when updating arrays in-place conceptually.

    Your task:
    1. Create update_in_place(x, y) that returns x + y
       Use @partial(jit, donate_argnums=(0,)) to donate first argument
    2. Create update_params(params, grads, lr) that returns params - lr * grads
       Also use donate_argnums=(0,)

    Hints:
    - donate_argnums=(0,) means the first argument's buffer can be reused
    - After calling with donated args, the input becomes invalid
    - This is useful for training loops where you update parameters
    - Only works when input and output have same shape/dtype
    """
    # TODO: Implement this function

    # TODO: Use @partial(jit, donate_argnums=(0,))
    @partial(jit, donate_argnums=(0,))
    def update_in_place(x, y):
        """Add y to x. The x buffer can be reused."""
        # TODO: return x + y
        return x + y

    # TODO: Create x and y arrays of shape (1000, 1000)
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))

    # TODO: Call update_in_place
    result = update_in_place(x, y)

    # TODO: Use @partial(jit, donate_argnums=(0,))
    @partial(jit, donate_argnums=(0,))
    def update_params(params, grads, lr):
        # TODO: return params - lr * grads
        return params - lr * grads

    # TODO: Create params and grads arrays of shape (1000,)
    params = jnp.ones((1000,))
    grads = jnp.ones((1000,))

    # TODO: Update params with lr=0.01
    new_params = update_params(params, grads, 0.01)

    return {
        'result_shape': result.shape if result is not None else None,
        'new_params_shape': new_params.shape if new_params is not None else None,
        'memory_efficient': True
    }


# =============================================================================
# Exercise 5: JIT Caching and Recompilation Behavior
# =============================================================================
def exercise_jit_caching():
    """
    JIT caches compiled functions based on input shapes and dtypes.
    Changes in shape or dtype trigger recompilation.

    Your task:
    1. Create a traced_fn that multiplies x by 2
    2. Use a list compile_count = [0] to track compilations
    3. Increment compile_count[0] inside the function (this happens at trace time)
    4. Call with different shapes and dtypes to see recompilation behavior

    Hints:
    - Side effects (like incrementing a counter) only happen during tracing
    - Same shape reuses the cached compilation
    - Different shape triggers recompilation
    - Different dtype also triggers recompilation
    - compile_count[0] will show how many times the function was traced
    """
    # TODO: Implement this function

    compile_count = [0]  # Using list to track across calls

    @jit
    def traced_fn(x):
        # TODO: Increment compile_count[0] (this runs during tracing)
        # TODO: Return x * 2
        compile_count[0] += 1
        return x * 2

    # TODO: First call with shape (3,) integer array [1, 2, 3]
    _ = traced_fn(jnp.array([1, 2, 3]))
    count_after_first = compile_count[0]

    # TODO: Same shape - should use cache
    _ = traced_fn(jnp.array([4, 5, 6]))
    count_after_same = compile_count[0]

    # TODO: Different shape (4,) - should recompile
    _ = traced_fn(jnp.array([1, 2, 3, 4]))
    count_after_different = compile_count[0]

    # TODO: Different dtype (float) - should recompile
    _ = traced_fn(jnp.array([1.0, 2.0, 3.0]))
    count_after_dtype = compile_count[0]

    return {
        'compiles_for_first': count_after_first,
        'compiles_after_same_shape': count_after_same,
        'compiles_after_diff_shape': count_after_different,
        'compiles_after_diff_dtype': count_after_dtype
    }


# =============================================================================
# Exercise 6: Debugging with jax.disable_jit()
# =============================================================================
def exercise_debugging():
    """
    jax.disable_jit() context manager disables JIT for debugging.
    Useful when you need to:
    - Print intermediate values
    - Use Python debugger (pdb)
    - Understand control flow

    Your task:
    1. Create a JIT-compiled function that does: x * 2 -> sin -> add 1
    2. Call it normally (JIT enabled)
    3. Call it inside jax.disable_jit() context
    4. Verify both produce the same result

    Hints:
    - Use the @jit decorator
    - Use `with jax.disable_jit():` context manager
    - Inside the context, JIT is disabled so you can debug
    - Results should match whether JIT is enabled or disabled
    """
    # TODO: Implement this function

    # Track what happens during tracing vs execution
    trace_log = []

    @jit
    def complex_fn(x):
        a = x * 2
        b = jnp.sin(a)

        # Regular print() only runs during TRACING, not execution!
        # When JIT traces, 'a' is a Tracer object, not real values
        print(f"  [print] This runs during tracing. a = {a}")
        print(f"  [print] Type of a: {type(a).__name__}")

        # Side effects like appending to lists also only happen at trace time
        trace_log.append("traced!")

        c = b + 1
        return c

    x = jnp.array([1.0, 2.0, 3.0])

    # === FIRST: Normal JIT execution ===
    print("\n>>> Calling complex_fn(x) WITH JIT (1st call - traces):")
    result_jitted = complex_fn(x)
    print(f"  Result: {result_jitted}")
    print(f"  trace_log after 1st call: {trace_log}")

    print("\n>>> Calling complex_fn(x) WITH JIT (2nd call - cached, no tracing):")
    result_jitted_2 = complex_fn(x)
    print(f"  Result: {result_jitted_2}")
    print(f"  trace_log after 2nd call: {trace_log}")  # Still just 1 entry!

    # === NOW: With disable_jit() ===
    print("\n>>> Calling complex_fn(x) with jax.disable_jit():")
    with jax.disable_jit():
        result_debug = complex_fn(x)
    print(f"  Result: {result_debug}")
    print(f"  trace_log after disable_jit call: {trace_log}")  # Now 2 entries!

    results_match = jnp.allclose(result_jitted, result_debug)

    print("\n>>> KEY INSIGHT:")
    print("  - With JIT: print() runs once during tracing, shows Tracer objects")
    print("  - With disable_jit(): print() runs every call, shows real values")
    print("  - Use jax.debug.print() if you want prints during JIT execution")

    return {
        'jitted_result': result_jitted,
        'debug_result': result_debug,
        'results_match': results_match,
        'trace_count': len(trace_log),  # Shows tracing happened twice total
        'explanation': 'JIT traced once (cached), disable_jit ran Python again'
    }


# =============================================================================
# Exercise 7: Visualizing Computation Graphs with make_jaxpr
# =============================================================================
def exercise_make_jaxpr():
    """
    jax.make_jaxpr() shows the intermediate representation (jaxpr)
    that JAX compiles. Useful for understanding what JIT sees.

    jaxpr = JAX Program Representation

    Your task:
    1. Create simple_fn(x, y) that returns sin(x) + y * 2
    2. Use jax.make_jaxpr to visualize the computation graph
    3. Create matmul_relu(W, x, b) that returns max(0, W @ x + b)
    4. Visualize its jaxpr too

    Hints:
    - jax.make_jaxpr(fn)(args...) returns a jaxpr object
    - str(jaxpr) gives a readable string representation
    - The jaxpr shows the sequence of operations JAX will execute
    - jnp.maximum(0, x) implements ReLU activation
    """
    # TODO: Implement this function

    def simple_fn(x, y):
        # TODO: return jnp.sin(x) + y * 2
        return jnp.sin(x) + y * 2

    # TODO: Create input arrays x and y of shape (2,)
    x = jnp.ones((2,))
    y = jnp.ones((2,))

    # TODO: Get jaxpr using jax.make_jaxpr(simple_fn)(x, y)
    jaxpr = jax.make_jaxpr(simple_fn)(x, y)
    jaxpr_str = str(jaxpr)

    def matmul_relu(W, x, b):
        # TODO: return jnp.maximum(0, W @ x + b)
        return jnp.maximum(0, W @ x + b)

    # TODO: Create W of shape (3, 4), x_vec of shape (4,), b of shape (3,)
    W = jnp.ones((3,4))
    x_vec = jnp.ones((4,))
    b = jnp.ones((3,))

    # TODO: Get jaxpr for matmul_relu
    jaxpr_mlp = jax.make_jaxpr(matmul_relu)(W, x_vec, b)

    return {
        'simple_jaxpr': jaxpr_str[:200] + '...' if jaxpr_str else None,
        'mlp_jaxpr_lines': len(str(jaxpr_mlp).split('\n')) if jaxpr_mlp else None
    }


# =============================================================================
# Exercise 8: JIT and Side Effects (What NOT to Do)
# =============================================================================
def exercise_side_effects():
    """
    JIT traces functions, so side effects only happen during tracing.
    This is a common source of bugs!

    Side effects to avoid:
    - print() statements
    - Modifying global variables
    - I/O operations
    - Random number generation without explicit keys

    Your task:
    1. Create a function that increments trace_count[0] and returns x + 1
    2. Call it 3 times with same-shaped arrays
    3. Observe that trace_count only increments once (at trace time)

    Hints:
    - Side effects in JIT functions only run during tracing
    - Same shape/dtype inputs reuse the cached trace
    - For real debugging, use jax.debug.print() instead of print()
    - The counter should equal 1 after all calls (traced once)
    """
    # TODO: Implement this function

    trace_count = [0]

    @jit
    def bad_example(x):
        # TODO: Increment trace_count[0] (this only happens during tracing!)
        # TODO: Return x + 1

        trace_count[0] += 1
        return x + 1

    # TODO: Call 3 times with same shape arrays
    _ = bad_example(jnp.array([1.0]))
    _ = bad_example(jnp.array([2.0]))
    _ = bad_example(jnp.array([3.0]))
    return {
        'trace_count': trace_count[0],
        'expected_traces': 1,
        'explanation': "Function traced once, cached for same shape"
    }


# =============================================================================
# Exercise 9: Pure Functions Requirement for JIT
# =============================================================================
def exercise_pure_functions():
    """
    JIT requires pure functions: same inputs -> same outputs.
    Functions must not depend on or modify external state.

    Your task:
    1. Create a pure_fn(x, counter) that returns (x + counter + 1, counter + 1)
       This passes state explicitly instead of using globals
    2. Call it 3 times, threading the counter through
    3. JIT compile it and verify it works

    Hints:
    - Pure functions have no side effects
    - All state must be passed in and returned explicitly
    - This pattern is called "state threading"
    - pure_fn(1.0, 0) should return (2.0, 1)
    - pure_fn(1.0, 1) should return (3.0, 2)
    """
    # TODO: Implement this function
    
    # @partial(jit, donate_argnums=(1,))
    def pure_fn(x, counter):
        # TODO: new_counter = counter + 1
        # TODO: return x + new_counter, new_counter
        new_counter = counter + 1
        return x + new_counter, new_counter

    # TODO: Create x = 1.0
    x = 1.0

    # TODO: Initialize counter = 0
    counter = 0

    # TODO: Call pure_fn 3 times, threading counter through
    result1, counter = pure_fn(x, counter)
    result2, counter = pure_fn(x, counter)
    result3, counter = pure_fn(x, counter)

    # TODO: JIT compile pure_fn and call with x=1.0, counter=0
    pure_fn_jit = partial(jax.jit, donate_argnums=(0,))(pure_fn)
    result_jit, counter_jit = pure_fn_jit(x=1.0, counter=0)

    return {
        'pure_results': [float(result1) if result1 is not None else None,
                        float(result2) if result2 is not None else None,
                        float(result3) if result3 is not None else None],
        'jit_result': float(result_jit) if result_jit is not None else None,
        'final_counter': counter
    }


# =============================================================================
# Exercise 10: Dynamic Shapes and Recompilation Costs
# =============================================================================
def exercise_dynamic_shapes():
    """
    Dynamic shapes cause recompilation, which can be expensive.
    Strategies to handle this:
    1. Pad to fixed sizes
    2. Use static_argnums for dimensions
    3. Accept the recompilation cost for rare shapes

    Your task:
    1. Create a JIT function process_batch(x) that computes mean along axis 0
    2. Create a masked_mean(x, mask) function that computes mean only over
       valid elements indicated by the mask

    Hints:
    - Different batch sizes cause recompilation
    - Padding to fixed size avoids recompilation
    - masked = x * mask[:, None] applies mask to each column
    - Sum masked values and divide by sum of mask for correct mean
    """
    # TODO: Implement this function

    @jit
    def process_batch(x):
        # TODO: return jnp.mean(x, axis=0)
        return jnp.mean(x, axis=0)

    # TODO: Create batches of different sizes
    batch_16 = jnp.ones((16, 10))  # shape (16, 10)
    batch_32 = jnp.ones((32, 10))  # shape (32, 10)
    batch_64 = jnp.ones((64, 10))  # shape (64, 10)

    # TODO: Call process_batch on each (each causes recompilation)
    _ = process_batch(batch_16) if batch_16 is not None else None
    _ = process_batch(batch_32) if batch_32 is not None else None
    _ = process_batch(batch_64) if batch_64 is not None else None

    @jit
    def masked_mean(x, mask):
        """Compute mean only over valid elements."""
        # TODO: masked = x * mask[:, None]
        # TODO: return jnp.sum(masked, axis=0) / jnp.sum(mask)
        masked = x * mask[:, None]
        return jnp.sum(masked, axis = 0) / jnp.sum(mask)

    # TODO: Create padded array x_padded of shape (64, 10) filled with ones
    x_padded = jnp.ones((64, 10))

    # TODO: Create mask with first 32 elements = 1.0, rest = 0.0
    mask = jnp.concatenate([jnp.ones((32,)), jnp.zeros((32, ))])
    # TODO: Compute masked mean
    result_masked = masked_mean(x_padded, mask)

    return {
        'padded_shape': x_padded.shape if x_padded is not None else None,
        'masked_mean_shape': result_masked.shape if result_masked is not None else None,
        'strategy': 'Pad to fixed size and use masks to handle variable sizes'
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX JIT Exercises")
    print("=" * 60)

    exercises = [
        ("1. Basic JIT", exercise_basic_jit),
        ("2. Timing Comparison", exercise_timing_comparison),
        ("3. static_argnums", exercise_static_argnums),
        ("4. donate_argnums", exercise_donate_argnums),
        ("5. JIT Caching", exercise_jit_caching),
        ("6. Debugging", exercise_debugging),
        ("7. make_jaxpr", exercise_make_jaxpr),
        ("8. Side Effects", exercise_side_effects),
        ("9. Pure Functions", exercise_pure_functions),
        ("10. Dynamic Shapes", exercise_dynamic_shapes),
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
