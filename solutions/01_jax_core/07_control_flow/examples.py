"""
JAX Control Flow - 10 Examples
===============================

Python if/for/while are evaluated at trace time, not runtime.
For runtime control flow inside JIT, use JAX primitives:
- jax.lax.cond: if-else
- jax.lax.switch: multi-branch
- jax.lax.while_loop: dynamic loops
- jax.lax.fori_loop: fixed iteration loops

Reference: https://jax.readthedocs.io/en/latest/jax-101/07-control-flow.html
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# =============================================================================
# Example 1: lax.cond for If-Else
# =============================================================================
def example_lax_cond():
    """
    lax.cond(condition, true_fn, false_fn, *operands)

    - condition: boolean scalar
    - true_fn: function called if condition is True
    - false_fn: function called if condition is False
    - Both branches must return same-shaped outputs
    """
    def safe_divide(x, y):
        """Divide x by y, returning 0 if y is 0."""
        return lax.cond(
            y != 0,
            lambda: x / y,   # true branch
            lambda: 0.0      # false branch
        )

    result1 = safe_divide(10.0, 2.0)   # 5.0
    result2 = safe_divide(10.0, 0.0)   # 0.0

    # With operands (cleaner for complex branches)
    def abs_value(x):
        return lax.cond(
            x >= 0,
            lambda val: val,    # true: return x
            lambda val: -val,   # false: return -x
            x  # operand passed to both functions
        )

    abs_neg = abs_value(-5.0)  # 5.0
    abs_pos = abs_value(3.0)   # 3.0

    # JIT compatible
    @jax.jit
    def relu(x):
        return lax.cond(x > 0, lambda: x, lambda: 0.0)

    relu_result = relu(jnp.array(-3.0))

    return {
        'divide_normal': result1,
        'divide_zero': result2,
        'abs_negative': abs_neg,
        'abs_positive': abs_pos,
        'relu_negative': relu_result
    }


# =============================================================================
# Example 2: lax.switch for Multi-Branch
# =============================================================================
def example_lax_switch():
    """
    lax.switch(index, branches, *operands)

    - index: integer selecting which branch
    - branches: sequence of functions
    - Executes branches[index](*operands)
    """
    def activation(x, activation_type):
        """Apply different activations based on type."""
        branches = [
            lambda val: val,                    # 0: linear
            lambda val: jnp.maximum(0, val),    # 1: relu
            lambda val: jnp.tanh(val),          # 2: tanh
            lambda val: jax.nn.sigmoid(val),    # 3: sigmoid
        ]
        return lax.switch(activation_type, branches, x)

    x = jnp.array([-1.0, 0.0, 1.0])

    linear = activation(x, 0)
    relu = activation(x, 1)
    tanh = activation(x, 2)
    sigmoid = activation(x, 3)

    # JIT compatible selection
    @jax.jit
    def select_operation(x, op_type):
        ops = [
            lambda v: v + 1,   # add
            lambda v: v * 2,   # multiply
            lambda v: v ** 2,  # square
        ]
        return lax.switch(op_type, ops, x)

    added = select_operation(5.0, 0)    # 6.0
    multiplied = select_operation(5.0, 1)  # 10.0
    squared = select_operation(5.0, 2)    # 25.0

    return {
        'linear': linear,
        'relu': relu,
        'tanh': tanh,
        'sigmoid': sigmoid,
        'add_result': added,
        'multiply_result': multiplied,
        'square_result': squared
    }


# =============================================================================
# Example 3: lax.while_loop Basics
# =============================================================================
def example_while_loop():
    """
    lax.while_loop(cond_fn, body_fn, init_val)

    - cond_fn(val) -> bool: continue while True
    - body_fn(val) -> val: loop body
    - init_val: initial state

    Loop state can be any pytree.
    """
    # Simple countdown
    def cond_fn(val):
        return val > 0

    def body_fn(val):
        return val - 1

    result = lax.while_loop(cond_fn, body_fn, 10)  # 0

    # Sum until threshold
    def sum_until_threshold(arr, threshold):
        """Sum array elements until sum exceeds threshold."""
        def cond(state):
            idx, total = state
            return (total < threshold) & (idx < len(arr))

        def body(state):
            idx, total = state
            return (idx + 1, total + arr[idx])

        init_state = (0, 0.0)
        final_idx, final_sum = lax.while_loop(cond, body, init_state)
        return final_idx, final_sum

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    num_elements, total = sum_until_threshold(arr, 5.0)

    # Newton's method for sqrt
    def newton_sqrt(x, tol=1e-6):
        def cond(state):
            guess, _ = state
            return jnp.abs(guess ** 2 - x) > tol

        def body(state):
            guess, iters = state
            new_guess = (guess + x / guess) / 2
            return (new_guess, iters + 1)

        final_guess, iterations = lax.while_loop(cond, body, (x / 2, 0))
        return final_guess, iterations

    sqrt_9, iters = newton_sqrt(9.0)

    return {
        'countdown': result,
        'sum_elements': num_elements,
        'sum_value': total,
        'newton_sqrt_9': sqrt_9,
        'newton_iterations': iters
    }


# =============================================================================
# Example 4: lax.fori_loop for Fixed Iterations
# =============================================================================
def example_fori_loop():
    """
    lax.fori_loop(lower, upper, body_fn, init_val)

    - Iterates from lower to upper-1
    - body_fn(i, val) -> new_val
    - More efficient than while_loop when iterations known
    """
    # Sum 0 to 9
    def body_fn(i, total):
        return total + i

    result = lax.fori_loop(0, 10, body_fn, 0)  # 45

    # Matrix power via repeated multiplication
    def matrix_power(A, n):
        def body(_, current):
            return current @ A

        return lax.fori_loop(1, n, body, A)

    A = jnp.array([[1, 1], [1, 0]], dtype=jnp.float32)  # Fibonacci matrix
    A_cubed = matrix_power(A, 3)

    # Running mean
    def running_mean(arr):
        def body(i, state):
            total, count = state
            new_total = total + arr[i]
            new_count = count + 1
            return (new_total, new_count)

        total, count = lax.fori_loop(0, len(arr), body, (0.0, 0))
        return total / count

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = running_mean(arr)  # 3.0

    # Accumulate all intermediate values
    def cumsum_via_fori(arr):
        n = len(arr)
        def body(i, cumsum):
            return cumsum.at[i].set(cumsum[i-1] + arr[i] if i > 0 else arr[i])

        init = jnp.zeros(n)
        init = init.at[0].set(arr[0])
        return lax.fori_loop(1, n, body, init)

    cumsum = cumsum_via_fori(arr)

    return {
        'sum_0_to_9': result,
        'matrix_cubed': A_cubed,
        'running_mean': mean,
        'cumsum': cumsum
    }


# =============================================================================
# Example 5: Nested Control Flow
# =============================================================================
def example_nested_control_flow():
    """
    Control flow primitives can be nested.
    """
    def nested_example(x, y):
        """Complex nested control flow."""
        # Outer condition
        return lax.cond(
            x > 0,
            # If x > 0, check y
            lambda: lax.cond(
                y > 0,
                lambda: x + y,      # Both positive
                lambda: x - y       # x pos, y non-pos
            ),
            # If x <= 0
            lambda: lax.cond(
                y > 0,
                lambda: y - x,      # x non-pos, y pos
                lambda: -(x + y)    # Both non-positive
            )
        )

    results = {
        'both_pos': nested_example(3.0, 2.0),     # 5
        'x_pos_y_neg': nested_example(3.0, -2.0), # 5
        'x_neg_y_pos': nested_example(-3.0, 2.0), # 5
        'both_neg': nested_example(-3.0, -2.0),   # 5
    }

    # Loop with conditionals inside
    def conditional_sum(arr, threshold):
        """Sum only values above threshold."""
        def body(i, total):
            return lax.cond(
                arr[i] > threshold,
                lambda: total + arr[i],
                lambda: total
            )

        return lax.fori_loop(0, len(arr), body, 0.0)

    arr = jnp.array([1.0, 5.0, 2.0, 8.0, 3.0])
    sum_above_3 = conditional_sum(arr, 3.0)  # 5 + 8 = 13

    return {
        **results,
        'conditional_sum': sum_above_3
    }


# =============================================================================
# Example 6: Control Flow with grad
# =============================================================================
def example_control_flow_grad():
    """
    JAX can differentiate through control flow!
    Gradients follow the path taken in the forward pass.
    """
    def f(x):
        """Function with branching."""
        return lax.cond(
            x > 0,
            lambda: x ** 2,    # d/dx = 2x
            lambda: -x         # d/dx = -1
        )

    df = jax.grad(f)

    grad_positive = df(3.0)   # 2 * 3 = 6
    grad_negative = df(-3.0)  # -1

    # Gradient through while_loop
    def power_iteration(x, n):
        """Compute x^n using iteration."""
        def cond(state):
            i, _ = state
            return i < n

        def body(state):
            i, result = state
            return (i + 1, result * x)

        _, result = lax.while_loop(cond, body, (0, 1.0))
        return result

    # d/dx(x^3) = 3x^2
    grad_power = jax.grad(lambda x: power_iteration(x, 3))(2.0)  # 3 * 4 = 12

    # Gradient of max (non-smooth)
    def smooth_max(x, y):
        """Differentiable approximation of max."""
        return lax.cond(x > y, lambda: x, lambda: y)

    grad_x_larger = jax.grad(lambda x: smooth_max(x, 0.0))(1.0)  # 1.0
    grad_x_smaller = jax.grad(lambda x: smooth_max(x, 0.0))(-1.0)  # 0.0

    return {
        'grad_positive_branch': grad_positive,
        'grad_negative_branch': grad_negative,
        'grad_power_iteration': grad_power,
        'grad_max_x_larger': grad_x_larger,
        'grad_max_x_smaller': grad_x_smaller
    }


# =============================================================================
# Example 7: Python vs JAX Control Flow
# =============================================================================
def example_python_vs_jax():
    """
    When to use Python control flow vs JAX control flow.

    Python control flow: evaluated at TRACE TIME (compile time)
    JAX control flow: evaluated at RUN TIME

    Use Python when:
    - Condition depends on static values (shapes, dtypes)
    - Branching doesn't depend on array values

    Use JAX when:
    - Condition depends on array values
    - Inside JIT-compiled functions
    """
    # Python control flow is OK when condition is static
    def python_branch(x, use_square=True):
        """Static branching - use_square known at trace time."""
        if use_square:  # Python if is fine here
            return x ** 2
        else:
            return x ** 3

    # This works and is efficient
    squared = python_branch(jnp.array([1.0, 2.0, 3.0]), use_square=True)
    cubed = python_branch(jnp.array([1.0, 2.0, 3.0]), use_square=False)

    # JAX control flow needed when condition is dynamic
    @jax.jit
    def jax_branch(x, threshold):
        """Dynamic branching - threshold is an array value."""
        # if x > threshold:  # WRONG - x is traced, not known
        return lax.cond(
            x > threshold,
            lambda: x ** 2,
            lambda: x ** 3
        )

    dynamic_result = jax_branch(5.0, 3.0)  # x > threshold, so x^2 = 25

    # Looping over static range is OK with Python
    def python_loop(x, n):
        """Static loop - n known at trace time."""
        for i in range(n):  # Python for is fine
            x = x + i
        return x

    static_loop_result = python_loop(0.0, 5)  # 0+1+2+3+4 = 10

    # Must use JAX loop when iterations depend on runtime values
    @jax.jit
    def jax_loop(x, n):
        """Dynamic loop - n is runtime value."""
        return lax.fori_loop(0, n, lambda i, acc: acc + i, x)

    dynamic_loop_result = jax_loop(0.0, 5)  # Same result

    return {
        'python_squared': squared,
        'python_cubed': cubed,
        'jax_dynamic_branch': dynamic_result,
        'python_static_loop': static_loop_result,
        'jax_dynamic_loop': dynamic_loop_result
    }


# =============================================================================
# Example 8: lax.select for Element-wise Conditionals
# =============================================================================
def example_lax_select():
    """
    lax.select(condition, on_true, on_false)

    Element-wise conditional selection (vectorized if-else).
    Like jnp.where but lower-level.
    """
    x = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.float32)

    # Element-wise ReLU
    relu = lax.select(x > 0, x, jnp.zeros_like(x))

    # Clip to range
    clipped = lax.select(x > 1, jnp.ones_like(x), x)
    clipped = lax.select(clipped < -1, -jnp.ones_like(clipped), clipped)

    # Piecewise function
    # f(x) = x^2 if x >= 0 else -x
    piecewise = lax.select(x >= 0, x ** 2, -x)

    # Combining multiple conditions
    # f(x) = 0 if x < -1, x+1 if -1 <= x < 0, x if x >= 0
    result = jnp.zeros_like(x)
    result = lax.select(x >= 0, x, result)
    result = lax.select((x >= -1) & (x < 0), x + 1, result)

    # Compare with jnp.where
    where_result = jnp.where(x > 0, x, jnp.zeros_like(x))

    return {
        'relu': relu,
        'clipped': clipped,
        'piecewise': piecewise,
        'multi_condition': result,
        'where_matches': jnp.allclose(relu, where_result)
    }


# =============================================================================
# Example 9: Early Stopping Patterns
# =============================================================================
def example_early_stopping():
    """
    Patterns for early termination in loops.
    """
    # Find first element greater than threshold
    def find_first_above(arr, threshold):
        """Find index of first element > threshold."""
        def cond(state):
            idx, found = state
            return (idx < len(arr)) & ~found

        def body(state):
            idx, _ = state
            found = arr[idx] > threshold
            return (idx + 1, found)

        final_idx, found = lax.while_loop(cond, body, (0, False))
        return lax.cond(found, lambda: final_idx - 1, lambda: -1)

    arr = jnp.array([1.0, 2.0, 5.0, 3.0, 8.0])
    first_above_4 = find_first_above(arr, 4.0)  # 2 (index of 5.0)
    first_above_10 = find_first_above(arr, 10.0)  # -1 (not found)

    # Converge until tolerance
    def iterate_until_converge(x0, tol=1e-6, max_iters=100):
        """Iterate x = cos(x) until convergence."""
        def cond(state):
            x, prev_x, iters = state
            diff = jnp.abs(x - prev_x)
            return (diff > tol) & (iters < max_iters)

        def body(state):
            x, _, iters = state
            new_x = jnp.cos(x)
            return (new_x, x, iters + 1)

        x, _, iterations = lax.while_loop(cond, body, (x0, x0 + 1, 0))
        return x, iterations

    fixed_point, iters = iterate_until_converge(0.5)

    # Gradient descent with early stopping
    def gradient_descent(x0, learning_rate=0.1, tol=1e-6, max_iters=1000):
        """Minimize f(x) = (x-3)^2 with early stopping."""
        def f(x):
            return (x - 3) ** 2

        def cond(state):
            x, prev_x, iters = state
            return (jnp.abs(x - prev_x) > tol) & (iters < max_iters)

        def body(state):
            x, _, iters = state
            grad = jax.grad(f)(x)
            new_x = x - learning_rate * grad
            return (new_x, x, iters + 1)

        x, _, iterations = lax.while_loop(cond, body, (x0, x0 + 1, 0))
        return x, iterations

    min_x, gd_iters = gradient_descent(0.0)

    return {
        'first_above_4': first_above_4,
        'first_above_10': first_above_10,
        'fixed_point': fixed_point,
        'convergence_iters': iters,
        'gd_minimum': min_x,
        'gd_iterations': gd_iters
    }


# =============================================================================
# Example 10: Control Flow Performance Considerations
# =============================================================================
def example_performance():
    """
    Performance tips for control flow in JAX.
    """
    import time

    # fori_loop vs while_loop for fixed iterations
    n = 1000

    # fori_loop: knows iteration count, more optimizable
    def fori_sum():
        return lax.fori_loop(0, n, lambda i, s: s + i, 0)

    # while_loop: more general but slightly less efficient
    def while_sum():
        def cond(state):
            i, _ = state
            return i < n

        def body(state):
            i, s = state
            return (i + 1, s + i)

        _, result = lax.while_loop(cond, body, (0, 0))
        return result

    # Both give same result
    fori_result = fori_sum()
    while_result = while_sum()

    # JIT compilation amortizes control flow overhead
    jit_fori = jax.jit(fori_sum)
    jit_while = jax.jit(while_sum)

    # Warm up
    _ = jit_fori()
    _ = jit_while()

    # For simple operations, vmap may be faster than explicit loops
    def loop_approach(arr):
        """Sum using fori_loop."""
        return lax.fori_loop(0, len(arr), lambda i, s: s + arr[i], 0.0)

    def vectorized(arr):
        """Sum using vectorized operation."""
        return jnp.sum(arr)

    arr = jnp.arange(10000, dtype=jnp.float32)

    jit_loop = jax.jit(loop_approach)
    jit_vec = jax.jit(vectorized)

    # Warm up
    _ = jit_loop(arr).block_until_ready()
    _ = jit_vec(arr).block_until_ready()

    # Vectorized is usually faster for simple reductions
    start = time.perf_counter()
    for _ in range(100):
        _ = jit_vec(arr).block_until_ready()
    vec_time = time.perf_counter() - start

    return {
        'fori_result': fori_result,
        'while_result': while_result,
        'results_match': fori_result == while_result,
        'vectorized_time_100': vec_time,
        'tip': 'Use vectorized operations when possible'
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Control Flow Examples")
    print("=" * 60)

    examples = [
        ("1. lax.cond", example_lax_cond),
        ("2. lax.switch", example_lax_switch),
        ("3. lax.while_loop", example_while_loop),
        ("4. lax.fori_loop", example_fori_loop),
        ("5. Nested Control Flow", example_nested_control_flow),
        ("6. Control Flow + grad", example_control_flow_grad),
        ("7. Python vs JAX", example_python_vs_jax),
        ("8. lax.select", example_lax_select),
        ("9. Early Stopping", example_early_stopping),
        ("10. Performance", example_performance),
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
