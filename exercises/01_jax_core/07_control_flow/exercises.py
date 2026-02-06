"""
JAX Control Flow - 10 Exercises
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
# Exercise 1: lax.cond for If-Else
# =============================================================================
def exercise_lax_cond():
    """
    lax.cond(condition, true_fn, false_fn, *operands)

    - condition: boolean scalar
    - true_fn: function called if condition is True
    - false_fn: function called if condition is False
    - Both branches must return same-shaped outputs

    TODO:
    - Implement safe_divide(x, y) that returns x/y if y != 0, else 0.0
    - Test with (10.0, 2.0) -> 5.0 and (10.0, 0.0) -> 0.0
    - Implement abs_value(x) using lax.cond with operands
    - Test with -5.0 -> 5.0 and 3.0 -> 3.0
    - Implement JIT-compiled relu(x) using lax.cond
    - Test with -3.0 -> 0.0

    Returns dict with:
    - 'divide_normal': result of safe_divide(10.0, 2.0)
    - 'divide_zero': result of safe_divide(10.0, 0.0)
    - 'abs_negative': abs_value(-5.0)
    - 'abs_positive': abs_value(3.0)
    - 'relu_negative': relu(-3.0)
    """
    # TODO: Implement this function
    def safe_divide(x, y):
        """Divide x by y, returning 0 if y is 0."""
        # TODO: Implement using lax.cond
        pass

    result1 = None
    result2 = None

    # TODO: Implement abs_value with operands
    def abs_value(x):
        # TODO: Implement using lax.cond with operand
        pass

    abs_neg = None
    abs_pos = None

    # TODO: Implement JIT-compiled relu
    @jax.jit
    def relu(x):
        # TODO: Implement using lax.cond
        pass

    relu_result = None

    return {
        'divide_normal': result1,
        'divide_zero': result2,
        'abs_negative': abs_neg,
        'abs_positive': abs_pos,
        'relu_negative': relu_result
    }


# =============================================================================
# Exercise 2: lax.switch for Multi-Branch
# =============================================================================
def exercise_lax_switch():
    """
    lax.switch(index, branches, *operands)

    - index: integer selecting which branch
    - branches: sequence of functions
    - Executes branches[index](*operands)

    TODO:
    - Implement activation(x, activation_type) with 4 branches:
      - 0: linear (identity)
      - 1: relu (max(0, x))
      - 2: tanh
      - 3: sigmoid
    - Test with x = [-1.0, 0.0, 1.0] for all activation types
    - Implement JIT-compiled select_operation(x, op_type) with 3 branches:
      - 0: add 1
      - 1: multiply by 2
      - 2: square
    - Test with x=5.0 for all operations

    Returns dict with:
    - 'linear': activation(x, 0)
    - 'relu': activation(x, 1)
    - 'tanh': activation(x, 2)
    - 'sigmoid': activation(x, 3)
    - 'add_result': select_operation(5.0, 0)
    - 'multiply_result': select_operation(5.0, 1)
    - 'square_result': select_operation(5.0, 2)
    """
    # TODO: Implement this function
    def activation(x, activation_type):
        """Apply different activations based on type."""
        # TODO: Implement using lax.switch
        pass

    x = jnp.array([-1.0, 0.0, 1.0])

    linear = None
    relu = None
    tanh = None
    sigmoid = None

    # TODO: Implement JIT-compiled select_operation
    @jax.jit
    def select_operation(x, op_type):
        # TODO: Implement using lax.switch
        pass

    added = None
    multiplied = None
    squared = None

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
# Exercise 3: lax.while_loop Basics
# =============================================================================
def exercise_while_loop():
    """
    lax.while_loop(cond_fn, body_fn, init_val)

    - cond_fn(val) -> bool: continue while True
    - body_fn(val) -> val: loop body
    - init_val: initial state

    Loop state can be any pytree.

    TODO:
    - Implement countdown from 10 to 0 using while_loop
    - Implement sum_until_threshold(arr, threshold) that sums array elements
      until sum exceeds threshold, returning (num_elements, final_sum)
    - Test with arr=[1,2,3,4,5] and threshold=5 -> (3, 6.0)
    - Implement newton_sqrt(x) using Newton's method:
      - guess = (guess + x/guess) / 2
      - Stop when |guess^2 - x| < 1e-6
    - Test with x=9.0 -> ~3.0

    Returns dict with:
    - 'countdown': result of countdown (should be 0)
    - 'sum_elements': number of elements summed
    - 'sum_value': total sum
    - 'newton_sqrt_9': sqrt(9) via Newton's method
    - 'newton_iterations': number of iterations for sqrt(9)
    """
    # TODO: Implement this function

    # Simple countdown
    def cond_fn(val):
        # TODO: Implement condition
        pass

    def body_fn(val):
        # TODO: Implement body
        pass

    result = None

    # Sum until threshold
    def sum_until_threshold(arr, threshold):
        """Sum array elements until sum exceeds threshold."""
        # TODO: Implement using while_loop with tuple state (idx, total)
        pass

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    num_elements = None
    total = None

    # Newton's method for sqrt
    def newton_sqrt(x, tol=1e-6):
        # TODO: Implement using while_loop
        pass

    sqrt_9 = None
    iters = None

    return {
        'countdown': result,
        'sum_elements': num_elements,
        'sum_value': total,
        'newton_sqrt_9': sqrt_9,
        'newton_iterations': iters
    }


# =============================================================================
# Exercise 4: lax.fori_loop for Fixed Iterations
# =============================================================================
def exercise_fori_loop():
    """
    lax.fori_loop(lower, upper, body_fn, init_val)

    - Iterates from lower to upper-1
    - body_fn(i, val) -> new_val
    - More efficient than while_loop when iterations known

    TODO:
    - Sum 0 to 9 using fori_loop -> 45
    - Implement matrix_power(A, n) using fori_loop
    - Test with Fibonacci matrix [[1,1],[1,0]] cubed
    - Implement running_mean(arr) using fori_loop
    - Test with [1,2,3,4,5] -> 3.0
    - Implement cumsum_via_fori(arr) that computes cumulative sum
    - Test with [1,2,3,4,5] -> [1,3,6,10,15]

    Returns dict with:
    - 'sum_0_to_9': sum of 0 to 9
    - 'matrix_cubed': A^3 where A is Fibonacci matrix
    - 'running_mean': mean of [1,2,3,4,5]
    - 'cumsum': cumulative sum of [1,2,3,4,5]
    """
    # TODO: Implement this function

    # Sum 0 to 9
    def body_fn(i, total):
        # TODO: Implement
        pass

    result = None

    # Matrix power via repeated multiplication
    def matrix_power(A, n):
        # TODO: Implement using fori_loop
        pass

    A = jnp.array([[1, 1], [1, 0]], dtype=jnp.float32)  # Fibonacci matrix
    A_cubed = None

    # Running mean
    def running_mean(arr):
        # TODO: Implement using fori_loop with tuple state (total, count)
        pass

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = None

    # Cumulative sum
    def cumsum_via_fori(arr):
        # TODO: Implement using fori_loop
        pass

    cumsum = None

    return {
        'sum_0_to_9': result,
        'matrix_cubed': A_cubed,
        'running_mean': mean,
        'cumsum': cumsum
    }


# =============================================================================
# Exercise 5: Nested Control Flow
# =============================================================================
def exercise_nested_control_flow():
    """
    Control flow primitives can be nested.

    TODO:
    - Implement nested_example(x, y) with nested lax.cond:
      - x > 0, y > 0: return x + y
      - x > 0, y <= 0: return x - y
      - x <= 0, y > 0: return y - x
      - x <= 0, y <= 0: return -(x + y)
    - Test with (3,2), (3,-2), (-3,2), (-3,-2) - all should give 5
    - Implement conditional_sum(arr, threshold) using fori_loop with cond inside
    - Test with [1,5,2,8,3] and threshold=3 -> 13 (5+8)

    Returns dict with:
    - 'both_pos': nested_example(3.0, 2.0)
    - 'x_pos_y_neg': nested_example(3.0, -2.0)
    - 'x_neg_y_pos': nested_example(-3.0, 2.0)
    - 'both_neg': nested_example(-3.0, -2.0)
    - 'conditional_sum': sum of values > 3
    """
    # TODO: Implement this function
    def nested_example(x, y):
        """Complex nested control flow."""
        # TODO: Implement using nested lax.cond
        pass

    results = {
        'both_pos': None,
        'x_pos_y_neg': None,
        'x_neg_y_pos': None,
        'both_neg': None,
    }

    # Loop with conditionals inside
    def conditional_sum(arr, threshold):
        """Sum only values above threshold."""
        # TODO: Implement using fori_loop with lax.cond inside
        pass

    arr = jnp.array([1.0, 5.0, 2.0, 8.0, 3.0])
    sum_above_3 = None

    return {
        **results,
        'conditional_sum': sum_above_3
    }


# =============================================================================
# Exercise 6: Control Flow with grad
# =============================================================================
def exercise_control_flow_grad():
    """
    JAX can differentiate through control flow!
    Gradients follow the path taken in the forward pass.

    TODO:
    - Implement f(x) that returns x^2 if x > 0, else -x
    - Compute gradient at x=3.0 (should be 6, since d/dx(x^2) = 2x)
    - Compute gradient at x=-3.0 (should be -1, since d/dx(-x) = -1)
    - Implement power_iteration(x, n) that computes x^n using while_loop
    - Compute d/dx(x^3) at x=2.0 (should be 12, since 3*x^2 = 3*4)
    - Implement smooth_max(x, y) using lax.cond
    - Compute d/dx(smooth_max(x, 0)) at x=1.0 (should be 1)
    - Compute d/dx(smooth_max(x, 0)) at x=-1.0 (should be 0)

    Returns dict with:
    - 'grad_positive_branch': gradient at x=3.0
    - 'grad_negative_branch': gradient at x=-3.0
    - 'grad_power_iteration': d/dx(x^3) at x=2.0
    - 'grad_max_x_larger': d/dx(max(x,0)) at x=1.0
    - 'grad_max_x_smaller': d/dx(max(x,0)) at x=-1.0
    """
    # TODO: Implement this function
    def f(x):
        """Function with branching."""
        # TODO: Implement using lax.cond
        pass

    df = None  # TODO: Create gradient function

    grad_positive = None
    grad_negative = None

    # Gradient through while_loop
    def power_iteration(x, n):
        """Compute x^n using iteration."""
        # TODO: Implement using while_loop
        pass

    # TODO: Compute gradient of x^3 at x=2.0
    grad_power = None

    # Gradient of max (non-smooth)
    def smooth_max(x, y):
        """Differentiable approximation of max."""
        # TODO: Implement using lax.cond
        pass

    grad_x_larger = None
    grad_x_smaller = None

    return {
        'grad_positive_branch': grad_positive,
        'grad_negative_branch': grad_negative,
        'grad_power_iteration': grad_power,
        'grad_max_x_larger': grad_x_larger,
        'grad_max_x_smaller': grad_x_smaller
    }


# =============================================================================
# Exercise 7: Python vs JAX Control Flow
# =============================================================================
def exercise_python_vs_jax():
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

    TODO:
    - Implement python_branch(x, use_square=True) using Python if
      - Returns x^2 if use_square else x^3
    - Test with [1,2,3] for both cases
    - Implement JIT-compiled jax_branch(x, threshold) using lax.cond
      - Returns x^2 if x > threshold else x^3
    - Test with x=5.0, threshold=3.0 -> 25.0
    - Implement python_loop(x, n) using Python for loop
      - Adds 0+1+2+...+(n-1) to x
    - Test with x=0, n=5 -> 10
    - Implement JIT-compiled jax_loop(x, n) using fori_loop
    - Test with x=0, n=5 -> 10

    Returns dict with:
    - 'python_squared': python_branch([1,2,3], True)
    - 'python_cubed': python_branch([1,2,3], False)
    - 'jax_dynamic_branch': jax_branch(5.0, 3.0)
    - 'python_static_loop': python_loop(0, 5)
    - 'jax_dynamic_loop': jax_loop(0, 5)
    """
    # TODO: Implement this function

    # Python control flow is OK when condition is static
    def python_branch(x, use_square=True):
        """Static branching - use_square known at trace time."""
        # TODO: Implement using Python if
        pass

    squared = None
    cubed = None

    # JAX control flow needed when condition is dynamic
    @jax.jit
    def jax_branch(x, threshold):
        """Dynamic branching - threshold is an array value."""
        # TODO: Implement using lax.cond
        pass

    dynamic_result = None

    # Looping over static range is OK with Python
    def python_loop(x, n):
        """Static loop - n known at trace time."""
        # TODO: Implement using Python for loop
        pass

    static_loop_result = None

    # Must use JAX loop when iterations depend on runtime values
    @jax.jit
    def jax_loop(x, n):
        """Dynamic loop - n is runtime value."""
        # TODO: Implement using fori_loop
        pass

    dynamic_loop_result = None

    return {
        'python_squared': squared,
        'python_cubed': cubed,
        'jax_dynamic_branch': dynamic_result,
        'python_static_loop': static_loop_result,
        'jax_dynamic_loop': dynamic_loop_result
    }


# =============================================================================
# Exercise 8: lax.select for Element-wise Conditionals
# =============================================================================
def exercise_lax_select():
    """
    lax.select(condition, on_true, on_false)

    Element-wise conditional selection (vectorized if-else).
    Like jnp.where but lower-level.

    TODO:
    - Apply element-wise ReLU to x = [-2,-1,0,1,2] using lax.select
    - Clip x to range [-1, 1] using lax.select
    - Implement piecewise function: f(x) = x^2 if x >= 0 else -x
    - Implement multi-condition: f(x) = 0 if x < -1, x+1 if -1 <= x < 0, x if x >= 0
    - Compare with jnp.where to verify they match

    Returns dict with:
    - 'relu': element-wise ReLU result
    - 'clipped': values clipped to [-1, 1]
    - 'piecewise': piecewise function result
    - 'multi_condition': multi-condition result
    - 'where_matches': bool, True if lax.select matches jnp.where
    """
    # TODO: Implement this function
    x = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.float32)

    # Element-wise ReLU
    relu = None

    # Clip to range
    clipped = None

    # Piecewise function
    piecewise = None

    # Combining multiple conditions
    result = None

    # Compare with jnp.where
    where_result = None

    return {
        'relu': relu,
        'clipped': clipped,
        'piecewise': piecewise,
        'multi_condition': result,
        'where_matches': None
    }


# =============================================================================
# Exercise 9: Early Stopping Patterns
# =============================================================================
def exercise_early_stopping():
    """
    Patterns for early termination in loops.

    TODO:
    - Implement find_first_above(arr, threshold) using while_loop
      - Returns index of first element > threshold, or -1 if not found
    - Test with [1,2,5,3,8] and threshold=4 -> 2 (index of 5)
    - Test with threshold=10 -> -1
    - Implement iterate_until_converge(x0) that iterates x = cos(x)
      - Stop when |x - prev_x| < 1e-6 or max_iters=100
      - Returns (final_x, num_iterations)
    - Test with x0=0.5 -> ~0.739 (fixed point of cos)
    - Implement gradient_descent to minimize f(x) = (x-3)^2
      - Use learning_rate=0.1, tol=1e-6, max_iters=1000
      - Returns (x, num_iterations)
    - Test with x0=0.0 -> ~3.0

    Returns dict with:
    - 'first_above_4': index of first element > 4
    - 'first_above_10': index of first element > 10 (-1)
    - 'fixed_point': fixed point of cos(x)
    - 'convergence_iters': iterations to converge
    - 'gd_minimum': minimum found by gradient descent
    - 'gd_iterations': iterations for gradient descent
    """
    # TODO: Implement this function

    # Find first element greater than threshold
    def find_first_above(arr, threshold):
        """Find index of first element > threshold."""
        # TODO: Implement using while_loop with early stopping
        pass

    arr = jnp.array([1.0, 2.0, 5.0, 3.0, 8.0])
    first_above_4 = None
    first_above_10 = None

    # Converge until tolerance
    def iterate_until_converge(x0, tol=1e-6, max_iters=100):
        """Iterate x = cos(x) until convergence."""
        # TODO: Implement using while_loop
        pass

    fixed_point = None
    iters = None

    # Gradient descent with early stopping
    def gradient_descent(x0, learning_rate=0.1, tol=1e-6, max_iters=1000):
        """Minimize f(x) = (x-3)^2 with early stopping."""
        # TODO: Implement using while_loop
        pass

    min_x = None
    gd_iters = None

    return {
        'first_above_4': first_above_4,
        'first_above_10': first_above_10,
        'fixed_point': fixed_point,
        'convergence_iters': iters,
        'gd_minimum': min_x,
        'gd_iterations': gd_iters
    }


# =============================================================================
# Exercise 10: Control Flow Performance Considerations
# =============================================================================
def exercise_performance():
    """
    Performance tips for control flow in JAX.

    TODO:
    - Implement fori_sum() that sums 0 to n-1 using fori_loop (n=1000)
    - Implement while_sum() that sums 0 to n-1 using while_loop
    - Verify both give same result (499500)
    - JIT-compile both and verify they still match
    - Note: fori_loop is more efficient when iteration count is known

    Returns dict with:
    - 'fori_result': sum using fori_loop
    - 'while_result': sum using while_loop
    - 'results_match': bool, True if both give same result
    - 'vectorized_time_100': time for 100 vectorized sums (just a placeholder)
    - 'tip': performance tip string
    """
    # TODO: Implement this function
    import time

    n = 1000

    # fori_loop: knows iteration count, more optimizable
    def fori_sum():
        # TODO: Implement using fori_loop
        pass

    # while_loop: more general but slightly less efficient
    def while_sum():
        # TODO: Implement using while_loop
        pass

    fori_result = None
    while_result = None

    # JIT compilation
    jit_fori = None
    jit_while = None

    # Vectorized approach for comparison
    def vectorized(arr):
        """Sum using vectorized operation."""
        return jnp.sum(arr)

    arr = jnp.arange(10000, dtype=jnp.float32)
    jit_vec = jax.jit(vectorized)

    # Warm up and time
    _ = jit_vec(arr).block_until_ready()

    start = time.perf_counter()
    for _ in range(100):
        _ = jit_vec(arr).block_until_ready()
    vec_time = time.perf_counter() - start

    return {
        'fori_result': fori_result,
        'while_result': while_result,
        'results_match': None,
        'vectorized_time_100': vec_time,
        'tip': 'Use vectorized operations when possible'
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Control Flow Exercises")
    print("=" * 60)

    exercises = [
        ("1. lax.cond", exercise_lax_cond),
        ("2. lax.switch", exercise_lax_switch),
        ("3. lax.while_loop", exercise_while_loop),
        ("4. lax.fori_loop", exercise_fori_loop),
        ("5. Nested Control Flow", exercise_nested_control_flow),
        ("6. Control Flow + grad", exercise_control_flow_grad),
        ("7. Python vs JAX", exercise_python_vs_jax),
        ("8. lax.select", exercise_lax_select),
        ("9. Early Stopping", exercise_early_stopping),
        ("10. Performance", exercise_performance),
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
