"""
JAX Automatic Differentiation (grad) - 10 Exercises
====================================================

JAX's grad transformation computes gradients automatically.
This is the foundation for training neural networks.

Key concepts:
- grad(f) returns a function that computes the gradient of f
- Works with scalar-output functions by default
- Can differentiate through control flow
- Composable with other transformations (jit, vmap)

Reference: https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html

Instructions:
- Complete each exercise by implementing the TODOs
- Each function should return a dictionary with the specified keys
- Run the tests with: pytest test_exercises.py -v
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
from functools import partial


# =============================================================================
# Exercise 1: Basic grad for Scalar Functions
# =============================================================================
def exercise_basic_grad():
    """
    grad(f) returns a function that computes df/dx.
    f must return a scalar (single number).

    Tasks:
    1. Define f(x) = x^2 and compute its gradient at x=3
       Hint: The derivative of x^2 is 2x, so at x=3 it should be 6

    2. Define g(x) = sin(x) * cos(x) and compute its gradient at x=pi/4
       Hint: The derivative is cos(2x), so at x=pi/4 it should be ~0

    3. Define h(x) = exp(-x^2) (Gaussian) and compute its gradient at x=1
       Hint: The derivative is -2x * exp(-x^2)

    Returns:
        dict with keys:
        - 'x_squared_grad_at_3': gradient of x^2 at x=3 (should be 6.0)
        - 'expected_6': the value 6.0 for comparison
        - 'sin_cos_grad': gradient of sin(x)*cos(x) at x=pi/4 (should be ~0)
        - 'gaussian_grad': gradient of exp(-x^2) at x=1 (should be -2*exp(-1))
    """
    # TODO: Implement f(x) = x^2
    def f(x):
        # TODO: Implement this function
        return x**2

    # TODO: Create gradient function and compute gradient at x=3
    df = grad(f)  # Hint: use grad(f)
    x = 3.0
    gradient = df(x)  # Hint: call df(x)

    # TODO: Implement g(x) = sin(x) * cos(x)
    def g(x):
        # TODO: Implement this function
        return jnp.sin(x) * jnp.cos(x)

    # TODO: Create gradient function and compute gradient at x=pi/4
    dg = grad(g)
    x2 = jnp.pi / 4
    gradient2 = dg(x2)

    # TODO: Implement h(x) = exp(-x^2)
    def h(x):
        # TODO: Implement this function
        return jnp.exp(-(x**2))

    # TODO: Create gradient function and compute gradient at x=1
    dh = grad(h)
    x3 = 1.0
    gradient3 = dh(x3)

    return {
        'x_squared_grad_at_3': gradient,
        'expected_6': 6.0,
        'sin_cos_grad': gradient2,
        'gaussian_grad': gradient3
    }


# =============================================================================
# Exercise 2: value_and_grad for Efficiency
# =============================================================================
def exercise_value_and_grad():
    """
    value_and_grad computes both f(x) and grad(f)(x) in one pass.
    More efficient than calling f(x) and grad(f)(x) separately.

    Tasks:
    1. Define a loss function: MSE between predictions and targets
       loss(w, x, y) = mean((w * x - y)^2)

    2. Use value_and_grad to get both loss and gradient
       Hint: loss_and_grad = value_and_grad(loss)

    3. Verify that value_and_grad gives the same results as separate calls

    Returns:
        dict with keys:
        - 'loss': the loss value (should be 0.0 when w=2 and y=2x)
        - 'gradient': the gradient w.r.t. w (should be 0.0 at optimum)
        - 'same_loss': bool, whether value_and_grad gives same loss
        - 'same_grad': bool, whether value_and_grad gives same gradient
    """
    # TODO: Implement the MSE loss function
    def loss(w, x, y):
        """Mean squared error loss."""
        # TODO: Implement this function
        # pred = w * x
        # return mean of (pred - y)^2
        pred = w * x
        return jnp.mean((pred - y)**2)

    # TODO: Create value_and_grad function
    loss_and_grad = value_and_grad(loss)  # Hint: value_and_grad(loss)

    w = 2.0
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([2.0, 4.0, 6.0])  # y = 2x, so optimal w = 2

    # TODO: Get both loss and gradient using value_and_grad
    loss_val, grad_val = loss_and_grad(w, x, y)  # Hint: loss_and_grad(w, x, y)

    # TODO: Compare with separate calls
    loss_separate = loss(w, x, y)  # Hint: loss(w, x, y)
    grad_separate = grad(loss)(w, x, y)  # Hint: grad(loss)(w, x, y)

    return {
        'loss': loss_val,
        'gradient': grad_val,
        'same_loss': jnp.allclose(loss_val, loss_separate) if loss_val is not None else False,
        'same_grad': jnp.allclose(grad_val, grad_separate) if grad_val is not None else False
    }


# =============================================================================
# Exercise 3: grad with argnums for Multiple Inputs
# =============================================================================
def exercise_argnums():
    """
    Use argnums to specify which argument(s) to differentiate with respect to.
    Default is argnums=0 (first argument).

    Tasks:
    1. Define f(x, y, z) = x*y + y*z + z*x

    2. Compute gradients w.r.t. each argument individually
       Hint: grad(f, argnums=0) for df/dx

    3. Compute gradients w.r.t. all arguments at once
       Hint: grad(f, argnums=(0, 1, 2)) returns a tuple

    Expected gradients at (x=1, y=2, z=3):
    - df/dx = y + z = 5
    - df/dy = x + z = 4
    - df/dz = y + x = 3

    Returns:
        dict with keys:
        - 'grad_x': gradient w.r.t. x (should be 5.0)
        - 'grad_y': gradient w.r.t. y (should be 4.0)
        - 'grad_z': gradient w.r.t. z (should be 3.0)
        - 'grad_all': tuple of all gradients ((5.0, 4.0, 3.0))
        - 'expected': the expected tuple (5.0, 4.0, 3.0)
    """
    # TODO: Implement f(x, y, z) = x*y + y*z + z*x
    def f(x, y, z):
        # TODO: Implement this function
        return x*y + y*z + z*x

    # TODO: Create gradient functions for each argument
    df_dx = grad(f, argnums=0)  # Hint: grad(f, argnums=0)
    df_dy = grad(f, argnums=1)  # Hint: grad(f, argnums=1)
    df_dz = grad(f, argnums=2)  # Hint: grad(f, argnums=2)

    # TODO: Create gradient function for all arguments
    df_dxyz = grad(f, argnums=(0,1,2))  # Hint: grad(f, argnums=(0, 1, 2))

    x, y, z = 1.0, 2.0, 3.0

    # TODO: Compute gradients
    grad_x = df_dx(x, y, z)  # Should be 5.0
    grad_y = df_dy(x, y, z)  # Should be 4.0
    grad_z = df_dz(x, y, z)  # Should be 3.0
    grad_all = df_dxyz(x, y, z)  # Should be (5.0, 4.0, 3.0)

    return {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'grad_z': grad_z,
        'grad_all': grad_all,
        'expected': (5.0, 4.0, 3.0)
    }


# =============================================================================
# Exercise 4: Higher-Order Derivatives with Nested grad
# =============================================================================
def exercise_higher_order():
    """
    Nested grad calls compute higher-order derivatives.
    grad(grad(f)) gives the second derivative.

    Tasks:
    1. Define f(x) = x^3

    2. Compute first, second, and third derivatives at x=2
       Hint: df = grad(f), d2f = grad(grad(f)), d3f = grad(grad(grad(f)))

       For f(x) = x^3:
       - f'(x) = 3x^2, so f'(2) = 12
       - f''(x) = 6x, so f''(2) = 12
       - f'''(x) = 6, so f'''(2) = 6

    3. Verify that derivatives of exp(x) equal exp(x)

    Returns:
        dict with keys:
        - 'cubic_first_deriv': first derivative at x=2 (should be 12.0)
        - 'cubic_second_deriv': second derivative at x=2 (should be 12.0)
        - 'cubic_third_deriv': third derivative at x=2 (should be 6.0)
        - 'exp_value': exp(1)
        - 'exp_first_deriv': d/dx exp(x) at x=1
        - 'exp_second_deriv': d^2/dx^2 exp(x) at x=1
        - 'exp_derivs_equal': bool, whether exp and its derivative are equal
    """
    # TODO: Implement f(x) = x^3
    def f(x):
        # TODO: Implement this function
        return x**3

    # TODO: Create first, second, and third derivative functions
    df = grad(f)     # First derivative
    d2f = grad(df)    # Second derivative
    d3f = grad(d2f)  # Third derivative

    x = 2.0
    # TODO: Compute derivatives at x=2
    first = df(2.0)   # Should be 12.0
    second = d2f(2.0)  # Should be 12.0
    third = d3f(2.0)   # Should be 6.0

    # TODO: Implement exp_fn(x) = exp(x)
    def exp_fn(x):
        # TODO: Implement this function
        return jnp.exp(x)

    # TODO: Create derivative functions for exp
    d_exp = grad(exp_fn)
    d2_exp = grad(d_exp)

    x2 = 1.0
    # TODO: Compute exp and its derivatives at x=1
    exp_val = exp_fn(x2)
    d_exp_val = d_exp(x2)
    d2_exp_val = d2_exp(x2)

    return {
        'cubic_first_deriv': first,
        'cubic_second_deriv': second,
        'cubic_third_deriv': third,
        'exp_value': exp_val,
        'exp_first_deriv': d_exp_val,
        'exp_second_deriv': d2_exp_val,
        'exp_derivs_equal': jnp.allclose(exp_val, d_exp_val) if exp_val is not None else False
    }


# =============================================================================
# Exercise 5: Jacobian with jacfwd and jacrev
# =============================================================================
def exercise_jacobian():
    """
    For vector-valued functions, use Jacobian instead of gradient.
    - jacfwd: forward-mode (efficient for narrow Jacobians)
    - jacrev: reverse-mode (efficient for wide Jacobians)

    Jacobian[i,j] = df_i / dx_j

    Tasks:
    1. Define f: R^2 -> R^3
       f(x) = [x[0]^2, x[0]*x[1], sin(x[1])]

    2. Compute Jacobian using both jacfwd and jacrev
       Hint: jac = jax.jacfwd(f)(x)

    3. Verify they give the same result

    Expected Jacobian at x = [2, pi/2]:
    [[4, 0],        # df1/dx0 = 2*x0 = 4, df1/dx1 = 0
     [pi/2, 2],     # df2/dx0 = x1 = pi/2, df2/dx1 = x0 = 2
     [0, 0]]        # df3/dx0 = 0, df3/dx1 = cos(pi/2) = 0

    Returns:
        dict with keys:
        - 'jacobian_fwd': Jacobian computed with jacfwd (shape: 3x2)
        - 'jacobian_rev': Jacobian computed with jacrev (shape: 3x2)
        - 'both_same': bool, whether both methods give same result
        - 'matches_expected': bool, whether result matches expected
    """
    # TODO: Implement f: R^2 -> R^3
    def f(x):
        # TODO: Implement this function
        # Return array of [x[0]^2, x[0]*x[1], sin(x[1])]
        return jnp.array([x[0]**2, x[0]*x[1], jnp.sin(x[1])])

    x = jnp.array([2.0, jnp.pi / 2])

    # TODO: Compute Jacobian using forward and reverse mode
    jac_fwd = jax.jacfwd(f)(x)  # Hint: jax.jacfwd(f)(x)
    jac_rev = jax.jacrev(f)(x)  # Hint: jax.jacrev(f)(x)

    expected = jnp.array([
        [4.0, 0.0],
        [jnp.pi / 2, 2.0],
        [0.0, 0.0]
    ])

    return {
            'jacobian_fwd': jac_fwd,
        'jacobian_rev': jac_rev,
        'both_same': jnp.allclose(jac_fwd, jac_rev) if jac_fwd is not None else False,
        'matches_expected': jnp.allclose(jac_fwd, expected, atol=1e-5) if jac_fwd is not None else False
    }


# =============================================================================
# Exercise 6: Hessian Computation
# =============================================================================
def exercise_hessian():
    """
    Hessian is the matrix of second derivatives.
    Hessian[i,j] = d^2f / (dx_i dx_j)

    Compute using jacfwd(jacrev(f)) or jacrev(jacfwd(f)).

    Tasks:
    1. Define f(x) = x0^2 + x0*x1 + x1^2

    2. Compute Hessian using three methods:
       - jacfwd(grad(f))
       - jacrev(grad(f))
       - jax.hessian(f)

    Expected Hessian (constant, independent of x):
    [[2, 1],
     [1, 2]]

    Returns:
        dict with keys:
        - 'hessian_jacfwd': Hessian via jacfwd(grad(f))
        - 'hessian_jacrev': Hessian via jacrev(grad(f))
        - 'hessian_direct': Hessian via jax.hessian
        - 'all_same': bool, whether all methods give same result
        - 'matches_expected': bool, whether result matches expected
    """
    # TODO: Implement f(x) = x0^2 + x0*x1 + x1^2
    def f(x):
        # TODO: Implement this function
        return x[0]**2 + x[0]*x[1] + x[1]**2


    # TODO: Create Hessian functions using three methods
    hessian_fn = jax.jacfwd(jax.grad(f))   # Hint: jax.jacfwd(jax.grad(f))
    hessian_fn2 = jax.jacrev(jax.grad(f))  # Hint: jax.jacrev(jax.grad(f))
    hessian_fn3 = jax.hessian(f)  # Hint: jax.hessian(f)

    x = jnp.array([1.0, 2.0])

    # TODO: Compute Hessians
    hess1 = hessian_fn(x)
    hess2 = hessian_fn2(x)
    hess3 = hessian_fn3(x)

    expected = jnp.array([[2.0, 1.0], [1.0, 2.0]])

    return {
        'hessian_jacfwd': hess1,
        'hessian_jacrev': hess2,
        'hessian_direct': hess3,
        'all_same': (jnp.allclose(hess1, hess2) and jnp.allclose(hess2, hess3)) if hess1 is not None else False,
        'matches_expected': jnp.allclose(hess1, expected) if hess1 is not None else False
    }


# =============================================================================
# Exercise 7: grad Through Control Flow
# =============================================================================
def exercise_grad_control_flow():
    """
    JAX can differentiate through control flow!
    The gradient follows the path taken during the forward pass.

    Tasks:
    1. Implement ReLU using jax.lax.cond
       Hint: jax.lax.cond(condition, true_fn, false_fn)

    2. Implement absolute value using jax.lax.cond

    3. Implement power function using a loop and differentiate it

    Returns:
        dict with keys:
        - 'relu_grad_positive': gradient of ReLU at x=2 (should be 1.0)
        - 'relu_grad_negative': gradient of ReLU at x=-2 (should be 0.0)
        - 'abs_grad_positive': gradient of |x| at x=3 (should be 1.0)
        - 'abs_grad_negative': gradient of |x| at x=-3 (should be -1.0)
        - 'power_grad': gradient of x^4 at x=2 (should be 32.0)
        - 'expected_power_grad': 32.0
    """
    # TODO: Implement ReLU using jax.lax.cond
    def relu(x):
        """ReLU: max(0, x) using control flow."""
        # TODO: Implement this function
        # Hint: return jax.lax.cond(x > 0, lambda: x, lambda: 0.0)
        return jax.lax.cond(x > 0, lambda x: x, lambda x: 0.0, x)

    # TODO: Compute ReLU gradients
    d_relu = grad(relu)  # Hint: grad(relu)
    grad_positive = d_relu(12.0)  # Should be 1.0
    grad_negative = d_relu(-1.0)  # Should be 0.0

    # TODO: Implement absolute value using jax.lax.cond
    def abs_fn(x):
        # TODO: Implement this function
        # Hint: return x if x >= 0, else -x
        return jax.lax.cond(x >= 0, lambda x: x, lambda x: -x, x)   

    # TODO: Compute abs gradients
    d_abs = grad(abs_fn)
    grad_pos = d_abs(12.0)   # Should be 1.0
    grad_neg = d_abs(-1.0)   # Should be -1.0

    # TODO: Implement power function using a loop
    def power_fn(x, n):
        """Compute x^n using a loop."""
        # TODO: Implement this function
        # Hint: multiply x by itself n times
        total = 1.0
        for i in range(n):
            total *= x
        return total

    # TODO: Compute power gradient
    d_power = grad(power_fn)  # Hint: grad(power_fn)
    # d/dx(x^4) = 4x^3, at x=2: 4 * 8 = 32
    grad_power = d_power(2.0, 4)  # Should be 32.0

    return {
        'relu_grad_positive': grad_positive,
        'relu_grad_negative': grad_negative,
        'abs_grad_positive': grad_pos,
        'abs_grad_negative': grad_neg,
        'power_grad': grad_power,
        'expected_power_grad': 32.0
    }


# =============================================================================
# Exercise 8: Custom Gradients with custom_vjp
# =============================================================================
def exercise_custom_gradient():
    """
    Define custom backward passes using jax.custom_vjp.
    Useful when:
    - The automatic gradient is numerically unstable
    - You want to checkpoint for memory efficiency
    - Implementing custom ops

    Tasks:
    1. Implement safe_sqrt with custom gradient
       - Forward: compute sqrt(x)
       - Backward: use 0.5/sqrt(x), but clip to 0 for x near 0

    2. Implement round with straight-through estimator
       - Forward: round(x)
       - Backward: gradient passes through unchanged

    Returns:
        dict with keys:
        - 'safe_sqrt_grad_normal': gradient at x=4 (should be 0.25)
        - 'safe_sqrt_grad_small': gradient at x=1e-12 (should be 0.0, clipped)
        - 'round_ste_grad': gradient of 2*round(x) at x=2.7 (should be 2.0)
        - 'expected_normal_grad': 0.25
    """
    # TODO: Implement safe_sqrt with custom gradient
    @jax.custom_vjp
    def safe_sqrt(x):
        """Square root with custom gradient to handle x near 0."""
        # TODO: Implement this function
        return jnp.sqrt(x)

    def safe_sqrt_fwd(x):
        """Forward pass: compute output and save residuals."""
        # TODO: Implement this function
        # Return (output, residuals_for_backward)
        # Hint: y = jnp.sqrt(x); return y, (x, y)
        y = safe_sqrt(x)
        return y, (x, y)

    def safe_sqrt_bwd(res, g):
        """Backward pass: compute gradient using residuals."""
        # TODO: Implement this function
        # Use jnp.where to clip gradient near 0
        # Hint: grad_x = g * jnp.where(x > 1e-10, 0.5 / y, 0.0)
        return (g * jnp.where(res[0] > 1e-10, 0.5 / res[1], 0.0), )

    # TODO: Register the custom VJP
    # Hint: safe_sqrt.defvjp(safe_sqrt_fwd, safe_sqrt_bwd)
    safe_sqrt.defvjp(safe_sqrt_fwd, safe_sqrt_bwd)

    # TODO: Test safe_sqrt gradients
    x_normal = 4.0
    grad_normal = grad(safe_sqrt)(x_normal)  # Should be 0.25

    x_small = 1e-12
    grad_small = grad(safe_sqrt)(x_small)   # Should be 0.0

    # TODO: Implement round with straight-through estimator
    @jax.custom_vjp
    def round_ste(x):
        """Round with straight-through estimator gradient."""
        # TODO: Implement this function
        return 2*jnp.round(x)

    def round_ste_fwd(x):
        # TODO: Implement this function
        return round_ste(x), x

    def round_ste_bwd(_, g):
        # TODO: Implement this function
        # Gradient passes through unchanged
        return (2*g,)

    # TODO: Register the custom VJP and compute gradient
    round_ste.defvjp(round_ste_fwd, round_ste_bwd)

    x_round = 2.7
    grad_round = grad(round_ste)(x_round)   # Should be 2.0

    return {
        'safe_sqrt_grad_normal': grad_normal,
        'safe_sqrt_grad_small': grad_small,
        'round_ste_grad': grad_round,
        'expected_normal_grad': 0.25
    }


# =============================================================================
# Exercise 9: Stop Gradient with stop_gradient
# =============================================================================
def exercise_stop_gradient():
    """
    jax.lax.stop_gradient prevents gradients from flowing through.
    Useful for:
    - Target networks in RL
    - Detaching values in complex loss functions
    - Implementing non-differentiable operations

    Tasks:
    1. Implement a loss function with a stopped baseline
       - The baseline should not receive gradients

    2. Implement f(x) = x^2 + stop_gradient(x)
       - Only x^2 should contribute to the gradient
       - Gradient should be 2x, not 2x + 1

    Returns:
        dict with keys:
        - 'grad_params': gradient of loss w.r.t. params
        - 'stopped_grad': gradient of f at x=3 (should be 6.0, not 7.0)
        - 'expected_stopped': 6.0
    """
    # TODO: Implement loss with stopped baseline
    def loss_with_baseline(params, x, baseline_params):
        """Loss with non-trainable baseline."""
        # TODO: Implement this function
        # pred = params * x
        # baseline = jax.lax.stop_gradient(baseline_params * x)
        # advantage = pred - baseline
        # return mean of advantage^2
    
        pred = params * x
        baseline = jax.lax.stop_gradient(baseline_params * x)
        advantage = pred - baseline

        return jnp.mean(advantage**2)

    # TODO: Create gradient function and compute gradient
    grad_fn = grad(loss_with_baseline)  # Hint: grad(loss_with_baseline)

    params = 2.0
    x = jnp.array([1.0, 2.0, 3.0])
    baseline_params = 1.5

    grad_params = grad_fn(params, x, baseline_params)  # Compute gradient

    # TODO: Implement f(x) = x^2 + stop_gradient(x)
    def f(x):
        # TODO: Implement this function
        # Only x^2 contributes to gradient
        sqr = x**2 
        total = sqr + jax.lax.stop_gradient(x) 
        return total

    # TODO: Compute gradient of f at x=3
    df = grad(f)
    x_test = 3.0
    grad_f = df(x_test)  # Should be 6.0 (not 7.0)

    return {
        'grad_params': grad_params,
        'stopped_grad': grad_f,
        'expected_stopped': 6.0
    }


# =============================================================================
# Exercise 10: Gradient of Loss Functions
# =============================================================================
def exercise_loss_gradients():
    """
    Computing gradients of common loss functions.
    These patterns are fundamental for training neural networks.

    Tasks:
    1. Compute gradient of MSE loss w.r.t. predictions
       MSE = mean((pred - target)^2)
       Gradient = 2 * (pred - target) / n

    2. Compute gradient of cross-entropy loss w.r.t. logits
       CE = -sum(labels * log_softmax(logits))
       Gradient = softmax(logits) - labels

    3. Compute gradients for a simple linear model

    Returns:
        dict with keys:
        - 'mse_grad': gradient of MSE w.r.t. predictions
        - 'expected_mse_grad': analytical gradient
        - 'mse_grads_match': bool
        - 'ce_grad': gradient of cross-entropy w.r.t. logits
        - 'expected_ce_grad': analytical gradient
        - 'ce_grads_match': bool
        - 'param_grads_w_shape': shape of weight gradient
        - 'param_grads_b_shape': shape of bias gradient
    """
    # TODO: Implement MSE loss
    def mse_loss(pred, target):
        return jnp.mean((pred - target)**2)

    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.0, 2.5])

    # TODO: Compute MSE gradient
    grad_mse = grad(mse_loss)(pred, target) # Hint: grad(mse_loss)(pred, target)
    # Analytical gradient: 2 * (pred - target) / n
    expected_mse_grad = 2 * (pred - target) / len(pred)

    # TODO: Implement cross-entropy loss
    def cross_entropy_loss(logits, labels):
        """Cross-entropy with softmax."""
        # TODO: Implement this function
        # log_probs = jax.nn.log_softmax(logits)
        # return -sum(labels * log_probs)
        
        log_probs = jax.nn.log_softmax(logits)

        return -jnp.sum(labels * log_probs)

    logits = jnp.array([1.0, 2.0, 0.5])
    labels = jnp.array([0.0, 1.0, 0.0])  # One-hot: class 1

    # TODO: Compute cross-entropy gradient
    grad_ce = grad(cross_entropy_loss)(logits, labels)  # Hint: grad(cross_entropy_loss)(logits, labels)
    # Analytical gradient: softmax(logits) - labels
    expected_ce_grad = jax.nn.softmax(logits) - labels

    # TODO: Implement simple linear model
    def model(params, x):
        """Simple linear model."""
        # TODO: Implement this function
        # return params['w'] @ x + params['b']
        return params["w"] @ x + params["b"]

    def loss_fn(params, x, y):
        # TODO: Implement this function
        # pred = model(params, x)
        # return mse_loss(pred, y)
        pred = model(params, x)
        return mse_loss(pred, y)

    params = {
        'w': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'b': jnp.array([0.0, 0.0])
    }
    x = jnp.array([1.0, 1.0])
    y = jnp.array([3.0, 7.0])

    # TODO: Compute gradients for all parameters
    grads = grad(loss_fn)(params, x, y)  # Hint: grad(loss_fn)(params, x, y)

    return {
        'mse_grad': grad_mse,
        'expected_mse_grad': expected_mse_grad,
        'mse_grads_match': jnp.allclose(grad_mse, expected_mse_grad) if grad_mse is not None else False,
        'ce_grad': grad_ce,
        'expected_ce_grad': expected_ce_grad,
        'ce_grads_match': jnp.allclose(grad_ce, expected_ce_grad) if grad_ce is not None else False,
        'param_grads_w_shape': grads['w'].shape if grads is not None else None,
        'param_grads_b_shape': grads['b'].shape if grads is not None else None
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Automatic Differentiation Exercises")
    print("=" * 60)

    exercises = [
        ("1. Basic grad", exercise_basic_grad),
        ("2. value_and_grad", exercise_value_and_grad),
        ("3. argnums", exercise_argnums),
        ("4. Higher-Order Derivatives", exercise_higher_order),
        ("5. Jacobian", exercise_jacobian),
        ("6. Hessian", exercise_hessian),
        ("7. grad Through Control Flow", exercise_grad_control_flow),
        ("8. Custom Gradients", exercise_custom_gradient),
        ("9. Stop Gradient", exercise_stop_gradient),
        ("10. Loss Gradients", exercise_loss_gradients),
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
