"""
JAX Automatic Differentiation (grad) - 10 Examples
====================================================

JAX's grad transformation computes gradients automatically.
This is the foundation for training neural networks.

Key concepts:
- grad(f) returns a function that computes the gradient of f
- Works with scalar-output functions by default
- Can differentiate through control flow
- Composable with other transformations (jit, vmap)

Reference: https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
from functools import partial


# =============================================================================
# Example 1: Basic grad for Scalar Functions
# =============================================================================
def example_basic_grad():
    """
    grad(f) returns a function that computes df/dx.
    f must return a scalar (single number).
    """
    # Simple quadratic: f(x) = x^2
    # Derivative: f'(x) = 2x
    def f(x):
        return x ** 2

    df = grad(f)

    x = 3.0
    gradient = df(x)  # Should be 2 * 3 = 6

    # More complex function
    # f(x) = sin(x) * cos(x)
    # f'(x) = cos(x)*cos(x) - sin(x)*sin(x) = cos(2x)
    def g(x):
        return jnp.sin(x) * jnp.cos(x)

    dg = grad(g)

    x2 = jnp.pi / 4
    gradient2 = dg(x2)  # Should be cos(pi/2) = 0

    # Function with multiple operations
    def h(x):
        return jnp.exp(-x ** 2)  # Gaussian

    dh = grad(h)
    x3 = 1.0
    gradient3 = dh(x3)  # -2x * exp(-x^2) = -2 * exp(-1)

    return {
        'x_squared_grad_at_3': gradient,
        'expected_6': 6.0,
        'sin_cos_grad': gradient2,
        'gaussian_grad': gradient3
    }


# =============================================================================
# Example 2: value_and_grad for Efficiency
# =============================================================================
def example_value_and_grad():
    """
    value_and_grad computes both f(x) and grad(f)(x) in one pass.
    More efficient than calling f(x) and grad(f)(x) separately.
    """
    def loss(w, x, y):
        """Mean squared error loss."""
        pred = w * x
        return jnp.mean((pred - y) ** 2)

    # Get both loss value and gradient
    loss_and_grad = value_and_grad(loss)

    w = 2.0
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([2.0, 4.0, 6.0])  # y = 2x, so optimal w = 2

    loss_val, grad_val = loss_and_grad(w, x, y)

    # Compare with separate calls
    loss_separate = loss(w, x, y)
    grad_separate = grad(loss)(w, x, y)

    return {
        'loss': loss_val,
        'gradient': grad_val,
        'same_loss': jnp.allclose(loss_val, loss_separate),
        'same_grad': jnp.allclose(grad_val, grad_separate)
    }


# =============================================================================
# Example 3: grad with argnums for Multiple Inputs
# =============================================================================
def example_argnums():
    """
    Use argnums to specify which argument(s) to differentiate with respect to.
    Default is argnums=0 (first argument).
    """
    def f(x, y, z):
        """f(x, y, z) = x*y + y*z + z*x"""
        return x * y + y * z + z * x

    # Gradient w.r.t. x (default)
    df_dx = grad(f, argnums=0)

    # Gradient w.r.t. y
    df_dy = grad(f, argnums=1)

    # Gradient w.r.t. z
    df_dz = grad(f, argnums=2)

    # Gradient w.r.t. multiple arguments
    df_dxyz = grad(f, argnums=(0, 1, 2))

    x, y, z = 1.0, 2.0, 3.0

    grad_x = df_dx(x, y, z)   # y + z = 5
    grad_y = df_dy(x, y, z)   # x + z = 4
    grad_z = df_dz(x, y, z)   # y + x = 3

    grad_all = df_dxyz(x, y, z)  # (5, 4, 3)

    return {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'grad_z': grad_z,
        'grad_all': grad_all,
        'expected': (5.0, 4.0, 3.0)
    }


# =============================================================================
# Example 4: Higher-Order Derivatives with Nested grad
# =============================================================================
def example_higher_order():
    """
    Nested grad calls compute higher-order derivatives.
    grad(grad(f)) gives the second derivative.
    """
    # f(x) = x^3
    # f'(x) = 3x^2
    # f''(x) = 6x
    # f'''(x) = 6
    def f(x):
        return x ** 3

    df = grad(f)            # First derivative
    d2f = grad(grad(f))     # Second derivative
    d3f = grad(grad(grad(f)))  # Third derivative

    x = 2.0
    first = df(x)    # 3 * 4 = 12
    second = d2f(x)  # 6 * 2 = 12
    third = d3f(x)   # 6

    # More interesting: Taylor series verification
    # exp(x) has all derivatives = exp(x)
    def exp_fn(x):
        return jnp.exp(x)

    d_exp = grad(exp_fn)
    d2_exp = grad(grad(exp_fn))

    x2 = 1.0
    exp_val = jnp.exp(x2)
    d_exp_val = d_exp(x2)
    d2_exp_val = d2_exp(x2)

    return {
        'cubic_first_deriv': first,
        'cubic_second_deriv': second,
        'cubic_third_deriv': third,
        'exp_value': exp_val,
        'exp_first_deriv': d_exp_val,
        'exp_second_deriv': d2_exp_val,
        'exp_derivs_equal': jnp.allclose(exp_val, d_exp_val)
    }


# =============================================================================
# Example 5: Jacobian with jacfwd and jacrev
# =============================================================================
def example_jacobian():
    """
    For vector-valued functions, use Jacobian instead of gradient.
    - jacfwd: forward-mode (efficient for narrow Jacobians)
    - jacrev: reverse-mode (efficient for wide Jacobians)

    Jacobian[i,j] = df_i / dx_j
    """
    # f: R^2 -> R^3
    def f(x):
        return jnp.array([
            x[0] ** 2,           # df1/dx0 = 2*x0, df1/dx1 = 0
            x[0] * x[1],         # df2/dx0 = x1,   df2/dx1 = x0
            jnp.sin(x[1])        # df3/dx0 = 0,    df3/dx1 = cos(x1)
        ])

    x = jnp.array([2.0, jnp.pi / 2])

    # Forward-mode Jacobian
    jac_fwd = jax.jacfwd(f)(x)

    # Reverse-mode Jacobian
    jac_rev = jax.jacrev(f)(x)

    # They should be the same
    # Expected Jacobian at (2, pi/2):
    # [[4, 0],
    #  [pi/2, 2],
    #  [0, 0]]  (cos(pi/2) = 0)

    expected = jnp.array([
        [4.0, 0.0],
        [jnp.pi / 2, 2.0],
        [0.0, 0.0]
    ])

    return {
        'jacobian_fwd': jac_fwd,
        'jacobian_rev': jac_rev,
        'both_same': jnp.allclose(jac_fwd, jac_rev),
        'matches_expected': jnp.allclose(jac_fwd, expected, atol=1e-5)
    }


# =============================================================================
# Example 6: Hessian Computation
# =============================================================================
def example_hessian():
    """
    Hessian is the matrix of second derivatives.
    Hessian[i,j] = d^2f / (dx_i dx_j)

    Compute using jacfwd(jacrev(f)) or jacrev(jacfwd(f)).
    """
    # f(x) = x0^2 + x0*x1 + x1^2
    # grad = [2*x0 + x1, x0 + 2*x1]
    # Hessian = [[2, 1], [1, 2]]
    def f(x):
        return x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    # Method 1: jacfwd of grad
    hessian_fn = jax.jacfwd(jax.grad(f))

    # Method 2: jacrev of grad
    hessian_fn2 = jax.jacrev(jax.grad(f))

    # Method 3: Use jax.hessian directly
    hessian_fn3 = jax.hessian(f)

    x = jnp.array([1.0, 2.0])

    hess1 = hessian_fn(x)
    hess2 = hessian_fn2(x)
    hess3 = hessian_fn3(x)

    expected = jnp.array([[2.0, 1.0], [1.0, 2.0]])

    return {
        'hessian_jacfwd': hess1,
        'hessian_jacrev': hess2,
        'hessian_direct': hess3,
        'all_same': jnp.allclose(hess1, hess2) and jnp.allclose(hess2, hess3),
        'matches_expected': jnp.allclose(hess1, expected)
    }


# =============================================================================
# Example 7: grad Through Control Flow
# =============================================================================
def example_grad_control_flow():
    """
    JAX can differentiate through control flow!
    The gradient follows the path taken during the forward pass.
    """
    # Using jax.lax.cond for JIT-compatible control flow
    def relu(x):
        """ReLU: max(0, x) using control flow."""
        return jax.lax.cond(x > 0, lambda: x, lambda: 0.0)

    # Gradient of ReLU
    d_relu = grad(relu)

    # Test at positive and negative values
    grad_positive = d_relu(2.0)   # Should be 1
    grad_negative = d_relu(-2.0)  # Should be 0

    # More complex: absolute value
    def abs_fn(x):
        return jax.lax.cond(x >= 0, lambda: x, lambda: -x)

    d_abs = grad(abs_fn)
    grad_pos = d_abs(3.0)   # 1
    grad_neg = d_abs(-3.0)  # -1

    # Looped computation
    def power_fn(x, n):
        """Compute x^n using a loop."""
        result = 1.0
        for _ in range(n):
            result = result * x
        return result

    d_power = grad(power_fn)
    # d/dx(x^4) = 4x^3
    grad_power = d_power(2.0, 4)  # 4 * 8 = 32

    return {
        'relu_grad_positive': grad_positive,
        'relu_grad_negative': grad_negative,
        'abs_grad_positive': grad_pos,
        'abs_grad_negative': grad_neg,
        'power_grad': grad_power,
        'expected_power_grad': 32.0
    }


# =============================================================================
# Example 8: Custom Gradients with custom_vjp
# =============================================================================
def example_custom_gradient():
    """
    Define custom backward passes using jax.custom_vjp.
    Useful when:
    - The automatic gradient is numerically unstable
    - You want to checkpoint for memory efficiency
    - Implementing custom ops
    """
    @jax.custom_vjp
    def safe_sqrt(x):
        """Square root with custom gradient to handle x near 0."""
        return jnp.sqrt(x)

    def safe_sqrt_fwd(x):
        """Forward pass: compute output and save residuals."""
        y = jnp.sqrt(x)
        return y, (x, y)  # Return output and residuals for backward

    def safe_sqrt_bwd(res, g):
        """Backward pass: compute gradient using residuals."""
        x, y = res
        # Normal gradient: 0.5 / sqrt(x) = 0.5 / y
        # But clip to avoid infinity at x=0
        grad_x = g * jnp.where(x > 1e-10, 0.5 / y, 0.0)
        return (grad_x,)

    safe_sqrt.defvjp(safe_sqrt_fwd, safe_sqrt_bwd)

    # Test normal case
    x_normal = 4.0
    grad_normal = grad(safe_sqrt)(x_normal)  # 0.5 / 2 = 0.25

    # Test near-zero case (would be inf with normal sqrt)
    x_small = 1e-12
    grad_small = grad(safe_sqrt)(x_small)  # Clipped to 0

    # Another example: straight-through estimator
    @jax.custom_vjp
    def round_ste(x):
        """Round with straight-through estimator gradient."""
        return jnp.round(x)

    def round_ste_fwd(x):
        return jnp.round(x), ()

    def round_ste_bwd(_, g):
        return (g,)  # Gradient passes through unchanged

    round_ste.defvjp(round_ste_fwd, round_ste_bwd)

    x_round = 2.7
    grad_round = grad(lambda x: round_ste(x) * 2)(x_round)  # 2

    return {
        'safe_sqrt_grad_normal': grad_normal,
        'safe_sqrt_grad_small': grad_small,
        'round_ste_grad': grad_round,
        'expected_normal_grad': 0.25
    }


# =============================================================================
# Example 9: Stop Gradient with stop_gradient
# =============================================================================
def example_stop_gradient():
    """
    jax.lax.stop_gradient prevents gradients from flowing through.
    Useful for:
    - Target networks in RL
    - Detaching values in complex loss functions
    - Implementing non-differentiable operations
    """
    def loss_with_baseline(params, x, baseline_params):
        """Loss with non-trainable baseline."""
        pred = params * x

        # Baseline doesn't receive gradients
        baseline = jax.lax.stop_gradient(baseline_params * x)

        # Advantage: how much better than baseline
        advantage = pred - baseline

        return jnp.mean(advantage ** 2)

    # Gradient only w.r.t. params, not baseline_params
    grad_fn = grad(loss_with_baseline)

    params = 2.0
    x = jnp.array([1.0, 2.0, 3.0])
    baseline_params = 1.5

    grad_params = grad_fn(params, x, baseline_params)

    # Verify baseline_params has no gradient
    grad_fn_both = grad(loss_with_baseline, argnums=(0, 2))
    grad_both = grad_fn_both(params, x, baseline_params)

    # The gradient w.r.t baseline_params should be 0
    # (Actually it won't compute gradient for stopped values)

    # Another example: use value but not gradient
    def f(x):
        # Use x for computation but stop its gradient
        detached_x = jax.lax.stop_gradient(x)
        return x ** 2 + detached_x  # Only x^2 contributes to gradient

    df = grad(f)
    x_test = 3.0
    grad_f = df(x_test)  # Should be 2*x = 6 (not 2*x + 1)

    return {
        'grad_params': grad_params,
        'stopped_grad': grad_f,
        'expected_stopped': 6.0
    }


# =============================================================================
# Example 10: Gradient of Loss Functions
# =============================================================================
def example_loss_gradients():
    """
    Computing gradients of common loss functions.
    These patterns are fundamental for training neural networks.
    """
    # Mean Squared Error
    def mse_loss(pred, target):
        return jnp.mean((pred - target) ** 2)

    # For neural network: loss w.r.t. predictions, then chain rule
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.5, 2.0, 2.5])

    grad_mse = grad(mse_loss)(pred, target)
    # d/d_pred MSE = 2 * (pred - target) / n
    expected_mse_grad = 2 * (pred - target) / len(pred)

    # Cross-entropy loss (for classification)
    def cross_entropy_loss(logits, labels):
        """Cross-entropy with softmax."""
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.sum(labels * log_probs)

    logits = jnp.array([1.0, 2.0, 0.5])
    labels = jnp.array([0.0, 1.0, 0.0])  # One-hot: class 1

    grad_ce = grad(cross_entropy_loss)(logits, labels)
    # Gradient of softmax cross-entropy is: softmax(logits) - labels
    expected_ce_grad = jax.nn.softmax(logits) - labels

    # Full training step example
    def model(params, x):
        """Simple linear model."""
        return params['w'] @ x + params['b']

    def loss_fn(params, x, y):
        pred = model(params, x)
        return mse_loss(pred, y)

    params = {
        'w': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'b': jnp.array([0.0, 0.0])
    }
    x = jnp.array([1.0, 1.0])
    y = jnp.array([3.0, 7.0])

    # Get gradients for all parameters
    grads = grad(loss_fn)(params, x, y)

    return {
        'mse_grad': grad_mse,
        'expected_mse_grad': expected_mse_grad,
        'mse_grads_match': jnp.allclose(grad_mse, expected_mse_grad),
        'ce_grad': grad_ce,
        'expected_ce_grad': expected_ce_grad,
        'ce_grads_match': jnp.allclose(grad_ce, expected_ce_grad),
        'param_grads_w_shape': grads['w'].shape,
        'param_grads_b_shape': grads['b'].shape
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Automatic Differentiation Examples")
    print("=" * 60)

    examples = [
        ("1. Basic grad", example_basic_grad),
        ("2. value_and_grad", example_value_and_grad),
        ("3. argnums", example_argnums),
        ("4. Higher-Order Derivatives", example_higher_order),
        ("5. Jacobian", example_jacobian),
        ("6. Hessian", example_hessian),
        ("7. grad Through Control Flow", example_grad_control_flow),
        ("8. Custom Gradients", example_custom_gradient),
        ("9. Stop Gradient", example_stop_gradient),
        ("10. Loss Gradients", example_loss_gradients),
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
