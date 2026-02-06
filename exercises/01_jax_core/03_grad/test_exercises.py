"""
Tests for JAX Automatic Differentiation Exercises
==================================================

Run with: pytest test_exercises.py -v
"""

import pytest
import jax
import jax.numpy as jnp

from exercises import (
    exercise_basic_grad,
    exercise_value_and_grad,
    exercise_argnums,
    exercise_higher_order,
    exercise_jacobian,
    exercise_hessian,
    exercise_grad_control_flow,
    exercise_custom_gradient,
    exercise_stop_gradient,
    exercise_loss_gradients,
)


class TestBasicGrad:
    def test_x_squared(self):
        """Test: d/dx(x^2) at x=3 should be 2*3 = 6"""
        result = exercise_basic_grad()
        assert result['x_squared_grad_at_3'] is not None, \
            "gradient is None - did you forget to compute df(x)?"
        assert jnp.allclose(result['x_squared_grad_at_3'], 6.0), \
            f"Expected gradient of x^2 at x=3 to be 6.0, got {result['x_squared_grad_at_3']}"

    def test_sin_cos(self):
        """Test: d/dx(sin(x)*cos(x)) at x=pi/4 should be cos(pi/2) = 0"""
        result = exercise_basic_grad()
        assert result['sin_cos_grad'] is not None, \
            "sin_cos_grad is None - did you implement g(x) = sin(x)*cos(x)?"
        assert jnp.abs(result['sin_cos_grad']) < 1e-5, \
            f"Expected gradient of sin(x)*cos(x) at x=pi/4 to be ~0, got {result['sin_cos_grad']}"

    def test_gaussian(self):
        """Test: d/dx(exp(-x^2)) at x=1 should be -2*exp(-1)"""
        result = exercise_basic_grad()
        assert result['gaussian_grad'] is not None, \
            "gaussian_grad is None - did you implement h(x) = exp(-x^2)?"
        expected = -2 * jnp.exp(-1)
        assert jnp.allclose(result['gaussian_grad'], expected), \
            f"Expected gradient of exp(-x^2) at x=1 to be {expected}, got {result['gaussian_grad']}"


class TestValueAndGrad:
    def test_both_computed(self):
        """Test that value_and_grad computes same results as separate calls"""
        result = exercise_value_and_grad()
        assert result['same_loss'] == True, \
            "value_and_grad should give same loss as calling loss() directly"
        assert result['same_grad'] == True, \
            "value_and_grad should give same gradient as calling grad(loss)() directly"

    def test_optimal_params(self):
        """Test: When w=2 and y=2x, loss and gradient should be 0"""
        result = exercise_value_and_grad()
        assert result['loss'] is not None, \
            "loss is None - did you call loss_and_grad(w, x, y)?"
        assert jnp.allclose(result['loss'], 0.0, atol=1e-6), \
            f"At optimal w=2, loss should be 0, got {result['loss']}"
        assert jnp.allclose(result['gradient'], 0.0, atol=1e-6), \
            f"At optimal w=2, gradient should be 0, got {result['gradient']}"


class TestArgnums:
    def test_individual_gradients(self):
        """Test gradients w.r.t. each argument individually"""
        result = exercise_argnums()
        # f(x,y,z) = xy + yz + zx
        # df/dx = y + z = 2 + 3 = 5
        # df/dy = x + z = 1 + 3 = 4
        # df/dz = y + x = 2 + 1 = 3
        assert result['grad_x'] is not None, \
            "grad_x is None - did you create df_dx = grad(f, argnums=0)?"
        assert jnp.allclose(result['grad_x'], 5.0), \
            f"df/dx should be y+z = 5, got {result['grad_x']}"
        assert jnp.allclose(result['grad_y'], 4.0), \
            f"df/dy should be x+z = 4, got {result['grad_y']}"
        assert jnp.allclose(result['grad_z'], 3.0), \
            f"df/dz should be y+x = 3, got {result['grad_z']}"

    def test_all_gradients(self):
        """Test computing all gradients at once with argnums=(0,1,2)"""
        result = exercise_argnums()
        assert result['grad_all'] is not None, \
            "grad_all is None - did you use grad(f, argnums=(0, 1, 2))?"
        expected = (5.0, 4.0, 3.0)
        for i, (got, exp) in enumerate(zip(result['grad_all'], expected)):
            assert jnp.allclose(got, exp), \
                f"grad_all[{i}] should be {exp}, got {got}"


class TestHigherOrder:
    def test_cubic_derivatives(self):
        """Test derivatives of f(x) = x^3 at x=2"""
        result = exercise_higher_order()
        # f(x) = x^3, at x=2
        # f'(2) = 3*4 = 12
        # f''(2) = 6*2 = 12
        # f'''(2) = 6
        assert result['cubic_first_deriv'] is not None, \
            "cubic_first_deriv is None - did you create df = grad(f)?"
        assert jnp.allclose(result['cubic_first_deriv'], 12.0), \
            f"First derivative of x^3 at x=2 should be 3*2^2 = 12, got {result['cubic_first_deriv']}"
        assert jnp.allclose(result['cubic_second_deriv'], 12.0), \
            f"Second derivative of x^3 at x=2 should be 6*2 = 12, got {result['cubic_second_deriv']}"
        assert jnp.allclose(result['cubic_third_deriv'], 6.0), \
            f"Third derivative of x^3 at x=2 should be 6, got {result['cubic_third_deriv']}"

    def test_exp_derivatives(self):
        """Test that all derivatives of exp(x) equal exp(x)"""
        result = exercise_higher_order()
        assert result['exp_derivs_equal'] == True, \
            "All derivatives of exp(x) should equal exp(x) - check your exp_fn implementation"


class TestJacobian:
    def test_jacobians_equal(self):
        """Test that jacfwd and jacrev give the same Jacobian"""
        result = exercise_jacobian()
        assert result['jacobian_fwd'] is not None, \
            "jacobian_fwd is None - did you use jax.jacfwd(f)(x)?"
        assert result['both_same'] == True, \
            "jacfwd and jacrev should give the same Jacobian"

    def test_matches_expected(self):
        """Test that Jacobian matches expected values"""
        result = exercise_jacobian()
        assert result['matches_expected'] == True, \
            "Jacobian doesn't match expected. Check your function implementation."

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape (3x2 for f: R^2 -> R^3)"""
        result = exercise_jacobian()
        assert result['jacobian_fwd'] is not None, \
            "jacobian_fwd is None"
        assert result['jacobian_fwd'].shape == (3, 2), \
            f"Jacobian should have shape (3, 2), got {result['jacobian_fwd'].shape}"


class TestHessian:
    def test_all_methods_equal(self):
        """Test that all three methods give the same Hessian"""
        result = exercise_hessian()
        assert result['hessian_jacfwd'] is not None, \
            "hessian_jacfwd is None - did you use jax.jacfwd(jax.grad(f))(x)?"
        assert result['all_same'] == True, \
            "All three Hessian methods should give the same result"

    def test_matches_expected(self):
        """Test that Hessian matches expected [[2,1],[1,2]]"""
        result = exercise_hessian()
        assert result['matches_expected'] == True, \
            "Hessian doesn't match expected [[2,1],[1,2]]. Check your f(x) implementation."

    def test_hessian_symmetric(self):
        """Test that Hessian is symmetric (required for smooth functions)"""
        result = exercise_hessian()
        if result['hessian_direct'] is not None:
            hess = result['hessian_direct']
            assert jnp.allclose(hess, hess.T), \
                "Hessian should be symmetric (Hessian[i,j] = Hessian[j,i])"


class TestGradControlFlow:
    def test_relu_gradients(self):
        """Test ReLU gradients: 1 for positive, 0 for negative"""
        result = exercise_grad_control_flow()
        assert result['relu_grad_positive'] is not None, \
            "relu_grad_positive is None - did you implement relu() with jax.lax.cond?"
        assert jnp.allclose(result['relu_grad_positive'], 1.0), \
            f"ReLU gradient at x=2 should be 1.0, got {result['relu_grad_positive']}"
        assert jnp.allclose(result['relu_grad_negative'], 0.0), \
            f"ReLU gradient at x=-2 should be 0.0, got {result['relu_grad_negative']}"

    def test_abs_gradients(self):
        """Test absolute value gradients: 1 for positive, -1 for negative"""
        result = exercise_grad_control_flow()
        assert result['abs_grad_positive'] is not None, \
            "abs_grad_positive is None - did you implement abs_fn() with jax.lax.cond?"
        assert jnp.allclose(result['abs_grad_positive'], 1.0), \
            f"|x| gradient at x=3 should be 1.0, got {result['abs_grad_positive']}"
        assert jnp.allclose(result['abs_grad_negative'], -1.0), \
            f"|x| gradient at x=-3 should be -1.0, got {result['abs_grad_negative']}"

    def test_power_gradient(self):
        """Test: d/dx(x^4) at x=2 should be 4*2^3 = 32"""
        result = exercise_grad_control_flow()
        assert result['power_grad'] is not None, \
            "power_grad is None - did you implement power_fn()?"
        assert jnp.allclose(result['power_grad'], result['expected_power_grad']), \
            f"d/dx(x^4) at x=2 should be 32, got {result['power_grad']}"


class TestCustomGradient:
    def test_safe_sqrt_normal(self):
        """Test: d/dx(sqrt(x)) at x=4 should be 0.5/sqrt(4) = 0.25"""
        result = exercise_custom_gradient()
        assert result['safe_sqrt_grad_normal'] is not None, \
            "safe_sqrt_grad_normal is None - did you implement and register safe_sqrt.defvjp()?"
        assert jnp.allclose(result['safe_sqrt_grad_normal'], 0.25), \
            f"sqrt gradient at x=4 should be 0.25, got {result['safe_sqrt_grad_normal']}"

    def test_safe_sqrt_small(self):
        """Test: Custom gradient clips to 0 near x=0 (prevents infinity)"""
        result = exercise_custom_gradient()
        assert result['safe_sqrt_grad_small'] is not None, \
            "safe_sqrt_grad_small is None"
        assert jnp.allclose(result['safe_sqrt_grad_small'], 0.0), \
            f"Custom sqrt gradient near 0 should be clipped to 0, got {result['safe_sqrt_grad_small']}"

    def test_round_ste(self):
        """Test: Straight-through estimator passes gradient through unchanged"""
        result = exercise_custom_gradient()
        assert result['round_ste_grad'] is not None, \
            "round_ste_grad is None - did you implement round_ste with straight-through gradient?"
        assert jnp.allclose(result['round_ste_grad'], 2.0), \
            f"Gradient of 2*round(x) should be 2 (passes through), got {result['round_ste_grad']}"


class TestStopGradient:
    def test_stopped_gradient(self):
        """Test: f(x) = x^2 + stop_gradient(x) should have gradient 2x, not 2x+1"""
        result = exercise_stop_gradient()
        assert result['stopped_grad'] is not None, \
            "stopped_grad is None - did you use jax.lax.stop_gradient()?"
        assert jnp.allclose(result['stopped_grad'], result['expected_stopped']), \
            f"Gradient of x^2 + stop_grad(x) at x=3 should be 6 (not 7), got {result['stopped_grad']}"


class TestLossGradients:
    def test_mse_gradient(self):
        """Test MSE gradient matches analytical formula"""
        result = exercise_loss_gradients()
        assert result['mse_grad'] is not None, \
            "mse_grad is None - did you implement mse_loss()?"
        assert result['mse_grads_match'] == True, \
            "MSE gradient doesn't match expected. Formula: 2*(pred-target)/n"

    def test_ce_gradient(self):
        """Test cross-entropy gradient matches analytical formula"""
        result = exercise_loss_gradients()
        assert result['ce_grad'] is not None, \
            "ce_grad is None - did you implement cross_entropy_loss()?"
        assert result['ce_grads_match'] == True, \
            "Cross-entropy gradient doesn't match expected. Formula: softmax(logits) - labels"

    def test_param_gradient_shapes(self):
        """Test that parameter gradients have correct shapes"""
        result = exercise_loss_gradients()
        assert result['param_grads_w_shape'] is not None, \
            "param_grads_w_shape is None - did you implement model() and loss_fn()?"
        assert result['param_grads_w_shape'] == (2, 2), \
            f"Weight gradient should have shape (2, 2), got {result['param_grads_w_shape']}"
        assert result['param_grads_b_shape'] == (2,), \
            f"Bias gradient should have shape (2,), got {result['param_grads_b_shape']}"


class TestGradComposition:
    """Additional tests for grad composition with jit"""

    def test_jit_grad(self):
        """Test that JIT and grad compose correctly"""
        def f(x):
            return x ** 2

        jit_grad_f = jax.jit(jax.grad(f))
        grad_jit_f = jax.grad(jax.jit(f))

        x = 3.0
        assert jnp.allclose(jit_grad_f(x), 6.0), \
            "jit(grad(f)) should give same result as grad alone"
        assert jnp.allclose(grad_jit_f(x), 6.0), \
            "grad(jit(f)) should give same result as grad alone"

    def test_grad_of_sum(self):
        """Test: Gradient of sum(x) should be all ones"""
        def f(x):
            return jnp.sum(x)

        df = jax.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(df(x), jnp.ones(3)), \
            "Gradient of sum(x) w.r.t. x should be all ones"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
