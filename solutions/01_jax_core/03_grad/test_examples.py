"""
Tests for JAX Automatic Differentiation Examples
================================================
"""

import pytest
import jax
import jax.numpy as jnp

from examples import (
    example_basic_grad,
    example_value_and_grad,
    example_argnums,
    example_higher_order,
    example_jacobian,
    example_hessian,
    example_grad_control_flow,
    example_custom_gradient,
    example_stop_gradient,
    example_loss_gradients,
)


class TestBasicGrad:
    def test_x_squared(self):
        result = example_basic_grad()
        # d/dx(x^2) at x=3 is 2*3 = 6
        assert jnp.allclose(result['x_squared_grad_at_3'], 6.0)

    def test_sin_cos(self):
        result = example_basic_grad()
        # d/dx(sin(x)*cos(x)) = cos(2x), at x=pi/4, cos(pi/2) = 0
        assert jnp.abs(result['sin_cos_grad']) < 1e-5

    def test_gaussian(self):
        result = example_basic_grad()
        # d/dx(exp(-x^2)) = -2x*exp(-x^2), at x=1: -2*exp(-1)
        expected = -2 * jnp.exp(-1)
        assert jnp.allclose(result['gaussian_grad'], expected)


class TestValueAndGrad:
    def test_both_computed(self):
        result = example_value_and_grad()
        assert result['same_loss'] == True
        assert result['same_grad'] == True

    def test_optimal_params(self):
        # When w=2 and y=2x, loss should be 0
        result = example_value_and_grad()
        assert jnp.allclose(result['loss'], 0.0, atol=1e-6)
        assert jnp.allclose(result['gradient'], 0.0, atol=1e-6)


class TestArgnums:
    def test_individual_gradients(self):
        result = example_argnums()
        # f(x,y,z) = xy + yz + zx
        # df/dx = y + z = 2 + 3 = 5
        # df/dy = x + z = 1 + 3 = 4
        # df/dz = y + x = 2 + 1 = 3
        assert jnp.allclose(result['grad_x'], 5.0)
        assert jnp.allclose(result['grad_y'], 4.0)
        assert jnp.allclose(result['grad_z'], 3.0)

    def test_all_gradients(self):
        result = example_argnums()
        expected = (5.0, 4.0, 3.0)
        for got, exp in zip(result['grad_all'], expected):
            assert jnp.allclose(got, exp)


class TestHigherOrder:
    def test_cubic_derivatives(self):
        result = example_higher_order()
        # f(x) = x^3, at x=2
        # f'(2) = 3*4 = 12
        # f''(2) = 6*2 = 12
        # f'''(2) = 6
        assert jnp.allclose(result['cubic_first_deriv'], 12.0)
        assert jnp.allclose(result['cubic_second_deriv'], 12.0)
        assert jnp.allclose(result['cubic_third_deriv'], 6.0)

    def test_exp_derivatives(self):
        result = example_higher_order()
        # All derivatives of exp(x) = exp(x)
        assert result['exp_derivs_equal'] == True


class TestJacobian:
    def test_jacobians_equal(self):
        result = example_jacobian()
        assert result['both_same'] == True

    def test_matches_expected(self):
        result = example_jacobian()
        assert result['matches_expected'] == True

    def test_jacobian_shape(self):
        result = example_jacobian()
        # f: R^2 -> R^3, so Jacobian is 3x2
        assert result['jacobian_fwd'].shape == (3, 2)


class TestHessian:
    def test_all_methods_equal(self):
        result = example_hessian()
        assert result['all_same'] == True

    def test_matches_expected(self):
        result = example_hessian()
        assert result['matches_expected'] == True

    def test_hessian_symmetric(self):
        result = example_hessian()
        hess = result['hessian_direct']
        # Hessian should be symmetric
        assert jnp.allclose(hess, hess.T)


class TestGradControlFlow:
    def test_relu_gradients(self):
        result = example_grad_control_flow()
        assert jnp.allclose(result['relu_grad_positive'], 1.0)
        assert jnp.allclose(result['relu_grad_negative'], 0.0)

    def test_abs_gradients(self):
        result = example_grad_control_flow()
        assert jnp.allclose(result['abs_grad_positive'], 1.0)
        assert jnp.allclose(result['abs_grad_negative'], -1.0)

    def test_power_gradient(self):
        result = example_grad_control_flow()
        assert jnp.allclose(result['power_grad'], result['expected_power_grad'])


class TestCustomGradient:
    def test_safe_sqrt_normal(self):
        result = example_custom_gradient()
        # d/dx(sqrt(x)) at x=4 is 0.5/sqrt(4) = 0.25
        assert jnp.allclose(result['safe_sqrt_grad_normal'], 0.25)

    def test_safe_sqrt_small(self):
        result = example_custom_gradient()
        # Near zero, gradient should be clipped
        assert jnp.allclose(result['safe_sqrt_grad_small'], 0.0)

    def test_round_ste(self):
        result = example_custom_gradient()
        # Straight-through: gradient passes through
        assert jnp.allclose(result['round_ste_grad'], 2.0)


class TestStopGradient:
    def test_stopped_gradient(self):
        result = example_stop_gradient()
        # f(x) = x^2 + stop_grad(x)
        # Only x^2 contributes gradient: 2x at x=3 is 6
        assert jnp.allclose(result['stopped_grad'], result['expected_stopped'])


class TestLossGradients:
    def test_mse_gradient(self):
        result = example_loss_gradients()
        assert result['mse_grads_match'] == True

    def test_ce_gradient(self):
        result = example_loss_gradients()
        assert result['ce_grads_match'] == True

    def test_param_gradient_shapes(self):
        result = example_loss_gradients()
        assert result['param_grads_w_shape'] == (2, 2)
        assert result['param_grads_b_shape'] == (2,)


class TestGradComposition:
    def test_jit_grad(self):
        """JIT and grad compose correctly."""
        def f(x):
            return x ** 2

        jit_grad_f = jax.jit(jax.grad(f))
        grad_jit_f = jax.grad(jax.jit(f))

        x = 3.0
        assert jnp.allclose(jit_grad_f(x), 6.0)
        assert jnp.allclose(grad_jit_f(x), 6.0)

    def test_grad_of_sum(self):
        """Gradient of sum of array equals ones."""
        def f(x):
            return jnp.sum(x)

        df = jax.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(df(x), jnp.ones(3))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
