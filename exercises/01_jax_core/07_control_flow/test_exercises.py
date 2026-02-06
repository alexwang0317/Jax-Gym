"""
Tests for JAX Control Flow Exercises
====================================
"""

import pytest
import jax
import jax.numpy as jnp
from jax import lax

from exercises import (
    exercise_lax_cond,
    exercise_lax_switch,
    exercise_while_loop,
    exercise_fori_loop,
    exercise_nested_control_flow,
    exercise_control_flow_grad,
    exercise_python_vs_jax,
    exercise_lax_select,
    exercise_early_stopping,
    exercise_performance,
)


class TestLaxCond:
    def test_safe_divide(self):
        result = exercise_lax_cond()
        assert jnp.allclose(result['divide_normal'], 5.0)
        assert jnp.allclose(result['divide_zero'], 0.0)

    def test_abs_value(self):
        result = exercise_lax_cond()
        assert jnp.allclose(result['abs_negative'], 5.0)
        assert jnp.allclose(result['abs_positive'], 3.0)

    def test_relu(self):
        result = exercise_lax_cond()
        assert jnp.allclose(result['relu_negative'], 0.0)


class TestLaxSwitch:
    def test_activations(self):
        result = exercise_lax_switch()
        x = jnp.array([-1.0, 0.0, 1.0])

        # Linear: identity
        assert jnp.allclose(result['linear'], x)

        # ReLU: max(0, x)
        assert jnp.allclose(result['relu'], jnp.array([0.0, 0.0, 1.0]))

        # Tanh
        assert jnp.allclose(result['tanh'], jnp.tanh(x))

    def test_operations(self):
        result = exercise_lax_switch()
        assert jnp.allclose(result['add_result'], 6.0)
        assert jnp.allclose(result['multiply_result'], 10.0)
        assert jnp.allclose(result['square_result'], 25.0)


class TestWhileLoop:
    def test_countdown(self):
        result = exercise_while_loop()
        assert result['countdown'] == 0

    def test_sum_threshold(self):
        result = exercise_while_loop()
        # 1+2+3 = 6 > 5, so 3 elements
        assert result['sum_elements'] == 3

    def test_newton_sqrt(self):
        result = exercise_while_loop()
        assert jnp.allclose(result['newton_sqrt_9'], 3.0, atol=1e-5)


class TestForiLoop:
    def test_sum(self):
        result = exercise_fori_loop()
        # 0+1+2+...+9 = 45
        assert result['sum_0_to_9'] == 45

    def test_running_mean(self):
        result = exercise_fori_loop()
        assert jnp.allclose(result['running_mean'], 3.0)

    def test_cumsum(self):
        result = exercise_fori_loop()
        expected = jnp.array([1.0, 3.0, 6.0, 10.0, 15.0])
        assert jnp.allclose(result['cumsum'], expected)


class TestNestedControlFlow:
    def test_nested_cond(self):
        result = exercise_nested_control_flow()
        # All cases should give 5 (by design)
        assert jnp.allclose(result['both_pos'], 5.0)
        assert jnp.allclose(result['x_pos_y_neg'], 5.0)
        assert jnp.allclose(result['x_neg_y_pos'], 5.0)
        assert jnp.allclose(result['both_neg'], 5.0)

    def test_conditional_sum(self):
        result = exercise_nested_control_flow()
        # Sum of values > 3: 5 + 8 = 13
        assert jnp.allclose(result['conditional_sum'], 13.0)


class TestControlFlowGrad:
    def test_branch_gradients(self):
        result = exercise_control_flow_grad()
        # d/dx(x^2) at x=3 is 6
        assert jnp.allclose(result['grad_positive_branch'], 6.0)
        # d/dx(-x) is -1
        assert jnp.allclose(result['grad_negative_branch'], -1.0)

    def test_loop_gradient(self):
        result = exercise_control_flow_grad()
        # d/dx(x^3) at x=2 is 3*4 = 12
        assert jnp.allclose(result['grad_power_iteration'], 12.0)

    def test_max_gradient(self):
        result = exercise_control_flow_grad()
        assert jnp.allclose(result['grad_max_x_larger'], 1.0)
        assert jnp.allclose(result['grad_max_x_smaller'], 0.0)


class TestPythonVsJax:
    def test_python_branching(self):
        result = exercise_python_vs_jax()
        assert jnp.allclose(result['python_squared'], jnp.array([1.0, 4.0, 9.0]))
        assert jnp.allclose(result['python_cubed'], jnp.array([1.0, 8.0, 27.0]))

    def test_jax_branching(self):
        result = exercise_python_vs_jax()
        assert jnp.allclose(result['jax_dynamic_branch'], 25.0)

    def test_loops_match(self):
        result = exercise_python_vs_jax()
        assert jnp.allclose(result['python_static_loop'], result['jax_dynamic_loop'])


class TestLaxSelect:
    def test_relu(self):
        result = exercise_lax_select()
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert jnp.allclose(result['relu'], expected)

    def test_clipped(self):
        result = exercise_lax_select()
        expected = jnp.array([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert jnp.allclose(result['clipped'], expected)

    def test_where_matches(self):
        result = exercise_lax_select()
        assert result['where_matches'] == True


class TestEarlyStopping:
    def test_find_first(self):
        result = exercise_early_stopping()
        # First element > 4 is at index 2 (value 5.0)
        assert result['first_above_4'] == 2
        assert result['first_above_10'] == -1

    def test_convergence(self):
        result = exercise_early_stopping()
        # cos(x) = x has solution ~0.739
        assert jnp.allclose(result['fixed_point'], 0.739, atol=1e-3)

    def test_gradient_descent(self):
        result = exercise_early_stopping()
        # Minimum of (x-3)^2 is at x=3
        assert jnp.allclose(result['gd_minimum'], 3.0, atol=1e-3)


class TestPerformance:
    def test_results_match(self):
        result = exercise_performance()
        assert result['results_match'] == True


class TestControlFlowEdgeCases:
    def test_cond_with_pytree(self):
        """cond works with pytree return values."""
        def f(x):
            return lax.cond(
                x > 0,
                lambda: {'a': x, 'b': x * 2},
                lambda: {'a': -x, 'b': -x * 2}
            )

        result = f(jnp.array(3.0))
        assert jnp.allclose(result['a'], 3.0)
        assert jnp.allclose(result['b'], 6.0)

    def test_nested_loops(self):
        """Nested loops work correctly."""
        def nested_sum(n, m):
            def outer_body(i, total):
                def inner_body(j, inner_total):
                    return inner_total + i * j

                return lax.fori_loop(0, m, inner_body, total)

            return lax.fori_loop(0, n, outer_body, 0)

        result = nested_sum(3, 4)
        # Sum of i*j for i in [0,1,2], j in [0,1,2,3]
        expected = sum(i * j for i in range(3) for j in range(4))
        assert result == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
