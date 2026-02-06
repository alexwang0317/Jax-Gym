"""
Tests for JAX JIT Examples
==========================
"""

import pytest
import jax
import jax.numpy as jnp
from functools import partial

from examples import (
    example_basic_jit,
    example_timing_comparison,
    example_static_argnums,
    example_donate_argnums,
    example_jit_caching,
    example_debugging,
    example_make_jaxpr,
    example_side_effects,
    example_pure_functions,
    example_dynamic_shapes,
)


class TestBasicJit:
    def test_results_match(self):
        result = example_basic_jit()
        assert result['results_match'] == True

    def test_jit_produces_valid_output(self):
        @jax.jit
        def fn(x):
            return x * 2

        x = jnp.array([1.0, 2.0, 3.0])
        result = fn(x)
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result, expected)


class TestTimingComparison:
    def test_results_match(self):
        result = example_timing_comparison()
        assert result['results_match'] == True

    def test_speedup_positive(self):
        result = example_timing_comparison()
        # JIT should provide some speedup (or at least not be slower)
        assert result['speedup'] >= 0.5  # Allow some variance


class TestStaticArgnums:
    def test_repeat_values(self):
        result = example_static_argnums()
        # repeat_sum(x, 3) should be x + x + x + x = 4x (original + 3 additions)
        x = jnp.array([1.0, 2.0, 3.0])
        expected_3 = x * 4  # x + 3*x
        assert jnp.allclose(result['repeat_3'], expected_3)

        expected_5 = x * 6  # x + 5*x
        assert jnp.allclose(result['repeat_5'], expected_5)

    def test_dynamic_reduce(self):
        result = example_static_argnums()
        # sum_rows: sum along axis 0 of (3, 4) ones -> (4,)
        assert result['sum_rows'].shape == (4,)
        assert jnp.allclose(result['sum_rows'], jnp.array([3., 3., 3., 3.]))

        # sum_cols: sum along axis 1 of (3, 4) ones -> (3,)
        assert result['sum_cols'].shape == (3,)
        assert jnp.allclose(result['sum_cols'], jnp.array([4., 4., 4.]))


class TestDonateArgnums:
    def test_shapes_preserved(self):
        result = example_donate_argnums()
        assert result['result_shape'] == (1000, 1000)
        assert result['new_params_shape'] == (1000,)


class TestJitCaching:
    def test_caching_behavior(self):
        result = example_jit_caching()
        # First call compiles
        assert result['compiles_for_first'] == 1
        # Same shape uses cache
        assert result['compiles_after_same_shape'] == 1
        # Different shape recompiles
        assert result['compiles_after_diff_shape'] == 2
        # Different dtype recompiles
        assert result['compiles_after_diff_dtype'] == 3


class TestDebugging:
    def test_results_match(self):
        result = example_debugging()
        assert result['results_match'] == True

    def test_disable_jit_context(self):
        @jax.jit
        def fn(x):
            return x + 1

        x = jnp.array([1.0, 2.0])

        # Normal
        result_normal = fn(x)

        # With disable_jit
        with jax.disable_jit():
            result_disabled = fn(x)

        assert jnp.allclose(result_normal, result_disabled)


class TestMakeJaxpr:
    def test_jaxpr_generation(self):
        result = example_make_jaxpr()
        # Check that jaxpr was generated
        assert len(result['simple_jaxpr']) > 0
        assert result['mlp_jaxpr_lines'] > 0


class TestSideEffects:
    def test_trace_count(self):
        result = example_side_effects()
        # Function should only trace once for same shape
        assert result['trace_count'] == result['expected_traces']


class TestPureFunctions:
    def test_pure_function_results(self):
        result = example_pure_functions()
        # pure_fn(1.0, 0) -> 1.0 + 1 = 2.0
        # pure_fn(1.0, 1) -> 1.0 + 2 = 3.0
        # pure_fn(1.0, 2) -> 1.0 + 3 = 4.0
        assert result['pure_results'] == [2.0, 3.0, 4.0]

    def test_jit_compatible(self):
        result = example_pure_functions()
        # JIT version with counter=0 should give 2.0
        assert result['jit_result'] == 2.0


class TestDynamicShapes:
    def test_masked_mean(self):
        result = example_dynamic_shapes()
        assert result['padded_shape'] == (64, 10)
        assert result['masked_mean_shape'] == (10,)

    def test_masking_strategy(self):
        @jax.jit
        def masked_mean(x, mask):
            masked = x * mask[:, None]
            return jnp.sum(masked, axis=0) / jnp.sum(mask)

        x = jnp.ones((10, 5))
        mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = masked_mean(x, mask)
        # Mean of first 3 rows of ones should be ones
        assert jnp.allclose(result, jnp.ones(5))


class TestJitEdgeCases:
    def test_nested_jit(self):
        """Nested JIT calls should work correctly."""
        @jax.jit
        def inner(x):
            return x * 2

        @jax.jit
        def outer(x):
            return inner(x) + 1

        x = jnp.array([1.0, 2.0, 3.0])
        result = outer(x)
        expected = jnp.array([3.0, 5.0, 7.0])
        assert jnp.allclose(result, expected)

    def test_jit_with_kwargs(self):
        """JIT should handle keyword arguments."""
        @jax.jit
        def fn(x, scale=2.0):
            return x * scale

        x = jnp.array([1.0, 2.0])
        result = fn(x, scale=3.0)
        expected = jnp.array([3.0, 6.0])
        assert jnp.allclose(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
