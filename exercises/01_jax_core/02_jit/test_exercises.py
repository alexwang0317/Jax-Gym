"""
Tests for JAX JIT Exercises
===========================

Run these tests to verify your exercise implementations:
    pytest test_exercises.py -v

Each test class corresponds to one exercise. Tests include helpful
assertion messages to guide you toward the correct implementation.
"""

import pytest
import jax
import jax.numpy as jnp
from functools import partial

from exercises import (
    exercise_basic_jit,
    exercise_timing_comparison,
    exercise_static_argnums,
    exercise_donate_argnums,
    exercise_jit_caching,
    exercise_debugging,
    exercise_make_jaxpr,
    exercise_side_effects,
    exercise_pure_functions,
    exercise_dynamic_shapes,
)


class TestBasicJit:
    """Tests for Exercise 1: Basic @jit Decorator Usage"""

    def test_results_match(self):
        """The JIT and non-JIT versions should produce identical results."""
        result = exercise_basic_jit()
        assert result['results_match'] == True, (
            "JIT and non-JIT results should match. "
            "Use jnp.allclose() to compare result_slow and result_fast."
        )

    def test_slow_result_not_none(self):
        """The slow_fn should return a valid result."""
        result = exercise_basic_jit()
        assert result['slow_result'] is not None, (
            "slow_result should not be None. "
            "Implement slow_fn to return jnp.sin(x) ** 2 + jnp.cos(x) ** 2"
        )

    def test_fast_result_not_none(self):
        """The fast_fn should return a valid result."""
        result = exercise_basic_jit()
        assert result['fast_result'] is not None, (
            "fast_result should not be None. "
            "Implement fast_fn with @jit decorator."
        )

    def test_jit_produces_valid_output(self):
        """Verify JIT produces correct numerical output."""
        @jax.jit
        def fn(x):
            return x * 2

        x = jnp.array([1.0, 2.0, 3.0])
        result = fn(x)
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result, expected), (
            "JIT function should produce correct numerical results."
        )


class TestTimingComparison:
    """Tests for Exercise 2: Timing Comparison"""

    def test_results_match(self):
        """JIT and non-JIT should produce the same numerical result."""
        result = exercise_timing_comparison()
        assert result['results_match'] == True, (
            "JIT and non-JIT results should match. "
            "Make sure matmul_chain returns the correct result."
        )

    def test_times_are_valid(self):
        """Timing values should be positive numbers."""
        result = exercise_timing_comparison()
        assert result['slow_time'] is not None and result['slow_time'] > 0, (
            "slow_time should be a positive number. "
            "Use time.perf_counter() to measure execution time."
        )
        assert result['fast_time'] is not None and result['fast_time'] > 0, (
            "fast_time should be a positive number. "
            "Use time.perf_counter() to measure execution time."
        )

    def test_speedup_calculated(self):
        """Speedup should be calculated as slow_time / fast_time."""
        result = exercise_timing_comparison()
        assert result['speedup'] is not None, (
            "speedup should not be None. "
            "Calculate as slow_time / fast_time."
        )
        # JIT should provide some speedup (or at least not be much slower)
        assert result['speedup'] >= 0.5, (
            "JIT should provide speedup. Make sure you're warming up the JIT "
            "function before timing (call once with .block_until_ready())."
        )


class TestStaticArgnums:
    """Tests for Exercise 3: static_argnums"""

    def test_repeat_values(self):
        """repeat_sum should correctly sum x with itself n times."""
        result = exercise_static_argnums()

        assert result['repeat_3'] is not None, (
            "repeat_3 should not be None. "
            "Use @partial(jit, static_argnums=(1,)) and implement repeat_sum."
        )

        # repeat_sum(x, 3) should be x + x + x + x = 4x (original + 3 additions)
        x = jnp.array([1.0, 2.0, 3.0])
        expected_3 = x * 4  # x + 3*x
        assert jnp.allclose(result['repeat_3'], expected_3), (
            f"repeat_sum([1,2,3], 3) should equal [4,8,12]. Got {result['repeat_3']}. "
            "Start with result = x, then loop n times adding x to result."
        )

        expected_5 = x * 6  # x + 5*x
        assert jnp.allclose(result['repeat_5'], expected_5), (
            f"repeat_sum([1,2,3], 5) should equal [6,12,18]. Got {result['repeat_5']}."
        )

    def test_dynamic_reduce(self):
        """dynamic_reduce should sum along the specified axis."""
        result = exercise_static_argnums()

        assert result['sum_rows'] is not None, (
            "sum_rows should not be None. "
            "Use @partial(jit, static_argnames=['axis']) and implement dynamic_reduce."
        )

        # sum_rows: sum along axis 0 of (3, 4) ones -> (4,)
        assert result['sum_rows'].shape == (4,), (
            f"sum_rows should have shape (4,). Got {result['sum_rows'].shape}. "
            "Sum along axis=0 reduces the first dimension."
        )
        assert jnp.allclose(result['sum_rows'], jnp.array([3., 3., 3., 3.])), (
            "sum_rows of a (3,4) ones matrix along axis=0 should be [3,3,3,3]."
        )

        # sum_cols: sum along axis 1 of (3, 4) ones -> (3,)
        assert result['sum_cols'].shape == (3,), (
            f"sum_cols should have shape (3,). Got {result['sum_cols'].shape}. "
            "Sum along axis=1 reduces the second dimension."
        )
        assert jnp.allclose(result['sum_cols'], jnp.array([4., 4., 4.])), (
            "sum_cols of a (3,4) ones matrix along axis=1 should be [4,4,4]."
        )


class TestDonateArgnums:
    """Tests for Exercise 4: donate_argnums"""

    def test_shapes_preserved(self):
        """Output shapes should match input shapes."""
        result = exercise_donate_argnums()

        assert result['result_shape'] is not None, (
            "result_shape should not be None. "
            "Implement update_in_place with @partial(jit, donate_argnums=(0,))."
        )
        assert result['result_shape'] == (1000, 1000), (
            f"result_shape should be (1000, 1000). Got {result['result_shape']}. "
            "x + y should preserve shape."
        )

        assert result['new_params_shape'] is not None, (
            "new_params_shape should not be None. "
            "Implement update_params with @partial(jit, donate_argnums=(0,))."
        )
        assert result['new_params_shape'] == (1000,), (
            f"new_params_shape should be (1000,). Got {result['new_params_shape']}."
        )


class TestJitCaching:
    """Tests for Exercise 5: JIT Caching"""

    def test_caching_behavior(self):
        """JIT should cache based on shape and dtype."""
        result = exercise_jit_caching()

        # First call compiles
        assert result['compiles_for_first'] == 1, (
            f"Expected 1 compilation for first call. Got {result['compiles_for_first']}. "
            "Make sure to increment compile_count[0] inside traced_fn."
        )

        # Same shape uses cache
        assert result['compiles_after_same_shape'] == 1, (
            f"Expected still 1 compilation after same shape. Got {result['compiles_after_same_shape']}. "
            "Same shape/dtype should use cached compilation."
        )

        # Different shape recompiles
        assert result['compiles_after_diff_shape'] == 2, (
            f"Expected 2 compilations after different shape. Got {result['compiles_after_diff_shape']}. "
            "Call with shape (4,) to trigger recompilation."
        )

        # Different dtype recompiles
        assert result['compiles_after_diff_dtype'] == 3, (
            f"Expected 3 compilations after different dtype. Got {result['compiles_after_diff_dtype']}. "
            "Call with float array to trigger recompilation."
        )


class TestDebugging:
    """Tests for Exercise 6: Debugging with disable_jit"""

    def test_results_match(self):
        """Results should match with and without JIT."""
        result = exercise_debugging()

        assert result['jitted_result'] is not None, (
            "jitted_result should not be None. "
            "Implement complex_fn: a = x * 2, b = sin(a), c = b + 1, return c"
        )

        assert result['debug_result'] is not None, (
            "debug_result should not be None. "
            "Use `with jax.disable_jit():` context manager."
        )

        assert result['results_match'] == True, (
            "JIT and debug results should match. "
            "Use jnp.allclose() to compare."
        )

    def test_disable_jit_context(self):
        """Verify disable_jit context works correctly."""
        @jax.jit
        def fn(x):
            return x + 1

        x = jnp.array([1.0, 2.0])

        result_normal = fn(x)
        with jax.disable_jit():
            result_disabled = fn(x)

        assert jnp.allclose(result_normal, result_disabled), (
            "Results should be identical with and without JIT."
        )


class TestMakeJaxpr:
    """Tests for Exercise 7: Visualizing with make_jaxpr"""

    def test_jaxpr_generation(self):
        """jaxpr should be generated successfully."""
        result = exercise_make_jaxpr()

        assert result['simple_jaxpr'] is not None, (
            "simple_jaxpr should not be None. "
            "Use jax.make_jaxpr(simple_fn)(x, y) to generate jaxpr."
        )
        assert len(result['simple_jaxpr']) > 0, (
            "simple_jaxpr should not be empty. "
            "Convert jaxpr to string with str(jaxpr)."
        )

        assert result['mlp_jaxpr_lines'] is not None, (
            "mlp_jaxpr_lines should not be None. "
            "Use jax.make_jaxpr(matmul_relu)(W, x_vec, b)."
        )
        assert result['mlp_jaxpr_lines'] > 0, (
            "mlp_jaxpr_lines should be positive. "
            "Count lines with len(str(jaxpr).split('\\n'))."
        )


class TestSideEffects:
    """Tests for Exercise 8: Side Effects"""

    def test_trace_count(self):
        """Function should only trace once for same shape inputs."""
        result = exercise_side_effects()

        assert result['trace_count'] == result['expected_traces'], (
            f"Expected {result['expected_traces']} trace(s), got {result['trace_count']}. "
            "Side effects only happen during tracing. "
            "Same shape inputs reuse the cached trace."
        )


class TestPureFunctions:
    """Tests for Exercise 9: Pure Functions"""

    def test_pure_function_results(self):
        """Pure function should thread state correctly."""
        result = exercise_pure_functions()

        # pure_fn(1.0, 0) -> 1.0 + 1 = 2.0
        # pure_fn(1.0, 1) -> 1.0 + 2 = 3.0
        # pure_fn(1.0, 2) -> 1.0 + 3 = 4.0
        expected = [2.0, 3.0, 4.0]
        assert result['pure_results'] == expected, (
            f"Expected {expected}, got {result['pure_results']}. "
            "pure_fn(x, counter) should return (x + counter + 1, counter + 1). "
            "Thread the counter through each call."
        )

    def test_jit_compatible(self):
        """Pure function should be JIT-compatible."""
        result = exercise_pure_functions()

        assert result['jit_result'] == 2.0, (
            f"Expected jit_result = 2.0, got {result['jit_result']}. "
            "JIT compile pure_fn and call with x=1.0, counter=0."
        )

    def test_final_counter(self):
        """Counter should be incremented correctly."""
        result = exercise_pure_functions()

        assert result['final_counter'] == 3, (
            f"Expected final_counter = 3, got {result['final_counter']}. "
            "After 3 calls, counter should go from 0 to 3."
        )


class TestDynamicShapes:
    """Tests for Exercise 10: Dynamic Shapes"""

    def test_masked_mean(self):
        """Masked mean should handle variable-size data correctly."""
        result = exercise_dynamic_shapes()

        assert result['padded_shape'] is not None, (
            "padded_shape should not be None. "
            "Create x_padded with shape (64, 10)."
        )
        assert result['padded_shape'] == (64, 10), (
            f"Expected padded_shape (64, 10), got {result['padded_shape']}."
        )

        assert result['masked_mean_shape'] is not None, (
            "masked_mean_shape should not be None. "
            "Implement masked_mean: masked = x * mask[:, None], "
            "return sum(masked, axis=0) / sum(mask)."
        )
        assert result['masked_mean_shape'] == (10,), (
            f"Expected masked_mean_shape (10,), got {result['masked_mean_shape']}."
        )

    def test_masking_strategy(self):
        """Verify masking correctly computes mean over valid elements."""
        @jax.jit
        def masked_mean(x, mask):
            masked = x * mask[:, None]
            return jnp.sum(masked, axis=0) / jnp.sum(mask)

        x = jnp.ones((10, 5))
        mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = masked_mean(x, mask)
        # Mean of first 3 rows of ones should be ones
        assert jnp.allclose(result, jnp.ones(5)), (
            "Masked mean of ones (with mask selecting first 3 rows) should be ones. "
            "Formula: sum(x * mask[:, None], axis=0) / sum(mask)."
        )


class TestJitEdgeCases:
    """Additional edge case tests for JIT behavior."""

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
        assert jnp.allclose(result, expected), (
            "Nested JIT functions should compose correctly. "
            "outer(x) = inner(x) + 1 = x * 2 + 1."
        )

    def test_jit_with_kwargs(self):
        """JIT should handle keyword arguments."""
        @jax.jit
        def fn(x, scale=2.0):
            return x * scale

        x = jnp.array([1.0, 2.0])
        result = fn(x, scale=3.0)
        expected = jnp.array([3.0, 6.0])
        assert jnp.allclose(result, expected), (
            "JIT functions should handle keyword arguments correctly."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
