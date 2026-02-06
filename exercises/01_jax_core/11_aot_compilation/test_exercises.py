"""
Tests for JAX AOT Compilation Exercises
=======================================

Run these tests to verify your exercise implementations:
    pytest test_exercises.py -v

Each test class corresponds to one exercise. Tests include helpful
assertion messages to guide you toward the correct implementation.
"""

import pytest
import jax
import jax.numpy as jnp

from exercises import (
    exercise_basic_aot,
    exercise_inspect_hlo,
    exercise_aot_shapes,
    exercise_timing_aot,
    exercise_aot_static_args,
    exercise_compilation_cost,
    exercise_reuse_compiled,
    exercise_aot_model_components,
    exercise_compilation_stats,
    exercise_production_pattern,
)


class TestBasicAOT:
    """Tests for Exercise 1: Basic AOT Compilation Pipeline"""

    def test_results_match(self):
        """AOT compiled function should produce correct results."""
        result = exercise_basic_aot()
        assert result['results_match'] == True, (
            "AOT compiled result should match expected (x + y) ** 2. "
            "Make sure you call: staged.lower(x, y).compile()"
        )

    def test_result_not_none(self):
        """Compiled function should produce a result."""
        result = exercise_basic_aot()
        assert result['result'] is not None, (
            "result should not be None. "
            "Call the compiled function with: compiled(x, y)"
        )

    def test_compiled_type(self):
        """Verify we got a compiled function object."""
        result = exercise_basic_aot()
        assert result['compiled_type'] is not None, (
            "Should return the type of the compiled object."
        )

    def test_aot_pipeline_works(self):
        """Verify the basic AOT pipeline works correctly."""
        def fn(x):
            return x * 2

        x = jnp.array([1.0, 2.0, 3.0])
        compiled = jax.jit(fn).lower(x).compile()
        result = compiled(x)
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result, expected), (
            "Basic AOT pipeline should work: jit -> lower -> compile -> call"
        )


class TestInspectHLO:
    """Tests for Exercise 2: Inspecting Lowered HLO"""

    def test_hlo_generated(self):
        """HLO text should be generated."""
        result = exercise_inspect_hlo()
        assert result['hlo_preview'] is not None, (
            "hlo_preview should not be None. "
            "Use lowered.as_text() to get the HLO representation."
        )
        assert result['hlo_length'] > 0, (
            "HLO text should have content."
        )

    def test_hlo_contains_operations(self):
        """HLO should contain expected operations."""
        result = exercise_inspect_hlo()
        # At least one of the expected operations should be present
        has_ops = result['has_matmul'] or result['has_add'] or result['has_maximum']
        assert has_ops, (
            "HLO should contain operations like dot/dot_general, add, or maximum. "
            "The function relu(x @ W + b) should have matmul, add, and max ops."
        )

    def test_result_shape_correct(self):
        """The compiled function should produce correct output shape."""
        result = exercise_inspect_hlo()
        assert result['result_shape'] == (4, 16), (
            f"Expected result shape (4, 16), got {result['result_shape']}. "
            "Input (4, 8) @ W (8, 16) + b (16,) should give (4, 16)."
        )


class TestAOTShapes:
    """Tests for Exercise 3: AOT with Different Shapes"""

    def test_correct_shape_works(self):
        """Compiled function should work with matching shapes."""
        result = exercise_aot_shapes()
        assert result['result_4_shape'] == (4,), (
            f"Expected (4,), got {result['result_4_shape']}. "
            "compiled_4 should work with shape (4,) input."
        )
        assert result['result_8_shape'] == (8,), (
            f"Expected (8,), got {result['result_8_shape']}. "
            "compiled_8 should work with shape (8,) input."
        )

    def test_wrong_shape_fails(self):
        """Compiled function should fail with wrong shapes."""
        result = exercise_aot_shapes()
        assert result['shape_specific'] == True, (
            "AOT compiled functions should reject wrong input shapes. "
            "Try calling compiled_4 with an array of shape (8,)."
        )
        assert result['wrong_shape_error'] is not None, (
            "Should catch an error when using wrong shape. "
            "Use try/except to catch the shape mismatch error."
        )

    def test_shape_specificity(self):
        """Verify AOT is truly shape-specific."""
        def fn(x):
            return x + 1

        x4 = jnp.ones((4,))
        x8 = jnp.ones((8,))

        compiled_4 = jax.jit(fn).lower(x4).compile()

        # Should work with correct shape
        result = compiled_4(x4)
        assert result.shape == (4,)

        # Should fail with wrong shape
        with pytest.raises(Exception):
            compiled_4(x8)


class TestTimingAOT:
    """Tests for Exercise 4: Timing Compile vs Execute"""

    def test_times_positive(self):
        """All timing values should be positive."""
        result = exercise_timing_aot()
        assert result['lower_time_ms'] > 0, "lower_time should be positive"
        assert result['compile_time_ms'] > 0, "compile_time should be positive"
        assert result['exec_time_1_ms'] > 0, "exec_time should be positive"

    def test_execution_times_exist(self):
        """Execution timing should be measured."""
        result = exercise_timing_aot()
        assert result['exec_time_1_ms'] is not None, (
            "exec_time_1 should be measured. "
            "Time the compiled function call with .block_until_ready()."
        )
        assert result['exec_time_2_ms'] is not None, (
            "exec_time_2 should be measured for second call."
        )

    def test_compile_time_measured(self):
        """Compilation time should be separated from execution."""
        result = exercise_timing_aot()
        assert result['total_compile_ms'] is not None, (
            "total_compile_ms should be calculated. "
            "Sum lower_time and compile_time."
        )


class TestAOTStaticArgs:
    """Tests for Exercise 5: AOT with Static Arguments"""

    def test_axis0_result(self):
        """Compiled with axis=0 should reduce correctly."""
        result = exercise_aot_static_args()
        assert result['result_axis0_shape'] == (4, 5), (
            f"Expected (4, 5), got {result['result_axis0_shape']}. "
            "Summing (3, 4, 5) along axis 0 should give (4, 5)."
        )
        assert result['axis0_correct'] == True

    def test_axis1_result(self):
        """Compiled with axis=1 should reduce correctly."""
        result = exercise_aot_static_args()
        assert result['result_axis1_shape'] == (3, 5), (
            f"Expected (3, 5), got {result['result_axis1_shape']}. "
            "Summing (3, 4, 5) along axis 1 should give (3, 5)."
        )
        assert result['axis1_correct'] == True

    def test_static_args_baked_in(self):
        """Static args should be baked into compiled function."""
        def fn(x, axis):
            return jnp.sum(x, axis=axis)

        x = jnp.ones((3, 4))

        # Compile with axis=0 as static
        compiled = jax.jit(fn, static_argnums=(1,)).lower(x, 0).compile()

        # Call without axis argument (it's baked in)
        result = compiled(x)
        assert result.shape == (4,), (
            "With axis=0 baked in, result should have shape (4,)"
        )


class TestCompilationCost:
    """Tests for Exercise 6: Compilation Cost Analysis"""

    def test_times_measured(self):
        """All compilation times should be measured."""
        result = exercise_compilation_cost()
        assert result['simple_compile_ms'] > 0
        assert result['medium_compile_ms'] > 0
        assert result['complex_compile_ms'] > 0

    def test_complexity_affects_time(self):
        """More complex functions should generally take longer to compile."""
        result = exercise_compilation_cost()
        # Complex should take longer than simple (usually)
        assert result['complexity_increases_time'] == True or \
               result['complex_compile_ms'] >= result['simple_compile_ms'] * 0.5, (
            "Complex functions typically take longer to compile. "
            "This may vary by platform but should generally hold."
        )


class TestReuseCompiled:
    """Tests for Exercise 7: Reusing Compiled Functions"""

    def test_multiple_calls(self):
        """Should be able to call compiled function many times."""
        result = exercise_reuse_compiled()
        assert result['num_calls'] >= 10, (
            "Should call the compiled function at least 10 times."
        )

    def test_consistent_timing(self):
        """Execution times should be relatively consistent."""
        result = exercise_reuse_compiled()
        assert result['consistent_timing'] == True or \
               result['time_variance_ms'] < 10, (
            f"Execution times should be consistent. "
            f"Variance: {result['time_variance_ms']:.2f}ms, "
            f"Avg: {result['avg_time_ms']:.2f}ms. "
            "All calls use the same compiled code."
        )

    def test_times_measured(self):
        """Timing statistics should be calculated."""
        result = exercise_reuse_compiled()
        assert result['avg_time_ms'] > 0
        assert result['min_time_ms'] > 0
        assert result['max_time_ms'] > 0


class TestAOTModelComponents:
    """Tests for Exercise 8: AOT Model Components"""

    def test_pipeline_shapes(self):
        """Encoder-decoder pipeline should preserve shapes."""
        result = exercise_aot_model_components()
        assert result['shapes_match'] == True, (
            f"Input shape {result['input_shape']} should match "
            f"output shape {result['output_shape']}. "
            "Autoencoder should reconstruct to original dimension."
        )

    def test_latent_shape(self):
        """Encoder should produce correct latent shape."""
        result = exercise_aot_model_components()
        assert result['latent_shape'] == (8, 16), (
            f"Expected latent shape (8, 16), got {result['latent_shape']}. "
            "Encoder projects from 64 to 16 dimensions."
        )

    def test_pipeline_works(self):
        """Full pipeline should execute successfully."""
        result = exercise_aot_model_components()
        assert result['pipeline_works'] == True


class TestCompilationStats:
    """Tests for Exercise 9: Compilation Statistics"""

    def test_hlo_sizes(self):
        """HLO sizes should be measured."""
        result = exercise_compilation_stats()
        assert result['small_hlo_chars'] > 0, (
            "small_hlo_chars should be positive."
        )
        assert result['large_hlo_chars'] > 0, (
            "large_hlo_chars should be positive."
        )

    def test_complexity_difference(self):
        """Larger function should have more complex HLO."""
        result = exercise_compilation_stats()
        assert result['large_more_complex'] == True, (
            f"Large function HLO ({result['large_hlo_chars']} chars) should be "
            f"larger than small function HLO ({result['small_hlo_chars']} chars)."
        )


class TestProductionPattern:
    """Tests for Exercise 10: Production-Ready Pattern"""

    def test_correct_output_shape(self):
        """Server should produce correct output shape."""
        result = exercise_production_pattern()
        assert result['correct_output_shape'] == True, (
            f"Expected output shape (16, 32), got {result['output_shape']}. "
            "Server with input_dim=64, output_dim=32, batch_size=16."
        )

    def test_validation_works(self):
        """Input validation should catch wrong shapes."""
        result = exercise_production_pattern()
        assert result['validation_works'] == True, (
            "Server should validate input shapes and reject mismatches. "
            "Implement validate_input() to check shapes."
        )
        assert result['validation_error'] is not None, (
            "Should return the validation error message."
        )

    def test_inference_time(self):
        """Inference should be fast and measured."""
        result = exercise_production_pattern()
        assert result['avg_inference_ms'] > 0, (
            "avg_inference_ms should be measured."
        )
        # Inference should typically be under 100ms for this small model
        assert result['avg_inference_ms'] < 100, (
            f"Inference seems slow ({result['avg_inference_ms']:.2f}ms). "
            "Pre-compiled functions should be fast."
        )

    def test_production_pattern(self):
        """Verify the production pattern works end-to-end."""
        result = exercise_production_pattern()
        assert result['production_ready'] == True


class TestAOTEdgeCases:
    """Additional edge case tests for AOT behavior."""

    def test_aot_with_multiple_outputs(self):
        """AOT should handle functions with multiple outputs."""
        def fn(x):
            return x * 2, x + 1

        x = jnp.array([1.0, 2.0, 3.0])
        compiled = jax.jit(fn).lower(x).compile()
        a, b = compiled(x)

        assert jnp.allclose(a, jnp.array([2.0, 4.0, 6.0]))
        assert jnp.allclose(b, jnp.array([2.0, 3.0, 4.0]))

    def test_aot_preserves_precision(self):
        """AOT compilation should preserve numerical precision."""
        def fn(x):
            return jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        compiled = jax.jit(fn).lower(x).compile()

        result_aot = compiled(x)
        result_direct = fn(x)

        assert jnp.allclose(result_aot, result_direct), (
            "AOT compiled result should match direct computation."
        )

    def test_aot_with_nested_structures(self):
        """AOT should handle pytree inputs."""
        def fn(params, x):
            return x @ params['w'] + params['b']

        params = {
            'w': jnp.ones((3, 2)),
            'b': jnp.zeros((2,))
        }
        x = jnp.ones((4, 3))

        compiled = jax.jit(fn).lower(params, x).compile()
        result = compiled(params, x)

        assert result.shape == (4, 2), (
            "AOT should handle dictionary (pytree) inputs correctly."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
