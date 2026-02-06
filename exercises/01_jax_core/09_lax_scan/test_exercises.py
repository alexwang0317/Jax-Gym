"""
Tests for JAX lax.scan Exercises
================================
"""

import pytest
import jax
import jax.numpy as jnp
from jax import lax

from exercises import (
    exercise_basic_scan,
    exercise_performance,
    exercise_carry_state,
    exercise_rnn_scan,
    exercise_reverse_scan,
    exercise_variable_length,
    exercise_scan_vmap,
    exercise_unrolling,
    exercise_time_series,
    exercise_checkpointed_scan,
)


class TestBasicScan:
    def test_cumsum(self):
        result = exercise_basic_scan()
        assert result['matches'] == True

    def test_final_carry(self):
        result = exercise_basic_scan()
        assert result['final_carry'] == 15


class TestPerformance:
    def test_all_match(self):
        result = exercise_performance()
        assert result['all_match'] == True


class TestCarryState:
    def test_count(self):
        result = exercise_carry_state()
        assert result['final_count'] == 10

    def test_mean(self):
        result = exercise_carry_state()
        assert jnp.allclose(result['final_mean'], result['expected_mean'])


class TestRnnScan:
    def test_hidden_shape(self):
        result = exercise_rnn_scan()
        assert result['hidden_states_shape'] == (50, 32)

    def test_final_shape(self):
        result = exercise_rnn_scan()
        assert result['final_h_shape'] == (32,)

    def test_jit(self):
        result = exercise_rnn_scan()
        assert result['jit_matches'] == True


class TestReverseScan:
    def test_forward(self):
        result = exercise_reverse_scan()
        expected = jnp.array([1, 3, 6, 10, 15])
        assert jnp.allclose(result['forward'], expected)

    def test_reverse(self):
        result = exercise_reverse_scan()
        # Reverse cumsum: [15, 14, 12, 9, 5]
        expected = jnp.array([15, 14, 12, 9, 5])
        assert jnp.allclose(result['reverse'], expected)

    def test_bidirectional_shape(self):
        result = exercise_reverse_scan()
        assert result['bidirectional_shape'] == (5, 2)


class TestVariableLength:
    def test_masked_sum(self):
        result = exercise_variable_length()
        # Sum of first 3 elements: 1 + 2 + 3 = 6
        assert jnp.allclose(result['expected'], 6.0)
        assert jnp.allclose(result['masked_final'], result['expected'])


class TestScanVmap:
    def test_cumsum_match(self):
        result = exercise_scan_vmap()
        assert result['cumsum_match'] == True

    def test_rnn_shapes(self):
        result = exercise_scan_vmap()
        # (batch, seq_len, hidden_dim)
        assert result['rnn_hidden_shape'] == (3, 10, 8)
        assert result['rnn_final_shape'] == (3, 8)


class TestUnrolling:
    def test_all_match(self):
        result = exercise_unrolling()
        assert result['all_match'] == True


class TestTimeSeries:
    def test_shapes(self):
        result = exercise_time_series()
        assert result['original_shape'] == (100,)
        assert result['smoothed_shape'] == (100,)
        assert result['diffs_shape'] == (99,)

    def test_cumprod(self):
        result = exercise_time_series()
        # 1.01 * 0.99 * 1.02 * 1.01 * 0.98
        expected = 1.01 * 0.99 * 1.02 * 1.01 * 0.98
        assert jnp.allclose(result['final_price'], expected, rtol=1e-5)


class TestCheckpointedScan:
    def test_gradients_match(self):
        result = exercise_checkpointed_scan()
        assert result['gradients_match'] == True

    def test_chunked_matches(self):
        result = exercise_checkpointed_scan()
        assert result['chunked_matches_normal'] == True


class TestScanEdgeCases:
    def test_empty_sequence(self):
        """Scan with empty sequence."""
        def scan_fn(c, x):
            return c + x, c + x

        xs = jnp.array([])
        final, outputs = lax.scan(scan_fn, 0.0, xs)

        assert final == 0.0
        assert outputs.shape == (0,)

    def test_single_element(self):
        """Scan with single element."""
        def scan_fn(c, x):
            return c + x, c + x

        xs = jnp.array([5.0])
        final, outputs = lax.scan(scan_fn, 0.0, xs)

        assert final == 5.0
        assert jnp.allclose(outputs, jnp.array([5.0]))

    def test_pytree_output(self):
        """Scan can output pytrees."""
        def scan_fn(c, x):
            return c + x, {'sum': c + x, 'squared': (c + x) ** 2}

        xs = jnp.array([1.0, 2.0, 3.0])
        final, outputs = lax.scan(scan_fn, 0.0, xs)

        assert outputs['sum'].shape == (3,)
        assert outputs['squared'].shape == (3,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
