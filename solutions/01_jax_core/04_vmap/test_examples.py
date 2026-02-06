"""
Tests for JAX vmap Examples
===========================
"""

import pytest
import jax
import jax.numpy as jnp

from examples import (
    example_basic_vmap,
    example_in_out_axes,
    example_multiple_args,
    example_nested_vmap,
    example_vmap_jit,
    example_vmap_grad,
    example_matrix_vector,
    example_pairwise_distances,
    example_none_axes,
    example_attention_scores,
)


class TestBasicVmap:
    def test_results_match(self):
        result = example_basic_vmap()
        assert result['results_match'] == True

    def test_norms_correct(self):
        result = example_basic_vmap()
        expected = jnp.array([5.0, 1.0, 2.0])
        assert jnp.allclose(result['batch_norms'], expected)


class TestInOutAxes:
    def test_broadcast_shape(self):
        result = example_in_out_axes()
        assert result['broadcast_result'].shape == (3, 2)

    def test_broadcast_values(self):
        result = example_in_out_axes()
        expected = jnp.array([[11, 22], [13, 24], [15, 26]])
        assert jnp.allclose(result['broadcast_result'], expected)

    def test_outer_shapes(self):
        result = example_in_out_axes()
        assert result['outer_shape'] == (3, 4, 5)
        assert result['outer_last_shape'] == (4, 5, 3)


class TestMultipleArgs:
    def test_all_batched(self):
        result = example_multiple_args()
        # w*x + (1-w)*y for each element
        # [0.5*1 + 0.5*10, 0.3*2 + 0.7*20, 0.7*3 + 0.3*30]
        expected = jnp.array([5.5, 14.6, 11.1])
        assert jnp.allclose(result['all_batched'], expected)

    def test_fixed_weight(self):
        result = example_multiple_args()
        # 0.5*x + 0.5*y
        expected = jnp.array([5.5, 11.0, 16.5])
        assert jnp.allclose(result['fixed_weight'], expected)


class TestNestedVmap:
    def test_single_dot(self):
        result = example_nested_vmap()
        assert jnp.allclose(result['single'], 32.0)

    def test_batched_shape(self):
        result = example_nested_vmap()
        assert result['double_batched_shape'] == (2, 4)

    def test_pairwise_shape(self):
        result = example_nested_vmap()
        assert result['pairwise_shape'] == (3, 2)

    def test_pairwise_values(self):
        result = example_nested_vmap()
        # a_set[0] = [1,0], b_set[0] = [1,0] -> dot = 1
        # a_set[0] = [1,0], b_set[1] = [0,1] -> dot = 0
        expected = jnp.array([[1, 0], [0, 1], [1, 1]], dtype=jnp.float32)
        assert jnp.allclose(result['pairwise'], expected)


class TestVmapJit:
    def test_results_match(self):
        result = example_vmap_jit()
        assert result['results_match'] == True

    def test_output_shape(self):
        result = example_vmap_jit()
        assert result['output_shape'] == (1000,)


class TestVmapGrad:
    def test_gradient_shapes(self):
        result = example_vmap_grad()
        assert result['single_grad_w_shape'] == (2, 2)
        assert result['batch_grad_w_shape'] == (3, 2, 2)
        assert result['batch_grad_b_shape'] == (3, 2)

    def test_mean_gradient_shape(self):
        result = example_vmap_grad()
        assert result['mean_grad_w'].shape == (2, 2)
        assert result['mean_grad_b'].shape == (2,)


class TestMatrixVector:
    def test_single(self):
        result = example_matrix_vector()
        expected = jnp.array([3.0, 7.0])
        assert jnp.allclose(result['single'], expected)

    def test_batch_shapes(self):
        result = example_matrix_vector()
        assert result['batch_x'].shape == (3, 2)
        assert result['batch_A'].shape == (3, 2)
        assert result['batch_both'].shape == (3, 2)


class TestPairwiseDistances:
    def test_methods_match(self):
        result = example_pairwise_distances()
        assert result['methods_match'] == True

    def test_self_diagonal(self):
        result = example_pairwise_distances()
        assert result['self_diagonal_zero'] == True

    def test_distance_values(self):
        result = example_pairwise_distances()
        # Distance from (0,0) to (1,1) is sqrt(2)
        assert jnp.allclose(result['pairwise_distances'][0, 0], jnp.sqrt(2))


class TestNoneAxes:
    def test_transformed_shape(self):
        result = example_none_axes()
        assert result['transformed'].shape == (3, 2)

    def test_transformed_values(self):
        result = example_none_axes()
        # First row: 2*[1,2] + [10,20] = [12, 24]
        assert jnp.allclose(result['transformed'][0], result['expected_first_row'])


class TestAttentionScores:
    def test_single_head_shape(self):
        result = example_attention_scores()
        assert result['single_head_shape'] == (4, 8)

    def test_multi_head_shape(self):
        result = example_attention_scores()
        assert result['multi_head_shape'] == (4, 4, 8)

    def test_batch_multi_head_shape(self):
        result = example_attention_scores()
        assert result['batch_multi_head_shape'] == result['expected_batch_shape']


class TestVmapEdgeCases:
    def test_empty_batch(self):
        """vmap handles empty batches."""
        def f(x):
            return x * 2

        empty = jnp.zeros((0, 3))
        result = jax.vmap(f)(empty)
        assert result.shape == (0, 3)

    def test_scalar_output(self):
        """vmap with scalar output function."""
        def f(x):
            return jnp.sum(x)

        batch = jnp.ones((5, 3))
        result = jax.vmap(f)(batch)
        assert result.shape == (5,)
        assert jnp.allclose(result, jnp.full(5, 3.0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
