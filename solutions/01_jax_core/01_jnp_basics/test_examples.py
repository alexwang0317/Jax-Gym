"""
Tests for JAX NumPy Basics Examples
===================================
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from examples import (
    example_array_creation,
    example_reshaping_slicing,
    example_math_operations,
    example_linear_algebra,
    example_broadcasting,
    example_reductions,
    example_boolean_indexing,
    example_immutability,
    example_einsum,
    example_device_placement,
)


class TestArrayCreation:
    def test_array_shapes(self):
        result = example_array_creation()
        assert result['from_list'].shape == (5,)
        assert result['zeros'].shape == (3, 4)
        assert result['ones'].shape == (2, 3)
        assert result['arange'].shape == (5,)
        assert result['linspace'].shape == (5,)
        assert result['eye'].shape == (3, 3)
        assert result['full'].shape == (2, 2)

    def test_array_values(self):
        result = example_array_creation()
        assert jnp.allclose(result['arange'], jnp.array([0, 2, 4, 6, 8]))
        assert jnp.allclose(result['full'], jnp.full((2, 2), 7.0))
        assert jnp.allclose(result['eye'], jnp.eye(3))

    def test_dtypes(self):
        result = example_array_creation()
        assert result['ones'].dtype == jnp.float32


class TestReshapingSlicing:
    def test_reshape(self):
        result = example_reshaping_slicing()
        assert result['reshaped'].shape == (3, 4)
        assert result['transposed'].shape == (4, 3)

    def test_slicing(self):
        result = example_reshaping_slicing()
        assert result['row_0'].shape == (4,)
        assert result['col_0'].shape == (3,)
        assert result['submatrix'].shape == (2, 2)

    def test_expand_squeeze(self):
        result = example_reshaping_slicing()
        assert result['expanded'].shape == (1, 12)
        assert result['squeezed'].shape == (12,)


class TestMathOperations:
    def test_trig_values(self):
        result = example_math_operations()
        # sin(0) = 0, sin(pi/2) ≈ 1
        assert jnp.abs(result['sin'][0]) < 1e-6
        assert result['sin'].shape == (100,)

    def test_exp_log(self):
        result = example_math_operations()
        # exp(0) = 1
        y = jnp.linspace(0.1, 5, 50)
        exp_y = jnp.exp(y)
        log_y = jnp.log(y)
        # log(exp(x)) = x
        assert jnp.allclose(jnp.log(exp_y), y)

    def test_sqrt(self):
        result = example_math_operations()
        expected = jnp.array([1, 2, 3, 4, 5])
        assert jnp.allclose(result['sqrt'], expected)


class TestLinearAlgebra:
    def test_dot_product(self):
        result = example_linear_algebra()
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert jnp.allclose(result['dot_product'], 32.0)

    def test_matmul(self):
        result = example_linear_algebra()
        expected = jnp.array([[19, 22], [43, 50]], dtype=jnp.float32)
        assert jnp.allclose(result['matmul'], expected)

    def test_inverse(self):
        result = example_linear_algebra()
        A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        A_inv = result['inverse']
        # A @ A_inv should be identity
        identity = A @ A_inv
        assert jnp.allclose(identity, jnp.eye(2), atol=1e-5)

    def test_determinant(self):
        result = example_linear_algebra()
        # det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        assert jnp.allclose(result['determinant'], -2.0)

    def test_trace(self):
        result = example_linear_algebra()
        # trace = 1 + 4 = 5
        assert jnp.allclose(result['trace'], 5.0)


class TestBroadcasting:
    def test_scalar_broadcast(self):
        result = example_broadcasting()
        expected = jnp.array([11, 12, 13])
        assert jnp.allclose(result['scalar_add'], expected)

    def test_row_col_broadcast(self):
        result = example_broadcasting()
        # with_row: 3x4 matrix + [1,2,3,4]
        assert result['with_row_vec'].shape == (3, 4)
        assert result['with_col_vec'].shape == (3, 4)

    def test_outer_via_broadcast(self):
        result = example_broadcasting()
        expected = jnp.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
        assert jnp.allclose(result['outer_via_broadcast'], expected)

    def test_batch_broadcast(self):
        result = example_broadcasting()
        assert result['batch_broadcast_shape'] == (2, 3, 4)


class TestReductions:
    def test_sum(self):
        result = example_reductions()
        assert jnp.allclose(result['sum'], 45.0)

    def test_mean(self):
        result = example_reductions()
        assert jnp.allclose(result['mean'], 5.0)

    def test_max(self):
        result = example_reductions()
        assert jnp.allclose(result['max'], 9.0)

    def test_row_col_sum(self):
        result = example_reductions()
        assert jnp.allclose(result['row_sum'], jnp.array([6, 15, 24]))
        assert jnp.allclose(result['col_sum'], jnp.array([12, 15, 18]))

    def test_argmax(self):
        result = example_reductions()
        assert result['argmax'] == 8  # Index of 9 in flattened array


class TestBooleanIndexing:
    def test_mask(self):
        result = example_boolean_indexing()
        expected = jnp.array([True, False, True, False, True, False])
        assert jnp.allclose(result['positive_mask'], expected)

    def test_num_positive(self):
        result = example_boolean_indexing()
        assert result['num_positive'] == 3

    def test_abs_via_where(self):
        result = example_boolean_indexing()
        expected = jnp.array([1, 2, 3, 4, 5, 6])
        assert jnp.allclose(result['abs_via_where'], expected)

    def test_clip(self):
        result = example_boolean_indexing()
        expected = jnp.array([1, -2, 3, -3, 3, -3])
        assert jnp.allclose(result['clipped'], expected)

    def test_all_any(self):
        result = example_boolean_indexing()
        assert result['all_positive'] == True
        assert result['any_greater_3'] == True


class TestImmutability:
    def test_original_unchanged(self):
        result = example_immutability()
        expected_original = jnp.array([1, 2, 3, 4, 5])
        assert jnp.allclose(result['original'], expected_original)

    def test_modified(self):
        result = example_immutability()
        expected = jnp.array([10, 2, 3, 4, 5])
        assert jnp.allclose(result['modified'], expected)

    def test_multi_update(self):
        result = example_immutability()
        expected = jnp.array([10, 2, 3, 4, 50])
        assert jnp.allclose(result['multi_update'], expected)

    def test_slice_update(self):
        result = example_immutability()
        expected = jnp.array([1, 0, 0, 0, 5])
        assert jnp.allclose(result['slice_update'], expected)

    def test_scatter(self):
        result = example_immutability()
        # indices [0, 2, 2] with values [1, 2, 3]
        # index 0 gets 1, index 2 gets 2+3=5
        expected = jnp.array([1, 0, 5, 0, 0])
        assert jnp.allclose(result['scattered'], expected)


class TestEinsum:
    def test_dot_product(self):
        result = example_einsum()
        # 1*4 + 2*5 + 3*6 = 32
        assert jnp.allclose(result['dot_product'], 32)

    def test_outer_product(self):
        result = example_einsum()
        expected = jnp.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
        assert jnp.allclose(result['outer_product'], expected)

    def test_matmul(self):
        result = example_einsum()
        expected = jnp.array([[19, 22], [43, 50]])
        assert jnp.allclose(result['matmul'], expected)

    def test_trace(self):
        result = example_einsum()
        assert jnp.allclose(result['trace'], 5)

    def test_batch_matmul_shape(self):
        result = example_einsum()
        assert result['batch_matmul_shape'] == (2, 3, 5)

    def test_attention_shape(self):
        result = example_einsum()
        assert result['attention_scores_shape'] == (2, 4, 8, 8)


class TestDevicePlacement:
    def test_devices_available(self):
        result = example_device_placement()
        assert len(result['devices']) > 0

    def test_default_backend(self):
        result = example_device_placement()
        assert result['default_backend'] in ['cpu', 'gpu', 'tpu']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
