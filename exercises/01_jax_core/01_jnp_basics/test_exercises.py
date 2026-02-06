"""
Tests for JAX NumPy Basics Exercises
====================================

Run with: pytest test_exercises.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from exercises import (
    exercise_array_creation,
    exercise_reshaping_slicing,
    exercise_math_operations,
    exercise_linear_algebra,
    exercise_broadcasting,
    exercise_reductions,
    exercise_boolean_indexing,
    exercise_immutability,
    exercise_einsum,
    exercise_device_placement,
)


class TestArrayCreation:
    """Exercise 1: Test array creation functions."""

    def test_array_shapes(self):
        result = exercise_array_creation()
        assert result['from_list'] is not None, "TODO: Create arr1 from list [1,2,3,4,5]"
        assert result['from_list'].shape == (5,), f"Expected shape (5,), got {result['from_list'].shape}"
        assert result['zeros'].shape == (3, 4), f"Expected shape (3,4), got {result['zeros'].shape}"
        assert result['ones'].shape == (2, 3), f"Expected shape (2,3), got {result['ones'].shape}"
        assert result['arange'].shape == (5,), f"Expected shape (5,), got {result['arange'].shape}"
        assert result['linspace'].shape == (5,), f"Expected shape (5,), got {result['linspace'].shape}"
        assert result['eye'].shape == (3, 3), f"Expected shape (3,3), got {result['eye'].shape}"
        assert result['full'].shape == (2, 2), f"Expected shape (2,2), got {result['full'].shape}"

    def test_array_values(self):
        result = exercise_array_creation()
        assert jnp.allclose(result['arange'], jnp.array([0, 2, 4, 6, 8])), \
            "arange should be [0, 2, 4, 6, 8]"
        assert jnp.allclose(result['full'], jnp.full((2, 2), 7.0)), \
            "full should be 2x2 matrix of 7.0"
        assert jnp.allclose(result['eye'], jnp.eye(3)), \
            "eye should be 3x3 identity matrix"

    def test_dtypes(self):
        result = exercise_array_creation()
        assert result['ones'].dtype == jnp.float32, \
            f"ones should have dtype float32, got {result['ones'].dtype}"


class TestReshapingSlicing:
    """Exercise 2: Test reshaping and slicing operations."""

    def test_reshape(self):
        result = exercise_reshaping_slicing()
        assert result['reshaped'] is not None, "TODO: Reshape arr to (3, 4)"
        assert result['reshaped'].shape == (3, 4), f"Expected (3,4), got {result['reshaped'].shape}"
        assert result['transposed'].shape == (4, 3), f"Expected (4,3), got {result['transposed'].shape}"

    def test_slicing(self):
        result = exercise_reshaping_slicing()
        assert result['row_0'].shape == (4,), f"First row should have shape (4,)"
        assert result['col_0'].shape == (3,), f"First column should have shape (3,)"
        assert result['submatrix'].shape == (2, 2), f"Submatrix should have shape (2,2)"

    def test_expand_squeeze(self):
        result = exercise_reshaping_slicing()
        assert result['expanded'].shape == (1, 12), f"Expanded should have shape (1,12)"
        assert result['squeezed'].shape == (12,), f"Squeezed should have shape (12,)"


class TestMathOperations:
    """Exercise 3: Test mathematical operations."""

    def test_trig_values(self):
        result = exercise_math_operations()
        assert result['sin'] is not None, "TODO: Compute sin(x)"
        assert jnp.abs(result['sin'][0]) < 1e-6, "sin(0) should be approximately 0"
        assert result['sin'].shape == (100,), f"sin should have shape (100,)"

    def test_exp_log(self):
        result = exercise_math_operations()
        assert result['exp'] is not None, "TODO: Compute exp(y)"
        y = jnp.linspace(0.1, 5, 50)
        assert jnp.allclose(jnp.log(result['exp']), y), "log(exp(y)) should equal y"

    def test_sqrt(self):
        result = exercise_math_operations()
        expected = jnp.array([1, 2, 3, 4, 5])
        assert jnp.allclose(result['sqrt'], expected), \
            f"sqrt([1,4,9,16,25]) should be [1,2,3,4,5], got {result['sqrt']}"


class TestLinearAlgebra:
    """Exercise 4: Test linear algebra operations."""

    def test_dot_product(self):
        result = exercise_linear_algebra()
        assert result['dot_product'] is not None, "TODO: Compute dot product"
        assert jnp.allclose(result['dot_product'], 32.0), \
            f"v1.v2 = 1*4 + 2*5 + 3*6 = 32, got {result['dot_product']}"

    def test_matmul(self):
        result = exercise_linear_algebra()
        expected = jnp.array([[19, 22], [43, 50]], dtype=jnp.float32)
        assert jnp.allclose(result['matmul'], expected), \
            f"A @ B should be [[19,22],[43,50]], got {result['matmul']}"

    def test_inverse(self):
        result = exercise_linear_algebra()
        assert result['inverse'] is not None, "TODO: Compute matrix inverse"
        A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        identity = A @ result['inverse']
        assert jnp.allclose(identity, jnp.eye(2), atol=1e-5), \
            "A @ A_inv should be identity matrix"

    def test_determinant(self):
        result = exercise_linear_algebra()
        assert jnp.allclose(result['determinant'], -2.0), \
            f"det([[1,2],[3,4]]) = -2, got {result['determinant']}"

    def test_trace(self):
        result = exercise_linear_algebra()
        assert jnp.allclose(result['trace'], 5.0), \
            f"trace([[1,2],[3,4]]) = 5, got {result['trace']}"


class TestBroadcasting:
    """Exercise 5: Test broadcasting operations."""

    def test_scalar_broadcast(self):
        result = exercise_broadcasting()
        expected = jnp.array([11, 12, 13])
        assert result['scalar_add'] is not None, "TODO: Add 10 to [1,2,3]"
        assert jnp.allclose(result['scalar_add'], expected), \
            f"[1,2,3] + 10 should be [11,12,13], got {result['scalar_add']}"

    def test_row_col_broadcast(self):
        result = exercise_broadcasting()
        assert result['with_row_vec'].shape == (3, 4), \
            f"Broadcasting row vec should give shape (3,4)"
        assert result['with_col_vec'].shape == (3, 4), \
            f"Broadcasting col vec should give shape (3,4)"

    def test_outer_via_broadcast(self):
        result = exercise_broadcasting()
        expected = jnp.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
        assert jnp.allclose(result['outer_via_broadcast'], expected), \
            f"Outer product via broadcasting should be [[4,5,6],[8,10,12],[12,15,18]]"

    def test_batch_broadcast(self):
        result = exercise_broadcasting()
        assert result['batch_broadcast_shape'] == (2, 3, 4), \
            f"Batch broadcast shape should be (2,3,4)"


class TestReductions:
    """Exercise 6: Test reduction operations."""

    def test_sum(self):
        result = exercise_reductions()
        assert result['sum'] is not None, "TODO: Compute sum"
        assert jnp.allclose(result['sum'], 45.0), f"Sum should be 45, got {result['sum']}"

    def test_mean(self):
        result = exercise_reductions()
        assert jnp.allclose(result['mean'], 5.0), f"Mean should be 5.0, got {result['mean']}"

    def test_max(self):
        result = exercise_reductions()
        assert jnp.allclose(result['max'], 9.0), f"Max should be 9, got {result['max']}"

    def test_row_col_sum(self):
        result = exercise_reductions()
        assert jnp.allclose(result['row_sum'], jnp.array([6, 15, 24])), \
            f"Row sums should be [6, 15, 24], got {result['row_sum']}"
        assert jnp.allclose(result['col_sum'], jnp.array([12, 15, 18])), \
            f"Column sums should be [12, 15, 18], got {result['col_sum']}"

    def test_argmax(self):
        result = exercise_reductions()
        assert result['argmax'] == 8, f"Argmax should be 8 (index of 9), got {result['argmax']}"


class TestBooleanIndexing:
    """Exercise 7: Test boolean indexing and where."""

    def test_mask(self):
        result = exercise_boolean_indexing()
        expected = jnp.array([True, False, True, False, True, False])
        assert result['positive_mask'] is not None, "TODO: Create positive mask"
        assert jnp.allclose(result['positive_mask'], expected), \
            f"Positive mask should be [T,F,T,F,T,F], got {result['positive_mask']}"

    def test_num_positive(self):
        result = exercise_boolean_indexing()
        assert result['num_positive'] == 3, \
            f"Number of positive values should be 3, got {result['num_positive']}"

    def test_abs_via_where(self):
        result = exercise_boolean_indexing()
        expected = jnp.array([1, 2, 3, 4, 5, 6])
        assert jnp.allclose(result['abs_via_where'], expected), \
            f"Absolute value via where should be [1,2,3,4,5,6], got {result['abs_via_where']}"

    def test_clip(self):
        result = exercise_boolean_indexing()
        expected = jnp.array([1, -2, 3, -3, 3, -3])
        assert jnp.allclose(result['clipped'], expected), \
            f"Clipped to [-3,3] should be [1,-2,3,-3,3,-3], got {result['clipped']}"

    def test_all_any(self):
        result = exercise_boolean_indexing()
        assert result['all_positive'] == True, "All values in [[1,2],[3,4]] are positive"
        assert result['any_greater_3'] == True, "Some values in [[1,2],[3,4]] are > 3"


class TestImmutability:
    """Exercise 8: Test immutable array operations."""

    def test_original_unchanged(self):
        result = exercise_immutability()
        expected_original = jnp.array([1, 2, 3, 4, 5])
        assert jnp.allclose(result['original'], expected_original), \
            "Original array should remain unchanged [1,2,3,4,5]"

    def test_modified(self):
        result = exercise_immutability()
        expected = jnp.array([10, 2, 3, 4, 5])
        assert result['modified'] is not None, "TODO: Use .at[0].set(10)"
        assert jnp.allclose(result['modified'], expected), \
            f"Modified should be [10,2,3,4,5], got {result['modified']}"

    def test_multi_update(self):
        result = exercise_immutability()
        expected = jnp.array([10, 2, 3, 4, 50])
        assert jnp.allclose(result['multi_update'], expected), \
            f"Multi update should be [10,2,3,4,50], got {result['multi_update']}"

    def test_slice_update(self):
        result = exercise_immutability()
        expected = jnp.array([1, 0, 0, 0, 5])
        assert jnp.allclose(result['slice_update'], expected), \
            f"Slice update should be [1,0,0,0,5], got {result['slice_update']}"

    def test_scatter(self):
        result = exercise_immutability()
        expected = jnp.array([1, 0, 5, 0, 0])
        assert result['scattered'] is not None, "TODO: Use .at[indices].add(values)"
        assert jnp.allclose(result['scattered'], expected), \
            f"Scatter should be [1,0,5,0,0] (index 2 gets 2+3), got {result['scattered']}"


class TestEinsum:
    """Exercise 9: Test einsum operations."""

    def test_dot_product(self):
        result = exercise_einsum()
        assert result['dot_product'] is not None, "TODO: Use einsum 'i,i->'"
        assert jnp.allclose(result['dot_product'], 32), \
            f"Dot product should be 32, got {result['dot_product']}"

    def test_outer_product(self):
        result = exercise_einsum()
        expected = jnp.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
        assert jnp.allclose(result['outer_product'], expected), \
            f"Outer product should be [[4,5,6],[8,10,12],[12,15,18]]"

    def test_matmul(self):
        result = exercise_einsum()
        expected = jnp.array([[19, 22], [43, 50]])
        assert jnp.allclose(result['matmul'], expected), \
            f"Matmul via einsum should be [[19,22],[43,50]], got {result['matmul']}"

    def test_trace(self):
        result = exercise_einsum()
        assert jnp.allclose(result['trace'], 5), \
            f"Trace via einsum should be 5, got {result['trace']}"

    def test_batch_matmul_shape(self):
        result = exercise_einsum()
        assert result['batch_matmul_shape'] == (2, 3, 5), \
            f"Batch matmul shape should be (2,3,5), got {result['batch_matmul_shape']}"

    def test_attention_shape(self):
        result = exercise_einsum()
        assert result['attention_scores_shape'] == (2, 4, 8, 8), \
            f"Attention scores shape should be (2,4,8,8), got {result['attention_scores_shape']}"


class TestDevicePlacement:
    """Exercise 10: Test device placement."""

    def test_devices_available(self):
        result = exercise_device_placement()
        assert result['devices'] is not None, "TODO: Get list of devices"
        assert len(result['devices']) > 0, "Should have at least one device"

    def test_default_backend(self):
        result = exercise_device_placement()
        assert result['default_backend'] in ['cpu', 'gpu', 'tpu'], \
            f"Backend should be cpu/gpu/tpu, got {result['default_backend']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
