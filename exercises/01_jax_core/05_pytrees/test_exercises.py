"""
Tests for JAX Pytrees Exercises
===============================

Run with: pytest test_exercises.py -v
"""

import pytest
import jax
import jax.numpy as jnp
from typing import NamedTuple

from exercises import (
    exercise_basic_pytree,
    exercise_tree_map,
    exercise_leaves_structure,
    exercise_tree_reduce,
    exercise_custom_pytree,
    exercise_nn_params,
    exercise_flatten_unflatten,
    exercise_pytree_transforms,
    exercise_combine_pytrees,
    exercise_pytree_utilities,
)


class TestBasicPytree:
    """Exercise 1: Test basic pytree structure."""

    def test_simple_leaves(self):
        result = exercise_basic_pytree()
        assert result['simple_leaves'] is not None, "TODO: Create simple_params dict"
        # simple_params has 2 arrays (weights and bias)
        assert len(result['simple_leaves']) == 2, \
            f"simple_params should have 2 leaves, got {len(result['simple_leaves'])}"

    def test_mlp_params(self):
        result = exercise_basic_pytree()
        assert result['mlp_num_params'] is not None, "TODO: Create mlp_params dict"
        # 3 layers * 2 arrays (w, b) = 6
        assert result['mlp_num_params'] == 6, \
            f"mlp_params should have 6 leaves, got {result['mlp_num_params']}"

    def test_list_leaves(self):
        result = exercise_basic_pytree()
        assert result['list_leaves'] is not None, "TODO: Create list_tree"
        assert len(result['list_leaves']) == 2, \
            f"list_tree should have 2 leaves, got {len(result['list_leaves'])}"


class TestTreeMap:
    """Exercise 2: Test jax.tree.map operations."""

    def test_doubled(self):
        result = exercise_tree_map()
        assert result['doubled_layer1_w_sum'] is not None, "TODO: Double params with tree.map"
        # Original was ones (3x4), doubled should have sum = 2 * 3 * 4 = 24
        assert jnp.allclose(result['doubled_layer1_w_sum'], 24.0), \
            f"doubled layer1 w sum should be 24.0, got {result['doubled_layer1_w_sum']}"

    def test_updated(self):
        result = exercise_tree_map()
        assert result['updated_layer1_w_sum'] is not None, "TODO: Apply gradient update"
        # Updated = 1 - 0.1 * 1 = 0.9, sum = 0.9 * 12 = 10.8
        assert jnp.allclose(result['updated_layer1_w_sum'], 10.8), \
            f"updated layer1 w sum should be 10.8, got {result['updated_layer1_w_sum']}"

    def test_structure_preserved(self):
        result = exercise_tree_map()
        assert result['structure_preserved'] is not None, "TODO: Use tree.map"
        assert result['structure_preserved'] == True, \
            "tree.map should preserve structure"


class TestLeavesStructure:
    """Exercise 3: Test jax.tree.leaves and jax.tree.structure."""

    def test_num_arrays(self):
        result = exercise_leaves_structure()
        assert result['num_arrays'] is not None, "TODO: Get leaves with tree.leaves"
        # conv1: w, b; conv2: w, b; dense: w, b = 6 arrays
        assert result['num_arrays'] == 6, \
            f"Should have 6 leaf arrays, got {result['num_arrays']}"

    def test_total_params(self):
        result = exercise_leaves_structure()
        assert result['total_params'] is not None, "TODO: Sum x.size for all leaves"
        # conv1: 3*3*3*64 + 64 = 1728 + 64 = 1792
        # conv2: 3*3*64*128 + 128 = 73728 + 128 = 73856
        # dense: 128*10 + 10 = 1280 + 10 = 1290
        # Total = 76938
        expected = 3*3*3*64 + 64 + 3*3*64*128 + 128 + 128*10 + 10
        assert result['total_params'] == expected, \
            f"Total params should be {expected}, got {result['total_params']}"

    def test_reconstructed(self):
        result = exercise_leaves_structure()
        assert result['reconstructed_matches'] is not None, "TODO: Use tree.unflatten"
        assert result['reconstructed_matches'] == True, \
            "Reconstructed pytree should have same keys"


class TestTreeReduce:
    """Exercise 4: Test jax.tree.reduce operations."""

    def test_total_params(self):
        result = exercise_tree_reduce()
        assert result['total_params'] is not None, "TODO: Use tree.reduce to count params"
        # layer1: 10*20 + 20 = 220
        # layer2: 20*5 + 5 = 105
        # Total = 325
        assert result['total_params'] == 325, \
            f"Total params should be 325, got {result['total_params']}"

    def test_via_leaves(self):
        result = exercise_tree_reduce()
        assert result['total_via_leaves'] is not None, "TODO: Count via tree.leaves"
        assert result['total_via_leaves'] == result['total_params'], \
            "Both methods should give same count"


class TestCustomPytree:
    """Exercise 5: Test custom pytree nodes."""

    def test_namedtuple(self):
        result = exercise_custom_pytree()
        assert result['namedtuple_works'] is not None, "TODO: Create LayerParams and double it"
        assert result['namedtuple_works'] == True, \
            "NamedTuple should work as pytree"
        # w was (3,4) ones, doubled = 24
        assert jnp.allclose(result['namedtuple_w_sum'], 24.0), \
            f"namedtuple w sum should be 24.0, got {result['namedtuple_w_sum']}"

    def test_custom_class(self):
        result = exercise_custom_pytree()
        assert result['custom_name_preserved'] is not None, "TODO: Register CustomLayer"
        assert result['custom_name_preserved'] == True, \
            "Custom class metadata (name) should be preserved"
        # w was (3,4) ones, tripled = 36
        assert jnp.allclose(result['custom_w_sum'], 36.0), \
            f"custom w sum should be 36.0, got {result['custom_w_sum']}"


class TestNNParams:
    """Exercise 6: Test neural network parameter management."""

    def test_output_shape(self):
        result = exercise_nn_params()
        assert result['output_shape'] is not None, "TODO: Initialize and run forward pass"
        # Batch of 32, output size 10
        assert result['output_shape'] == (32, 10), \
            f"Output shape should be (32, 10), got {result['output_shape']}"

    def test_grad_shape(self):
        result = exercise_nn_params()
        assert result['grad_layer0_w_shape'] is not None, "TODO: Compute gradients"
        # layer0: (784, 256)
        assert result['grad_layer0_w_shape'] == (784, 256), \
            f"Gradient shape should be (784, 256), got {result['grad_layer0_w_shape']}"

    def test_params_updated(self):
        result = exercise_nn_params()
        assert result['params_updated'] is not None, "TODO: Update params with tree.map"
        assert result['params_updated'] == True, \
            "Parameters should change after gradient update"


class TestFlattenUnflatten:
    """Exercise 7: Test flatten and unflatten operations."""

    def test_num_leaves(self):
        result = exercise_flatten_unflatten()
        assert result['num_leaves'] is not None, "TODO: Use tree_util.tree_flatten"
        # encoder: w, b; decoder: w, b = 4
        assert result['num_leaves'] == 4, \
            f"Should have 4 leaves, got {result['num_leaves']}"

    def test_roundtrip(self):
        result = exercise_flatten_unflatten()
        assert result['roundtrip_works'] is not None, "TODO: Verify flatten/unflatten roundtrip"
        assert result['roundtrip_works'] == True, \
            "Flatten/unflatten roundtrip should preserve values"

    def test_modified(self):
        result = exercise_flatten_unflatten()
        assert result['modified_sum'] is not None, "TODO: Modify leaves and unflatten"
        # encoder w was (10, 20) ones, +1 = 2 each, sum = 400
        assert jnp.allclose(result['modified_sum'], 400.0), \
            f"Modified sum should be 400.0, got {result['modified_sum']}"


class TestPytreeTransforms:
    """Exercise 8: Test pytrees with JAX transformations."""

    def test_output_shape(self):
        result = exercise_pytree_transforms()
        assert result['output_shape'] is not None, "TODO: Implement model and jit"
        assert result['output_shape'] == (16, 2), \
            f"Output shape should be (16, 2), got {result['output_shape']}"

    def test_grad_structure(self):
        result = exercise_pytree_transforms()
        assert result['grad_keys_match'] is not None, "TODO: Compute gradients"
        assert result['grad_keys_match'] == True, \
            "Gradient keys should match param keys"

    def test_grad_shape(self):
        result = exercise_pytree_transforms()
        assert result['grad_w1_shape'] is not None, "TODO: Compute gradients"
        assert result['grad_w1_shape'] == (4, 8), \
            f"Gradient w1 shape should be (4, 8), got {result['grad_w1_shape']}"


class TestCombinePytrees:
    """Exercise 9: Test combining multiple pytrees."""

    def test_momentum_update(self):
        result = exercise_combine_pytrees()
        assert result['momentum_w_sum'] is not None, "TODO: Compute new momentum"
        # momentum = 0.9 * 0 + 0.1 = 0.1 per element
        # w is (3, 4), so sum = 12 * 0.1 = 1.2
        assert jnp.allclose(result['momentum_w_sum'], 1.2), \
            f"Momentum w sum should be 1.2, got {result['momentum_w_sum']}"

    def test_params_update(self):
        result = exercise_combine_pytrees()
        assert result['params_updated_w_sum'] is not None, "TODO: Update params"
        # params = 1 - 0.01 * 0.1 = 0.999 per element
        # sum = 12 * 0.999 = 11.988
        assert jnp.allclose(result['params_updated_w_sum'], 11.988), \
            f"Updated params w sum should be 11.988, got {result['params_updated_w_sum']}"


class TestPytreeUtilities:
    """Exercise 10: Test pytree utility functions."""

    def test_all_finite(self):
        result = exercise_pytree_utilities()
        assert result['all_finite'] is not None, "TODO: Check all leaves are finite"
        assert result['all_finite'] == True, \
            "All params should be finite"

    def test_same_structure(self):
        result = exercise_pytree_utilities()
        assert result['same_structure'] is not None, "TODO: Compare structures"
        assert result['same_structure'] == True, \
            "Scaled params should have same structure"

    def test_stacked_shape(self):
        result = exercise_pytree_utilities()
        assert result['stacked_w_shape'] is not None, "TODO: Stack batched params"
        # 3 params stacked, each w is (3, 4)
        assert result['stacked_w_shape'] == (3, 3, 4), \
            f"Stacked w shape should be (3, 3, 4), got {result['stacked_w_shape']}"


class TestPytreeEdgeCases:
    """Additional edge case tests for pytrees."""

    def test_empty_pytree(self):
        """Empty containers are valid pytrees."""
        empty_dict = {}
        leaves = jax.tree.leaves(empty_dict)
        assert len(leaves) == 0

    def test_none_in_pytree(self):
        """None is treated as a leaf, not a container."""
        with_none = {'a': jnp.array([1, 2]), 'b': None}
        leaves = jax.tree.leaves(with_none)
        assert len(leaves) == 2  # Both a and None

    def test_nested_consistency(self):
        """Nested operations preserve structure."""
        params = {'layer': {'w': jnp.ones((2, 3))}}
        doubled = jax.tree.map(lambda x: x * 2, params)
        halved = jax.tree.map(lambda x: x / 2, doubled)
        assert jnp.allclose(halved['layer']['w'], params['layer']['w'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
