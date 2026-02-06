"""
Tests for JAX Pytrees Examples
==============================
"""

import pytest
import jax
import jax.numpy as jnp
from typing import NamedTuple

from examples import (
    example_basic_pytree,
    example_tree_map,
    example_leaves_structure,
    example_tree_reduce,
    example_custom_pytree,
    example_nn_params,
    example_flatten_unflatten,
    example_pytree_transforms,
    example_combine_pytrees,
    example_pytree_utilities,
)


class TestBasicPytree:
    def test_simple_leaves(self):
        result = example_basic_pytree()
        # simple_params has 2 arrays
        assert len(result['simple_leaves']) == 2

    def test_mlp_params(self):
        result = example_basic_pytree()
        # 3 layers * 2 arrays (w, b) = 6
        assert result['mlp_num_params'] == 6

    def test_list_leaves(self):
        result = example_basic_pytree()
        assert len(result['list_leaves']) == 2


class TestTreeMap:
    def test_doubled(self):
        result = example_tree_map()
        # Original was ones, doubled should have sum = 2 * 3 * 4 = 24
        assert jnp.allclose(result['doubled_layer1_w_sum'], 24.0)

    def test_updated(self):
        result = example_tree_map()
        # Updated = 1 - 0.1 * 1 = 0.9, sum = 0.9 * 12 = 10.8
        assert jnp.allclose(result['updated_layer1_w_sum'], 10.8)

    def test_structure_preserved(self):
        result = example_tree_map()
        assert result['structure_preserved'] == True


class TestLeavesStructure:
    def test_num_arrays(self):
        result = example_leaves_structure()
        # conv1: w, b; conv2: w, b; dense: w, b = 6 arrays
        assert result['num_arrays'] == 6

    def test_total_params(self):
        result = example_leaves_structure()
        # conv1: 3*3*3*64 + 64 = 1728 + 64 = 1792
        # conv2: 3*3*64*128 + 128 = 73728 + 128 = 73856
        # dense: 128*10 + 10 = 1280 + 10 = 1290
        # Total = 1792 + 73856 + 1290 = 76938
        expected = 3*3*3*64 + 64 + 3*3*64*128 + 128 + 128*10 + 10
        assert result['total_params'] == expected

    def test_reconstructed(self):
        result = example_leaves_structure()
        assert result['reconstructed_matches'] == True


class TestTreeReduce:
    def test_total_params(self):
        result = example_tree_reduce()
        # layer1: 10*20 + 20 = 220
        # layer2: 20*5 + 5 = 105
        # Total = 325
        assert result['total_params'] == 325

    def test_via_leaves(self):
        result = example_tree_reduce()
        assert result['total_via_leaves'] == result['total_params']


class TestCustomPytree:
    def test_namedtuple(self):
        result = example_custom_pytree()
        assert result['namedtuple_works'] == True
        # w was (3,4) ones, doubled = 24
        assert jnp.allclose(result['namedtuple_w_sum'], 24.0)

    def test_custom_class(self):
        result = example_custom_pytree()
        assert result['custom_name_preserved'] == True
        # w was (3,4) ones, tripled = 36
        assert jnp.allclose(result['custom_w_sum'], 36.0)


class TestNNParams:
    def test_output_shape(self):
        result = example_nn_params()
        # Batch of 32, output size 10
        assert result['output_shape'] == (32, 10)

    def test_grad_shape(self):
        result = example_nn_params()
        # layer0: (784, 256)
        assert result['grad_layer0_w_shape'] == (784, 256)

    def test_params_updated(self):
        result = example_nn_params()
        assert result['params_updated'] == True


class TestFlattenUnflatten:
    def test_num_leaves(self):
        result = example_flatten_unflatten()
        # encoder: w, b; decoder: w, b = 4
        assert result['num_leaves'] == 4

    def test_roundtrip(self):
        result = example_flatten_unflatten()
        assert result['roundtrip_works'] == True

    def test_modified(self):
        result = example_flatten_unflatten()
        # encoder w was (10, 20) ones, +1 = 2, sum = 400
        assert jnp.allclose(result['modified_sum'], 400.0)


class TestPytreeTransforms:
    def test_output_shape(self):
        result = example_pytree_transforms()
        assert result['output_shape'] == (16, 2)

    def test_grad_structure(self):
        result = example_pytree_transforms()
        assert result['grad_keys_match'] == True

    def test_grad_shape(self):
        result = example_pytree_transforms()
        assert result['grad_w1_shape'] == (4, 8)


class TestCombinePytrees:
    def test_momentum_update(self):
        result = example_combine_pytrees()
        # momentum = 0.9 * 0 + 0.1 = 0.1 per element
        # w is (3, 4), so sum = 12 * 0.1 = 1.2
        assert jnp.allclose(result['momentum_w_sum'], 1.2)

    def test_params_update(self):
        result = example_combine_pytrees()
        # params = 1 - 0.01 * 0.1 = 0.999 per element
        # sum = 12 * 0.999 = 11.988
        assert jnp.allclose(result['params_updated_w_sum'], 11.988)


class TestPytreeUtilities:
    def test_all_finite(self):
        result = example_pytree_utilities()
        assert result['all_finite'] == True

    def test_same_structure(self):
        result = example_pytree_utilities()
        assert result['same_structure'] == True

    def test_stacked_shape(self):
        result = example_pytree_utilities()
        # 3 params stacked, each w is (3, 4)
        assert result['stacked_w_shape'] == (3, 3, 4)


class TestPytreeEdgeCases:
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
