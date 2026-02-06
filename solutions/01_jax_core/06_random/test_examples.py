"""
Tests for JAX Random Examples
=============================
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from examples import (
    example_key_creation,
    example_key_splitting,
    example_distributions,
    example_choice,
    example_permutation,
    example_reproducibility,
    example_random_vmap,
    example_random_jit,
    example_dropout,
    example_weight_init,
)


class TestKeyCreation:
    def test_same_seed_same_value(self):
        result = example_key_creation()
        assert result['same_seed_same_value'] == True

    def test_different_seeds(self):
        result = example_key_creation()
        assert result['different_seeds_different'] == True


class TestKeySplitting:
    def test_reused_key(self):
        result = example_key_splitting()
        assert result['reused_key_same'] == True

    def test_split_keys_different(self):
        result = example_key_splitting()
        assert result['split_keys_different'] == True

    def test_multi_split(self):
        result = example_key_splitting()
        assert result['multi_split_count'] == 5
        assert result['all_different'] == True


class TestDistributions:
    def test_uniform_range(self):
        result = example_distributions()
        uniform = result['uniform_01']
        assert jnp.all(uniform >= 0) and jnp.all(uniform < 1)

    def test_truncated(self):
        result = example_distributions()
        assert result['truncated_in_range'] == True

    def test_integers_type(self):
        result = example_distributions()
        assert jnp.issubdtype(result['integers'].dtype, jnp.integer)


class TestChoice:
    def test_batch_indices(self):
        result = example_choice()
        assert result['batch_indices_shape'] == (32,)

    def test_samples_shape(self):
        result = example_choice()
        assert result['samples_from_array'].shape == (3,)

    def test_weighted_favors_early(self):
        result = example_choice()
        counts = result['weighted_counts']
        # First element should have more samples than last
        assert counts[0] > counts[-1]


class TestPermutation:
    def test_permutation_complete(self):
        result = example_permutation()
        perm = result['permutation']
        # Should contain all numbers 0-9
        assert set(perm.tolist()) == set(range(10))

    def test_different_epochs(self):
        result = example_permutation()
        assert result['different_epochs'] == True


class TestReproducibility:
    def test_same_seed_reproduces(self):
        result = example_reproducibility()
        assert result['same_seed_reproduces'] == True

    def test_different_seed_differs(self):
        result = example_reproducibility()
        assert result['different_seed_differs'] == True

    def test_checkpoint(self):
        result = example_reproducibility()
        assert result['checkpoint_works'] == True


class TestRandomVmap:
    def test_batch_shape(self):
        result = example_random_vmap()
        assert result['batch_samples_shape'] == (5, 3)

    def test_samples_different(self):
        result = example_random_vmap()
        assert result['samples_different'] == True


class TestRandomJit:
    def test_results_different(self):
        result = example_random_jit()
        assert result['results_different'] == True

    def test_jit_works(self):
        result = example_random_jit()
        assert result['jit_works'] == True


class TestDropout:
    def test_shape_preserved(self):
        result = example_dropout()
        assert result['dropped_shape'] == (10, 20)

    def test_inference_unchanged(self):
        result = example_dropout()
        assert result['inference_unchanged'] == True

    def test_scale_correct(self):
        result = example_dropout()
        assert result['scale_correct'] == True

    def test_approximate_rate(self):
        result = example_dropout()
        # Should be approximately 0.3, allow variance
        assert 0.1 < result['approx_drop_rate'] < 0.5


class TestWeightInit:
    def test_shape(self):
        result = example_weight_init()
        assert result['layer0_w_shape'] == (784, 256)

    def test_mean_near_zero(self):
        result = example_weight_init()
        assert abs(result['layer0_w_mean']) < 0.01

    def test_std_close(self):
        result = example_weight_init()
        assert result['std_close'] == True


class TestRandomEdgeCases:
    def test_split_preserves_type(self):
        """Split should preserve key type."""
        key = random.key(0)
        key1, key2 = random.split(key)
        assert key1.dtype == key.dtype
        assert key2.dtype == key.dtype

    def test_empty_shape(self):
        """Empty shape should return scalar."""
        key = random.key(0)
        scalar = random.normal(key, shape=())
        assert scalar.shape == ()

    def test_batched_keys(self):
        """Multiple splits should be independent."""
        key = random.key(0)
        keys = random.split(key, 10)
        vals = jax.vmap(lambda k: random.normal(k))(keys)
        # All should be different
        unique = len(set(vals.tolist()))
        assert unique == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
