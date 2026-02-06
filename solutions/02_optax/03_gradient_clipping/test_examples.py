"""
Tests for Optax Gradient Clipping Examples
==========================================
"""

import pytest
import jax.numpy as jnp
import optax

from examples import (
    example_clip_by_global_norm,
    example_clip_by_value,
    example_simple_clip,
    example_clipping_with_optimizer,
    example_adaptive_clipping,
    example_per_layer_clipping,
    example_gradient_monitoring,
    example_clipping_strategies,
    example_rnn_clipping,
    example_transformer_clipping,
)


class TestClipByGlobalNorm:
    def test_large_clipped(self):
        result = example_clip_by_global_norm()
        assert result['large_clipped_to_max'] == True

    def test_small_unchanged(self):
        result = example_clip_by_global_norm()
        assert abs(result['small_norm_before'] - result['small_norm_after']) < 1e-5


class TestClipByValue:
    def test_clipping(self):
        result = example_clip_by_value()
        assert result['matches'] == True


class TestSimpleClip:
    def test_in_range(self):
        result = example_simple_clip()
        assert result['in_range'] == True


class TestClippingWithOptimizer:
    def test_reduces_norm(self):
        result = example_clipping_with_optimizer()
        assert result['clipping_reduces_norm'] == True


class TestAdaptiveClipping:
    def test_adaptive(self):
        result = example_adaptive_clipping()
        assert result['small_grad_unchanged'] == True
        assert result['large_grad_clipped'] == True


class TestPerLayerClipping:
    def test_embedding_within_limit(self):
        result = example_per_layer_clipping()
        assert result['embedding_within_limit'] == True


class TestGradientMonitoring:
    def test_stats_computed(self):
        result = example_gradient_monitoring()
        assert 'mean_norm' in result
        assert 'max_norm' in result
        assert 'suggested_clip' in result


class TestClippingStrategies:
    def test_global_preserves_direction(self):
        result = example_clipping_strategies()
        assert result['global_preserves_direction'] == True


class TestRNNClipping:
    def test_was_clipped(self):
        result = example_rnn_clipping()
        assert result['was_clipped'] == True

    def test_clipped_norm(self):
        result = example_rnn_clipping()
        assert result['clipped_norm'] <= result['rnn_clip_value'] + 1e-5


class TestTransformerClipping:
    def test_clip_value(self):
        result = example_transformer_clipping()
        assert result['transformer_clip_value'] == 1.0


class TestClippingEdgeCases:
    def test_zero_gradients(self):
        """Zero gradients should not cause issues."""
        clip = optax.clip_by_global_norm(1.0)
        params = {'w': jnp.zeros((3, 4))}
        state = clip.init(params)
        grads = {'w': jnp.zeros((3, 4))}

        clipped, _ = clip.update(grads, state, params)
        assert jnp.allclose(clipped['w'], jnp.zeros((3, 4)))

    def test_very_small_gradients(self):
        """Very small gradients should not be scaled up."""
        clip = optax.clip_by_global_norm(1.0)
        params = {'w': jnp.ones((3, 4))}
        state = clip.init(params)
        grads = {'w': jnp.ones((3, 4)) * 1e-10}

        clipped, _ = clip.update(grads, state, params)
        # Should be unchanged (not scaled up)
        assert jnp.allclose(clipped['w'], grads['w'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
