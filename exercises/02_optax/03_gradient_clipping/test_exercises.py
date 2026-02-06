"""
Tests for Optax Gradient Clipping Exercises
============================================
"""

import pytest
import jax.numpy as jnp
import optax

from exercises import (
    exercise_clip_by_global_norm,
    exercise_clip_by_value,
    exercise_simple_clip,
    exercise_clipping_with_optimizer,
    exercise_adaptive_clipping,
    exercise_per_layer_clipping,
    exercise_gradient_monitoring,
    exercise_clipping_strategies,
    exercise_rnn_clipping,
    exercise_transformer_clipping,
)


class TestClipByGlobalNorm:
    def test_large_clipped(self):
        result = exercise_clip_by_global_norm()
        assert result['large_clipped_to_max'] == True

    def test_small_unchanged(self):
        result = exercise_clip_by_global_norm()
        assert abs(result['small_norm_before'] - result['small_norm_after']) < 1e-5


class TestClipByValue:
    def test_clipping(self):
        result = exercise_clip_by_value()
        assert result['matches'] == True


class TestSimpleClip:
    def test_in_range(self):
        result = exercise_simple_clip()
        assert result['in_range'] == True


class TestClippingWithOptimizer:
    def test_reduces_norm(self):
        result = exercise_clipping_with_optimizer()
        assert result['clipping_reduces_norm'] == True


class TestAdaptiveClipping:
    def test_adaptive(self):
        result = exercise_adaptive_clipping()
        assert result['small_grad_unchanged'] == True
        assert result['large_grad_clipped'] == True


class TestPerLayerClipping:
    def test_embedding_within_limit(self):
        result = exercise_per_layer_clipping()
        assert result['embedding_within_limit'] == True


class TestGradientMonitoring:
    def test_stats_computed(self):
        result = exercise_gradient_monitoring()
        assert 'mean_norm' in result
        assert 'max_norm' in result
        assert 'suggested_clip' in result


class TestClippingStrategies:
    def test_global_preserves_direction(self):
        result = exercise_clipping_strategies()
        assert result['global_preserves_direction'] == True


class TestRNNClipping:
    def test_was_clipped(self):
        result = exercise_rnn_clipping()
        assert result['was_clipped'] == True

    def test_clipped_norm(self):
        result = exercise_rnn_clipping()
        assert result['clipped_norm'] <= result['rnn_clip_value'] + 1e-5


class TestTransformerClipping:
    def test_clip_value(self):
        result = exercise_transformer_clipping()
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
