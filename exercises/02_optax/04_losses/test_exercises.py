"""
Tests for Optax Losses Exercises
================================
"""

import pytest
import jax.numpy as jnp
import optax

from exercises import (
    exercise_softmax_cross_entropy,
    exercise_softmax_cross_entropy_int_labels,
    exercise_sigmoid_binary_cross_entropy,
    exercise_l2_loss,
    exercise_huber_loss,
    exercise_cosine_similarity,
    exercise_custom_loss,
    exercise_label_smoothing,
    exercise_combined_loss,
    exercise_task_specific_losses,
)


class TestSoftmaxCrossEntropy:
    def test_manual_matches(self):
        result = exercise_softmax_cross_entropy()
        assert result['manual_matches'] == True


class TestIntLabelsCE:
    def test_matches_onehot(self):
        result = exercise_softmax_cross_entropy_int_labels()
        assert result['matches_onehot'] == True


class TestSigmoidBinaryCE:
    def test_computes_loss(self):
        result = exercise_sigmoid_binary_cross_entropy()
        assert len(result['binary_losses']) == 4


class TestL2Loss:
    def test_mse_matches(self):
        result = exercise_l2_loss()
        assert result['mse_matches'] == True


class TestHuberLoss:
    def test_more_robust(self):
        result = exercise_huber_loss()
        assert result['huber_more_robust'] == True


class TestCosineSimilarity:
    def test_same_vectors(self):
        result = exercise_cosine_similarity()
        assert abs(result['same_vectors'] - 1.0) < 1e-5


class TestCustomLoss:
    def test_focal_reduces_easy(self):
        result = exercise_custom_loss()
        assert result['focal_reduces_easy'] == True


class TestLabelSmoothing:
    def test_smoothing_increases_loss(self):
        result = exercise_label_smoothing()
        assert result['smoothing_increases_loss'] == True


class TestCombinedLoss:
    def test_has_gradients(self):
        result = exercise_combined_loss()
        assert result['has_gradients'] == True


class TestTaskSpecificLosses:
    def test_all_tasks(self):
        result = exercise_task_specific_losses()
        assert len(result['tasks']) > 0


class TestLossEdgeCases:
    def test_perfect_prediction(self):
        """Perfect prediction should have very low loss."""
        logits = jnp.array([[100.0, 0.0, 0.0]])  # Very confident class 0
        labels = jnp.array([0])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        assert float(loss[0]) < 0.01

    def test_wrong_prediction(self):
        """Wrong prediction should have high loss."""
        logits = jnp.array([[100.0, 0.0, 0.0]])  # Very confident class 0
        labels = jnp.array([1])  # But true class is 1
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        assert float(loss[0]) > 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
