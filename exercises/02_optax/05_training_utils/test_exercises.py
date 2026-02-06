"""
Tests for Optax Training Utilities Exercises
============================================
"""

import pytest
import jax.numpy as jnp

from exercises import (
    exercise_basic_training_loop,
    exercise_train_state,
    exercise_apply_updates_workflow,
    exercise_dataloader_integration,
    exercise_train_eval_mode,
    exercise_metric_tracking,
    exercise_checkpointing,
    exercise_early_stopping,
    exercise_gradient_accumulation,
    exercise_multi_gpu_basics,
)


class TestBasicTrainingLoop:
    def test_loss_decreased(self):
        result = exercise_basic_training_loop()
        assert result['loss_decreased'] == True


class TestTrainState:
    def test_step_increment(self):
        result = exercise_train_state()
        assert result['after_update_step'] == result['initial_step'] + 1

    def test_params_updated(self):
        result = exercise_train_state()
        assert result['params_updated'] == True


class TestApplyUpdatesWorkflow:
    def test_structure_preserved(self):
        result = exercise_apply_updates_workflow()
        assert result['structure_preserved'] == True

    def test_params_changed(self):
        result = exercise_apply_updates_workflow()
        assert result['params_changed'] == True


class TestDataLoaderIntegration:
    def test_batches(self):
        result = exercise_dataloader_integration()
        assert result['batches_seen'] == 5


class TestTrainEvalMode:
    def test_eval_deterministic(self):
        result = exercise_train_eval_mode()
        assert result['eval_deterministic'] == True


class TestMetricTracking:
    def test_total_steps(self):
        result = exercise_metric_tracking()
        assert result['total_steps'] == 100


class TestCheckpointing:
    def test_params_match(self):
        result = exercise_checkpointing()
        assert result['params_match'] == True

    def test_step_restored(self):
        result = exercise_checkpointing()
        assert result['saved_step'] == result['restored_step']


class TestEarlyStopping:
    def test_triggered(self):
        result = exercise_early_stopping()
        assert result['early_stopping_triggered'] == True


class TestGradientAccumulation:
    def test_accumulation(self):
        result = exercise_gradient_accumulation()
        # Should accumulate for 'accumulation_steps' before actual update
        assert result['accumulation_steps'] == 4


class TestMultiGPUBasics:
    def test_devices_available(self):
        result = exercise_multi_gpu_basics()
        assert result['num_devices'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
