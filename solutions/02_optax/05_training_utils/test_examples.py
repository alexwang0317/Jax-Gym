"""
Tests for Optax Training Utilities Examples
==========================================
"""

import pytest
import jax.numpy as jnp

from examples import (
    example_basic_training_loop,
    example_train_state,
    example_apply_updates_workflow,
    example_dataloader_integration,
    example_train_eval_mode,
    example_metric_tracking,
    example_checkpointing,
    example_early_stopping,
    example_gradient_accumulation,
    example_multi_gpu_basics,
)


class TestBasicTrainingLoop:
    def test_loss_decreased(self):
        result = example_basic_training_loop()
        assert result['loss_decreased'] == True


class TestTrainState:
    def test_step_increment(self):
        result = example_train_state()
        assert result['after_update_step'] == result['initial_step'] + 1

    def test_params_updated(self):
        result = example_train_state()
        assert result['params_updated'] == True


class TestApplyUpdatesWorkflow:
    def test_structure_preserved(self):
        result = example_apply_updates_workflow()
        assert result['structure_preserved'] == True

    def test_params_changed(self):
        result = example_apply_updates_workflow()
        assert result['params_changed'] == True


class TestDataLoaderIntegration:
    def test_batches(self):
        result = example_dataloader_integration()
        assert result['batches_seen'] == 5


class TestTrainEvalMode:
    def test_eval_deterministic(self):
        result = example_train_eval_mode()
        assert result['eval_deterministic'] == True


class TestMetricTracking:
    def test_total_steps(self):
        result = example_metric_tracking()
        assert result['total_steps'] == 100


class TestCheckpointing:
    def test_params_match(self):
        result = example_checkpointing()
        assert result['params_match'] == True

    def test_step_restored(self):
        result = example_checkpointing()
        assert result['saved_step'] == result['restored_step']


class TestEarlyStopping:
    def test_triggered(self):
        result = example_early_stopping()
        assert result['early_stopping_triggered'] == True


class TestGradientAccumulation:
    def test_accumulation(self):
        result = example_gradient_accumulation()
        # Should accumulate for 'accumulation_steps' before actual update
        assert result['accumulation_steps'] == 4


class TestMultiGPUBasics:
    def test_devices_available(self):
        result = example_multi_gpu_basics()
        assert result['num_devices'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
