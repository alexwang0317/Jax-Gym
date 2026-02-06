"""
Tests for Optax Learning Rate Schedules Examples
================================================
"""

import pytest
import jax.numpy as jnp
import optax

from examples import (
    example_constant_schedule,
    example_exponential_decay,
    example_cosine_decay,
    example_warmup_cosine,
    example_linear_schedule,
    example_piecewise_constant,
    example_join_schedules,
    example_inject_hyperparams,
    example_custom_schedule,
    example_visualize_schedules,
)


class TestConstantSchedule:
    def test_is_constant(self):
        result = example_constant_schedule()
        assert result['is_constant'] == True


class TestExponentialDecay:
    def test_decays(self):
        result = example_exponential_decay()
        assert result['decays_over_time'] == True


class TestCosineDecay:
    def test_starts_at_init(self):
        result = example_cosine_decay()
        assert result['starts_at_init'] == True

    def test_ends_near_zero(self):
        result = example_cosine_decay()
        assert result['ends_near_zero'] == True


class TestWarmupCosine:
    def test_warmup_linear(self):
        result = example_warmup_cosine()
        assert result['warmup_linear'] == True

    def test_decay_after_peak(self):
        result = example_warmup_cosine()
        assert result['decay_after_peak'] == True


class TestLinearSchedule:
    def test_warmup_increases(self):
        result = example_linear_schedule()
        assert result['warmup_increases'] == True

    def test_decay_decreases(self):
        result = example_linear_schedule()
        assert result['decay_decreases'] == True


class TestPiecewiseConstant:
    def test_drops(self):
        result = example_piecewise_constant()
        assert result['initial_lr'] == 0.1
        assert abs(result['after_first_drop'] - 0.01) < 1e-6
        assert abs(result['after_second_drop'] - 0.001) < 1e-6


class TestJoinSchedules:
    def test_phases(self):
        result = example_join_schedules()
        assert result['warmup_phase'] == True
        assert result['constant_phase'] == True
        assert result['decay_phase'] == True


class TestInjectHyperparams:
    def test_trackable(self):
        result = example_inject_hyperparams()
        assert result['lr_trackable'] == True

    def test_lr_changes(self):
        result = example_inject_hyperparams()
        # After 100 steps, should be at peak (warmup was 100 steps)
        assert result['lr_at_step_100'] > result['lr_at_step_0']


class TestCustomSchedule:
    def test_triangle_peaks(self):
        result = example_custom_schedule()
        assert result['triangle_peaks_at_middle'] == True


class TestVisualizeSchedules:
    def test_schedules_created(self):
        result = example_visualize_schedules()
        assert len(result['schedules_compared']) == 5


class TestScheduleEdgeCases:
    def test_schedule_at_zero(self):
        """Schedule should work at step 0."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=0.001,
            warmup_steps=100, decay_steps=1000
        )
        assert schedule(0) == 0.0

    def test_schedule_beyond_decay_steps(self):
        """Schedule should not go negative."""
        schedule = optax.cosine_decay_schedule(
            init_value=0.001, decay_steps=1000, alpha=0.0
        )
        # After decay_steps, should be at minimum
        lr_at_end = schedule(1000)
        lr_beyond = schedule(2000)
        assert lr_at_end >= 0
        assert lr_beyond >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
