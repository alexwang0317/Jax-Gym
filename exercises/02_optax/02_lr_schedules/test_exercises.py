"""
Tests for Optax Learning Rate Schedules Exercises
=================================================
"""

import pytest
import jax.numpy as jnp
import optax

from exercises import (
    exercise_constant_schedule,
    exercise_exponential_decay,
    exercise_cosine_decay,
    exercise_warmup_cosine,
    exercise_linear_schedule,
    exercise_piecewise_constant,
    exercise_join_schedules,
    exercise_inject_hyperparams,
    exercise_custom_schedule,
    exercise_visualize_schedules,
)


class TestConstantSchedule:
    def test_is_constant(self):
        result = exercise_constant_schedule()
        assert result['is_constant'] == True


class TestExponentialDecay:
    def test_decays(self):
        result = exercise_exponential_decay()
        assert result['decays_over_time'] == True


class TestCosineDecay:
    def test_starts_at_init(self):
        result = exercise_cosine_decay()
        assert result['starts_at_init'] == True

    def test_ends_near_zero(self):
        result = exercise_cosine_decay()
        assert result['ends_near_zero'] == True


class TestWarmupCosine:
    def test_warmup_linear(self):
        result = exercise_warmup_cosine()
        assert result['warmup_linear'] == True

    def test_decay_after_peak(self):
        result = exercise_warmup_cosine()
        assert result['decay_after_peak'] == True


class TestLinearSchedule:
    def test_warmup_increases(self):
        result = exercise_linear_schedule()
        assert result['warmup_increases'] == True

    def test_decay_decreases(self):
        result = exercise_linear_schedule()
        assert result['decay_decreases'] == True


class TestPiecewiseConstant:
    def test_drops(self):
        result = exercise_piecewise_constant()
        assert result['initial_lr'] == 0.1
        assert abs(result['after_first_drop'] - 0.01) < 1e-6
        assert abs(result['after_second_drop'] - 0.001) < 1e-6


class TestJoinSchedules:
    def test_phases(self):
        result = exercise_join_schedules()
        assert result['warmup_phase'] == True
        assert result['constant_phase'] == True
        assert result['decay_phase'] == True


class TestInjectHyperparams:
    def test_trackable(self):
        result = exercise_inject_hyperparams()
        assert result['lr_trackable'] == True

    def test_lr_changes(self):
        result = exercise_inject_hyperparams()
        # After 100 steps, should be at peak (warmup was 100 steps)
        assert result['lr_at_step_100'] > result['lr_at_step_0']


class TestCustomSchedule:
    def test_triangle_peaks(self):
        result = exercise_custom_schedule()
        assert result['triangle_peaks_at_middle'] == True


class TestVisualizeSchedules:
    def test_schedules_created(self):
        result = exercise_visualize_schedules()
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
