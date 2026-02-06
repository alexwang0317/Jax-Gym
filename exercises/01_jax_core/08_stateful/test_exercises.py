"""
Tests for JAX Stateful Computation Exercises
============================================
"""

import pytest
import jax
import jax.numpy as jnp

from exercises import (
    exercise_functional_state,
    exercise_state_pytree,
    exercise_accumulator,
    exercise_counter,
    exercise_running_stats,
    exercise_stateful_rng,
    exercise_optimizer_state,
    exercise_batchnorm_stats,
    exercise_memoization,
    exercise_checkpointing,
)


class TestFunctionalState:
    def test_counter(self):
        result = exercise_functional_state()
        assert result['counter_final'] == 5

    def test_process(self):
        result = exercise_functional_state()
        assert jnp.allclose(result['process_final_state'], 6.0)
        assert result['outputs'] == [2.0, 4.0, 6.0]


class TestStatePytree:
    def test_counts(self):
        result = exercise_state_pytree()
        assert result['final_count'] == 8

    def test_statistics(self):
        result = exercise_state_pytree()
        assert jnp.allclose(result['mean'], 3.875)
        assert jnp.allclose(result['min'], 1.0)
        assert jnp.allclose(result['max'], 9.0)

    def test_optimizer_state(self):
        result = exercise_state_pytree()
        assert result['opt_state_step'] == 0
        assert result['opt_state_momentum_shape'] == (10,)


class TestAccumulator:
    def test_simple_total(self):
        result = exercise_accumulator()
        assert jnp.allclose(result['simple_total'], 15.0)

    def test_jit_total(self):
        result = exercise_accumulator()
        assert jnp.allclose(result['jit_total'], 15.0)

    def test_variance(self):
        result = exercise_accumulator()
        assert jnp.allclose(result['mean'], 3.0)
        assert jnp.allclose(result['variance'], 2.0)


class TestCounter:
    def test_manual_count(self):
        result = exercise_counter()
        assert result['manual_count'] == 5

    def test_batch_count(self):
        result = exercise_counter()
        assert result['batch_count'] == 100


class TestRunningStats:
    def test_mean_matches(self):
        result = exercise_running_stats()
        assert result['mean_matches'] == True

    def test_var_matches(self):
        result = exercise_running_stats()
        assert result['var_matches'] == True


class TestStatefulRNG:
    def test_samples_different(self):
        result = exercise_stateful_rng()
        assert result['samples_different'] == True

    def test_reproducible(self):
        result = exercise_stateful_rng()
        assert result['reproducible'] == True


class TestOptimizerState:
    def test_step(self):
        result = exercise_optimizer_state()
        assert result['final_step'] == 10

    def test_shapes(self):
        result = exercise_optimizer_state()
        assert result['m_w_shape'] == (3, 4)
        assert result['v_b_shape'] == (4,)

    def test_params_updated(self):
        result = exercise_optimizer_state()
        # After 10 Adam updates, params should have changed from initial 1s and 0s
        assert result['params_w_mean'] < 1.0
        assert result['params_b_mean'] < 0.0


class TestBatchNormStats:
    def test_num_updates(self):
        result = exercise_batchnorm_stats()
        assert result['num_updates'] == 10

    def test_output_shape(self):
        result = exercise_batchnorm_stats()
        assert result['test_output_shape'] == (16, 5)


class TestMemoization:
    def test_lookup_table(self):
        result = exercise_memoization()
        assert result['lookup_table_size'] == 100


class TestCheckpointing:
    def test_step_preserved(self):
        result = exercise_checkpointing()
        assert result['original_step'] == result['restored_step']

    def test_params_preserved(self):
        result = exercise_checkpointing()
        assert result['params_match'] == True

    def test_loss_preserved(self):
        result = exercise_checkpointing()
        assert result['loss_match'] == True


class TestStatefulEdgeCases:
    def test_state_with_none(self):
        """State can include None values."""
        from typing import NamedTuple, Optional

        class State(NamedTuple):
            value: int
            optional: Optional[jnp.ndarray]

        s = State(value=1, optional=None)
        assert s.value == 1
        assert s.optional is None

    def test_nested_state(self):
        """Nested state structures work."""
        state = {
            'level1': {
                'level2': {
                    'value': jnp.array(1.0)
                }
            }
        }

        def update(state):
            return {
                'level1': {
                    'level2': {
                        'value': state['level1']['level2']['value'] + 1
                    }
                }
            }

        new_state = update(state)
        assert jnp.allclose(new_state['level1']['level2']['value'], 2.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
