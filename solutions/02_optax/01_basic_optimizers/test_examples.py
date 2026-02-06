"""
Tests for Optax Basic Optimizers Examples
=========================================
"""

import pytest
import jax
import jax.numpy as jnp
import optax

from examples import (
    example_sgd,
    example_sgd_momentum,
    example_adam,
    example_adamw,
    example_rmsprop,
    example_adagrad,
    example_chain,
    example_apply_updates,
    example_state_inspection,
    example_custom_optimizer,
)


class TestSGD:
    def test_sgd_update(self):
        result = example_sgd()
        assert result['matches'] == True

    def test_sgd_basic(self):
        optimizer = optax.sgd(0.1)
        params = {'w': jnp.array([1.0])}
        opt_state = optimizer.init(params)
        grads = {'w': jnp.array([1.0])}

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # w_new = 1.0 - 0.1 * 1.0 = 0.9
        assert jnp.allclose(new_params['w'], jnp.array([0.9]))


class TestSGDMomentum:
    def test_momentum_effect(self):
        result = example_sgd_momentum()
        assert result['momentum_effect'] == True


class TestAdam:
    def test_adam_adapts(self):
        result = example_adam()
        assert result['adam_adapts'] == True

    def test_adam_updates_params(self):
        optimizer = optax.adam(0.001)
        params = {'w': jnp.ones(5)}
        opt_state = optimizer.init(params)
        grads = {'w': jnp.ones(5) * 0.1}

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert not jnp.allclose(new_params['w'], params['w'])


class TestAdamW:
    def test_weight_decay(self):
        result = example_adamw()
        assert result['adamw_shrinks'] == True
        assert result['adam_unchanged'] == True


class TestRMSprop:
    def test_normalization(self):
        result = example_rmsprop()
        # RMSprop should normalize updates
        assert result['updates_normalized'] == True


class TestAdagrad:
    def test_learning_rate_decay(self):
        result = example_adagrad()
        assert result['learning_rate_decreases'] == True


class TestChain:
    def test_clipping(self):
        result = example_chain()
        assert result['clipping_applied'] == True


class TestApplyUpdates:
    def test_structure_preserved(self):
        result = example_apply_updates()
        assert result['structure_preserved'] == True

    def test_values_changed(self):
        result = example_apply_updates()
        assert result['values_changed'] == True


class TestStateInspection:
    def test_state_exists(self):
        result = example_state_inspection()
        assert result['num_state_components'] > 0


class TestCustomOptimizer:
    def test_custom_works(self):
        result = example_custom_optimizer()
        assert result['custom_optimizer_works'] == True


class TestOptimizerEdgeCases:
    def test_zero_learning_rate(self):
        """Zero learning rate should not update params."""
        optimizer = optax.sgd(0.0)
        params = {'w': jnp.ones(5)}
        opt_state = optimizer.init(params)
        grads = {'w': jnp.ones(5)}

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert jnp.allclose(new_params['w'], params['w'])

    def test_empty_params(self):
        """Optimizer handles empty params."""
        optimizer = optax.adam(0.001)
        params = {}
        opt_state = optimizer.init(params)
        grads = {}

        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert new_params == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
