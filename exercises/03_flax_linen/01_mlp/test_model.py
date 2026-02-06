"""
Tests for MLP Model
===================
"""

import pytest
import jax
import jax.numpy as jnp

from model import MLP, MLPWithBatchNorm, MLPWithDropout, MLPClassifier
from train import create_train_state, train_step, eval_step, train_mlp


class TestMLPModel:
    def test_basic_mlp_forward(self):
        """Test basic MLP forward pass."""
        model = MLP(features=[64, 32], output_dim=10)
        rng = jax.random.key(0)

        x = jnp.ones((4, 20))  # batch of 4, input dim 20
        variables = model.init(rng, x)
        output = model.apply(variables, x)

        assert output.shape == (4, 10)

    def test_mlp_with_batchnorm(self):
        """Test MLP with BatchNorm."""
        model = MLPWithBatchNorm(features=[64, 32], output_dim=10)
        rng = jax.random.key(0)

        x = jnp.ones((4, 20))
        variables = model.init(rng, x, training=True)

        # Should have batch_stats
        assert 'batch_stats' in variables

        # Training mode
        output, new_state = model.apply(
            variables, x, training=True, mutable=['batch_stats']
        )
        assert output.shape == (4, 10)

        # Eval mode
        output_eval = model.apply(variables, x, training=False)
        assert output_eval.shape == (4, 10)

    def test_mlp_with_dropout(self):
        """Test MLP with Dropout."""
        model = MLPWithDropout(features=[64, 32], output_dim=10, dropout_rate=0.5)
        rng = jax.random.key(0)

        x = jnp.ones((4, 20))
        variables = model.init(rng, x, training=True)

        # Training mode (dropout active)
        output_train = model.apply(
            variables, x, training=True,
            rngs={'dropout': jax.random.key(1)}
        )

        # Eval mode (dropout inactive)
        output_eval = model.apply(variables, x, training=False)

        assert output_train.shape == (4, 10)
        assert output_eval.shape == (4, 10)

    def test_mlp_classifier(self):
        """Test MLPClassifier."""
        model = MLPClassifier(hidden_dims=[128, 64], num_classes=10)
        rng = jax.random.key(0)

        # Test with flattened input
        x_flat = jnp.ones((4, 784))
        variables = model.init(rng, x_flat, training=True)
        output = model.apply(variables, x_flat, training=False)
        assert output.shape == (4, 10)

        # Test with image input (should be flattened)
        x_img = jnp.ones((4, 28, 28))
        output_img = model.apply(variables, x_img, training=False)
        assert output_img.shape == (4, 10)


class TestMLPTraining:
    def test_create_train_state(self):
        """Test training state creation."""
        model = MLPClassifier(hidden_dims=[64, 32], num_classes=10)
        rng = jax.random.key(0)

        state, batch_stats = create_train_state(rng, model, (100,), 0.001)

        assert state.step == 0
        assert 'Dense_0' in state.params

    def test_train_step(self):
        """Test single training step."""
        model = MLPClassifier(hidden_dims=[64, 32], num_classes=10)
        rng = jax.random.key(0)

        state, batch_stats = create_train_state(rng, model, (100,), 0.001)

        # Create batch
        x = jnp.ones((8, 100))
        y = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        batch = (x, y)

        # Train step
        new_state, new_batch_stats, metrics = train_step(
            state, batch_stats, model, batch
        )

        assert new_state.step == 1
        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_eval_step(self):
        """Test evaluation step."""
        model = MLPClassifier(hidden_dims=[64, 32], num_classes=10)
        rng = jax.random.key(0)

        state, batch_stats = create_train_state(rng, model, (100,), 0.001)

        # Create batch
        x = jnp.ones((8, 100))
        y = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        batch = (x, y)

        metrics = eval_step(state, batch_stats, model, batch)

        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_training_loop(self):
        """Test full training loop."""
        state, batch_stats, history = train_mlp(
            num_epochs=3,
            batch_size=32,
            learning_rate=0.01,
            input_dim=50,
            hidden_dims=(32, 16),
            num_classes=5,
            verbose=False
        )

        # Check training happened
        assert len(history['train_loss']) == 3
        assert len(history['test_acc']) == 3

        # Loss should generally decrease (not guaranteed with random data)
        # Just check it's finite
        assert all(jnp.isfinite(l) for l in history['train_loss'])


class TestMLPEdgeCases:
    def test_single_hidden_layer(self):
        """MLP with single hidden layer."""
        model = MLP(features=[64], output_dim=10)
        rng = jax.random.key(0)
        x = jnp.ones((4, 20))
        variables = model.init(rng, x)
        output = model.apply(variables, x)
        assert output.shape == (4, 10)

    def test_different_activations(self):
        """Test with different activation functions."""
        import flax.linen as nn

        for activation in [nn.relu, nn.tanh, nn.sigmoid]:
            model = MLP(features=[64, 32], output_dim=10, activation=activation)
            rng = jax.random.key(0)
            x = jnp.ones((4, 20))
            variables = model.init(rng, x)
            output = model.apply(variables, x)
            assert output.shape == (4, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
