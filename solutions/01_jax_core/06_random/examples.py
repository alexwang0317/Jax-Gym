"""
JAX Random Number Generation - 10 Examples
============================================

JAX uses explicit PRNG keys for reproducibility and JIT compatibility.
No global random state - keys must be passed explicitly.

Key concepts:
- Create keys with jax.random.key(seed)
- Split keys before each use
- Same key = same random numbers (deterministic)

Reference: https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
"""

import jax
import jax.numpy as jnp
from jax import random


# =============================================================================
# Example 1: Key Creation with jax.random.key
# =============================================================================
def example_key_creation():
    """
    Create PRNG keys from integer seeds.
    Keys are the basis of all randomness in JAX.
    """
    # Create a key from a seed
    key = random.key(0)

    # Different seeds give different keys
    key1 = random.key(42)
    key2 = random.key(123)

    # Same seed = same key = same random numbers
    key_a = random.key(42)
    key_b = random.key(42)

    val_a = random.uniform(key_a)
    val_b = random.uniform(key_b)

    # Keys can be any integer
    key_large = random.key(2**31 - 1)

    return {
        'key_type': type(key).__name__,
        'key_dtype': key.dtype,
        'same_seed_same_value': jnp.allclose(val_a, val_b),
        'different_seeds_different': not jnp.allclose(
            random.uniform(key1),
            random.uniform(key2)
        )
    }


# =============================================================================
# Example 2: Key Splitting
# =============================================================================
def example_key_splitting():
    """
    Split keys to get independent random streams.
    NEVER reuse a key - always split!

    Pattern: key, subkey = random.split(key)
    """
    key = random.key(0)

    # WRONG: Reusing the same key gives same numbers!
    wrong_val1 = random.normal(key)
    wrong_val2 = random.normal(key)  # Same as wrong_val1!

    # RIGHT: Split before each use
    key, subkey1 = random.split(key)
    val1 = random.normal(subkey1)

    key, subkey2 = random.split(key)
    val2 = random.normal(subkey2)  # Different from val1

    # Split into multiple keys at once
    key = random.key(42)
    keys = random.split(key, num=5)  # 5 independent keys

    vals = [random.uniform(k) for k in keys]

    # Named subkeys pattern (readable)
    key = random.key(0)
    key, key_init, key_dropout, key_sample = random.split(key, 4)

    return {
        'reused_key_same': jnp.allclose(wrong_val1, wrong_val2),
        'split_keys_different': not jnp.allclose(val1, val2),
        'multi_split_count': len(keys),
        'all_different': len(set([float(v) for v in vals])) == 5
    }


# =============================================================================
# Example 3: Uniform and Normal Distributions
# =============================================================================
def example_distributions():
    """
    Generate random numbers from common distributions.
    """
    key = random.key(42)

    # Uniform [0, 1)
    key, subkey = random.split(key)
    uniform_01 = random.uniform(subkey, shape=(5,))

    # Uniform [a, b)
    key, subkey = random.split(key)
    uniform_range = random.uniform(subkey, shape=(5,), minval=-1.0, maxval=1.0)

    # Standard normal (mean=0, std=1)
    key, subkey = random.split(key)
    normal_std = random.normal(subkey, shape=(5,))

    # Normal with custom mean and std
    key, subkey = random.split(key)
    mean, std = 5.0, 2.0
    normal_custom = mean + std * random.normal(subkey, shape=(5,))

    # Truncated normal (values clipped to range)
    key, subkey = random.split(key)
    truncated = random.truncated_normal(subkey, lower=-2, upper=2, shape=(5,))

    # Exponential
    key, subkey = random.split(key)
    exponential = random.exponential(subkey, shape=(5,))

    # Integers
    key, subkey = random.split(key)
    integers = random.randint(subkey, shape=(5,), minval=0, maxval=10)

    return {
        'uniform_01': uniform_01,
        'uniform_range': uniform_range,
        'normal_std': normal_std,
        'normal_custom_mean': jnp.mean(normal_custom),
        'truncated_in_range': jnp.all((truncated >= -2) & (truncated <= 2)),
        'integers': integers
    }


# =============================================================================
# Example 4: Random Choice for Sampling
# =============================================================================
def example_choice():
    """
    Sample from arrays with or without replacement.
    Useful for minibatching, bootstrapping, etc.
    """
    key = random.key(42)

    # Sample from range
    key, subkey = random.split(key)
    indices = random.choice(subkey, 100, shape=(10,))  # 10 samples from 0-99

    # Sample from array
    data = jnp.array([10, 20, 30, 40, 50])
    key, subkey = random.split(key)
    samples = random.choice(subkey, data, shape=(3,))

    # Sample without replacement
    key, subkey = random.split(key)
    unique_samples = random.choice(subkey, data, shape=(3,), replace=False)

    # Weighted sampling (non-uniform)
    weights = jnp.array([0.5, 0.3, 0.1, 0.05, 0.05])  # Favor first elements
    key, subkey = random.split(key)
    weighted = random.choice(subkey, data, shape=(100,), p=weights)

    # Count occurrences (should favor lower indices)
    counts = jnp.array([jnp.sum(weighted == v) for v in data])

    # Minibatch sampling pattern
    dataset_size = 1000
    batch_size = 32
    key, subkey = random.split(key)
    batch_indices = random.choice(subkey, dataset_size, shape=(batch_size,), replace=False)

    return {
        'random_indices': indices[:5],
        'samples_from_array': samples,
        'unique_samples': unique_samples,
        'weighted_counts': counts,
        'batch_indices_shape': batch_indices.shape
    }


# =============================================================================
# Example 5: Permutation for Shuffling
# =============================================================================
def example_permutation():
    """
    Shuffle arrays or generate random permutations.
    Essential for data shuffling in training.
    """
    key = random.key(42)

    # Random permutation of indices
    key, subkey = random.split(key)
    perm = random.permutation(subkey, 10)  # Permutation of 0-9

    # Shuffle an array directly
    data = jnp.array([10, 20, 30, 40, 50])
    key, subkey = random.split(key)
    shuffled = random.permutation(subkey, data)

    # Shuffle dataset for training
    X = jnp.arange(100).reshape(20, 5)  # 20 samples, 5 features
    y = jnp.arange(20)

    key, subkey = random.split(key)
    perm_idx = random.permutation(subkey, len(X))
    X_shuffled = X[perm_idx]
    y_shuffled = y[perm_idx]

    # Epoch-wise shuffling pattern
    def shuffle_data(key, X, y):
        perm = random.permutation(key, len(X))
        return X[perm], y[perm]

    key, subkey = random.split(key)
    X_epoch1, y_epoch1 = shuffle_data(subkey, X, y)
    key, subkey = random.split(key)
    X_epoch2, y_epoch2 = shuffle_data(subkey, X, y)

    return {
        'permutation': perm,
        'shuffled_data': shuffled,
        'X_shuffled_shape': X_shuffled.shape,
        'different_epochs': not jnp.allclose(y_epoch1, y_epoch2)
    }


# =============================================================================
# Example 6: Reproducibility Patterns
# =============================================================================
def example_reproducibility():
    """
    Patterns for reproducible random number generation.
    """
    # Same seed = same results
    def experiment(seed):
        key = random.key(seed)
        key, k1, k2 = random.split(key, 3)
        a = random.normal(k1, (3,))
        b = random.uniform(k2, (3,))
        return a, b

    result1 = experiment(42)
    result2 = experiment(42)
    result3 = experiment(123)  # Different seed

    same_seed_same = (
        jnp.allclose(result1[0], result2[0]) and
        jnp.allclose(result1[1], result2[1])
    )

    different_seed_different = not (
        jnp.allclose(result1[0], result3[0]) or
        jnp.allclose(result1[1], result3[1])
    )

    # Checkpoint and resume pattern
    key = random.key(0)
    key, subkey = random.split(key)
    val1 = random.normal(subkey)

    # Save key state here (e.g., serialize key)
    saved_key = key

    key, subkey = random.split(key)
    val2 = random.normal(subkey)

    # Resume from saved key
    resumed_key = saved_key
    resumed_key, subkey = random.split(resumed_key)
    val2_reproduced = random.normal(subkey)

    return {
        'same_seed_reproduces': same_seed_same,
        'different_seed_differs': different_seed_different,
        'checkpoint_works': jnp.allclose(val2, val2_reproduced)
    }


# =============================================================================
# Example 7: Random in vmap (Unique Keys per Batch)
# =============================================================================
def example_random_vmap():
    """
    Using random in vmap requires one key per batch element.
    """
    def sample_fn(key):
        """Function that uses randomness."""
        return random.normal(key, (3,))

    # WRONG: Same key for all batch elements
    key = random.key(42)
    batch_size = 5
    # wrong_samples = jax.vmap(sample_fn)(jnp.tile(key, (batch_size, 1)))
    # This would give same samples!

    # RIGHT: Split keys for each element
    key = random.key(42)
    keys = random.split(key, batch_size)
    correct_samples = jax.vmap(sample_fn)(keys)

    # All samples should be different
    all_different = not jnp.allclose(correct_samples[0], correct_samples[1])

    # Pattern for batched random operations
    def batched_dropout(key, x, rate=0.5):
        """Apply dropout to a batch."""
        keys = random.split(key, len(x))
        def single_dropout(k, xi):
            mask = random.bernoulli(k, 1 - rate, xi.shape)
            return xi * mask / (1 - rate)
        return jax.vmap(single_dropout)(keys, x)

    key, subkey = random.split(key)
    x_batch = jnp.ones((4, 10))
    dropped = batched_dropout(subkey, x_batch)

    return {
        'batch_samples_shape': correct_samples.shape,
        'samples_different': all_different,
        'dropout_output_shape': dropped.shape
    }


# =============================================================================
# Example 8: Random in JIT (Passing Keys Explicitly)
# =============================================================================
def example_random_jit():
    """
    JIT-compiled functions must receive keys as arguments.
    Keys cannot be created inside JIT (would break reproducibility).
    """
    # WRONG: Creating key inside JIT
    # @jax.jit
    # def bad_fn():
    #     key = random.key(42)  # Same key every call!
    #     return random.normal(key)

    # RIGHT: Pass key as argument
    @jax.jit
    def good_fn(key, x):
        noise = random.normal(key, x.shape)
        return x + 0.1 * noise

    key = random.key(0)
    x = jnp.ones(5)

    key, subkey = random.split(key)
    result1 = good_fn(subkey, x)

    key, subkey = random.split(key)
    result2 = good_fn(subkey, x)  # Different noise

    # Pattern for JIT with random state
    @jax.jit
    def train_step(key, params, x, y):
        key, dropout_key = random.split(key)
        # Use dropout_key for dropout
        # Compute loss and gradients...
        return key, params  # Return updated key

    # Initialize
    key = random.key(42)
    params = {'w': jnp.ones((3, 4))}
    x, y = jnp.ones((2, 3)), jnp.ones((2, 4))

    # Each step advances the key
    key, params = train_step(key, params, x, y)
    key, params = train_step(key, params, x, y)

    return {
        'results_different': not jnp.allclose(result1, result2),
        'jit_works': True
    }


# =============================================================================
# Example 9: Dropout Implementation
# =============================================================================
def example_dropout():
    """
    Implement dropout using JAX random.
    Dropout randomly zeros elements during training.
    """
    def dropout(key, x, rate, training=True):
        """
        Apply dropout to input x.

        Args:
            key: PRNG key
            x: Input array
            rate: Dropout rate (probability of dropping)
            training: If False, no dropout applied

        Returns:
            x with dropout applied (scaled by 1/(1-rate))
        """
        if not training or rate == 0:
            return x

        keep_rate = 1 - rate
        mask = random.bernoulli(key, keep_rate, x.shape)
        return jnp.where(mask, x / keep_rate, 0)

    x = jnp.ones((10, 20))
    key = random.key(42)

    # Training mode
    key, subkey = random.split(key)
    dropped = dropout(subkey, x, rate=0.3, training=True)

    # Inference mode (no dropout)
    no_drop = dropout(key, x, rate=0.3, training=False)

    # Verify scaling: non-zero values should be scaled by 1/(1-rate)
    non_zero_vals = dropped[dropped != 0]
    expected_scale = 1 / (1 - 0.3)

    # Count dropped elements
    dropped_count = jnp.sum(dropped == 0)
    total_elements = x.size

    return {
        'dropped_shape': dropped.shape,
        'approx_drop_rate': float(dropped_count / total_elements),
        'scale_correct': jnp.allclose(non_zero_vals[0], expected_scale),
        'inference_unchanged': jnp.allclose(no_drop, x)
    }


# =============================================================================
# Example 10: Weight Initialization Patterns
# =============================================================================
def example_weight_init():
    """
    Common weight initialization patterns for neural networks.
    """
    def xavier_uniform(key, shape):
        """Xavier/Glorot uniform initialization."""
        fan_in, fan_out = shape[-2], shape[-1]
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(key, shape, minval=-limit, maxval=limit)

    def xavier_normal(key, shape):
        """Xavier/Glorot normal initialization."""
        fan_in, fan_out = shape[-2], shape[-1]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        return random.normal(key, shape) * std

    def kaiming_uniform(key, shape):
        """Kaiming/He uniform initialization (for ReLU)."""
        fan_in = shape[-2]
        limit = jnp.sqrt(6.0 / fan_in)
        return random.uniform(key, shape, minval=-limit, maxval=limit)

    def kaiming_normal(key, shape):
        """Kaiming/He normal initialization (for ReLU)."""
        fan_in = shape[-2]
        std = jnp.sqrt(2.0 / fan_in)
        return random.normal(key, shape) * std

    def lecun_normal(key, shape):
        """LeCun normal initialization."""
        fan_in = shape[-2]
        std = jnp.sqrt(1.0 / fan_in)
        return random.normal(key, shape) * std

    # Initialize MLP layers
    key = random.key(42)
    layer_shapes = [(784, 256), (256, 128), (128, 10)]

    mlp_params = {}
    for i, shape in enumerate(layer_shapes):
        key, k1, k2 = random.split(key, 3)
        mlp_params[f'layer{i}'] = {
            'w': xavier_normal(k1, shape),
            'b': jnp.zeros(shape[1])
        }

    # Check initialization statistics
    w0 = mlp_params['layer0']['w']
    w0_mean = jnp.mean(w0)
    w0_std = jnp.std(w0)

    # Expected std for Xavier normal: sqrt(2 / (784 + 256))
    expected_std = jnp.sqrt(2.0 / (784 + 256))

    return {
        'layer0_w_shape': w0.shape,
        'layer0_w_mean': float(w0_mean),
        'layer0_w_std': float(w0_std),
        'expected_std': float(expected_std),
        'std_close': jnp.abs(w0_std - expected_std) < 0.01
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Random Number Generation Examples")
    print("=" * 60)

    examples = [
        ("1. Key Creation", example_key_creation),
        ("2. Key Splitting", example_key_splitting),
        ("3. Distributions", example_distributions),
        ("4. Random Choice", example_choice),
        ("5. Permutation", example_permutation),
        ("6. Reproducibility", example_reproducibility),
        ("7. Random in vmap", example_random_vmap),
        ("8. Random in JIT", example_random_jit),
        ("9. Dropout", example_dropout),
        ("10. Weight Init", example_weight_init),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
