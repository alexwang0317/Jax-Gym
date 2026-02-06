"""
JAX Random Number Generation - 10 Exercises
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
# Exercise 1: Key Creation with jax.random.key
# =============================================================================
def exercise_key_creation():
    """
    Create PRNG keys from integer seeds.
    Keys are the basis of all randomness in JAX.

    TODO:
    - Create a key from seed 0
    - Create key1 from seed 42 and key2 from seed 123
    - Create key_a and key_b both from seed 42
    - Generate uniform values from key_a and key_b to show same seed = same value
    - Create key_large from seed 2**31 - 1

    Returns dict with:
    - 'key_type': type name of the key
    - 'key_dtype': dtype of the key
    - 'same_seed_same_value': bool, True if same seed produces same random value
    - 'different_seeds_different': bool, True if different seeds produce different values
    """
    # TODO: Implement this function
    key = None
    key1 = None
    key2 = None
    key_a = None
    key_b = None
    val_a = None
    val_b = None
    key_large = None

    return {
        'key_type': None,
        'key_dtype': None,
        'same_seed_same_value': None,
        'different_seeds_different': None
    }


# =============================================================================
# Exercise 2: Key Splitting
# =============================================================================
def exercise_key_splitting():
    """
    Split keys to get independent random streams.
    NEVER reuse a key - always split!

    Pattern: key, subkey = random.split(key)

    TODO:
    - Create key from seed 0
    - Show that reusing the same key gives same numbers (wrong_val1, wrong_val2)
    - Split key properly to get different values (val1, val2)
    - Split a key into 5 keys at once
    - Generate values from all 5 keys and verify they're all different

    Returns dict with:
    - 'reused_key_same': bool, True if reusing key gives same value
    - 'split_keys_different': bool, True if split keys give different values
    - 'multi_split_count': int, number of keys from multi-split (should be 5)
    - 'all_different': bool, True if all 5 values are unique
    """
    # TODO: Implement this function
    key = None
    wrong_val1 = None
    wrong_val2 = None
    val1 = None
    val2 = None
    keys = None
    vals = None

    return {
        'reused_key_same': None,
        'split_keys_different': None,
        'multi_split_count': None,
        'all_different': None
    }


# =============================================================================
# Exercise 3: Uniform and Normal Distributions
# =============================================================================
def exercise_distributions():
    """
    Generate random numbers from common distributions.

    TODO:
    - Create key from seed 42
    - Generate 5 uniform values in [0, 1)
    - Generate 5 uniform values in [-1, 1)
    - Generate 5 standard normal values
    - Generate 5 normal values with mean=5.0, std=2.0
    - Generate 5 truncated normal values in [-2, 2]
    - Generate 5 exponential values
    - Generate 5 random integers in [0, 10)

    Remember to split the key before each use!

    Returns dict with:
    - 'uniform_01': array of 5 uniform values in [0,1)
    - 'uniform_range': array of 5 uniform values in [-1,1)
    - 'normal_std': array of 5 standard normal values
    - 'normal_custom_mean': mean of the custom normal values
    - 'truncated_in_range': bool, True if all truncated values in [-2,2]
    - 'integers': array of 5 random integers
    """
    # TODO: Implement this function
    key = None
    uniform_01 = None
    uniform_range = None
    normal_std = None
    normal_custom = None
    truncated = None
    exponential = None
    integers = None

    return {
        'uniform_01': None,
        'uniform_range': None,
        'normal_std': None,
        'normal_custom_mean': None,
        'truncated_in_range': None,
        'integers': None
    }


# =============================================================================
# Exercise 4: Random Choice for Sampling
# =============================================================================
def exercise_choice():
    """
    Sample from arrays with or without replacement.
    Useful for minibatching, bootstrapping, etc.

    TODO:
    - Create key from seed 42
    - Sample 10 indices from range 0-99
    - Sample 3 elements from data = [10, 20, 30, 40, 50]
    - Sample 3 unique elements (without replacement) from data
    - Do weighted sampling: 100 samples with weights [0.5, 0.3, 0.1, 0.05, 0.05]
    - Count occurrences of each value in weighted samples
    - Sample batch indices: 32 unique indices from dataset of size 1000

    Returns dict with:
    - 'random_indices': first 5 of the 10 sampled indices
    - 'samples_from_array': 3 sampled elements
    - 'unique_samples': 3 unique sampled elements
    - 'weighted_counts': count of each value in weighted samples
    - 'batch_indices_shape': shape of batch indices (should be (32,))
    """
    # TODO: Implement this function
    key = None
    indices = None
    data = jnp.array([10, 20, 30, 40, 50])
    samples = None
    unique_samples = None
    weights = jnp.array([0.5, 0.3, 0.1, 0.05, 0.05])
    weighted = None
    counts = None
    batch_indices = None

    return {
        'random_indices': None,
        'samples_from_array': None,
        'unique_samples': None,
        'weighted_counts': None,
        'batch_indices_shape': None
    }


# =============================================================================
# Exercise 5: Permutation for Shuffling
# =============================================================================
def exercise_permutation():
    """
    Shuffle arrays or generate random permutations.
    Essential for data shuffling in training.

    TODO:
    - Create key from seed 42
    - Generate a random permutation of indices 0-9
    - Shuffle data = [10, 20, 30, 40, 50]
    - Create dataset X (20 samples, 5 features) and y (20 labels)
    - Shuffle X and y together using same permutation indices
    - Create shuffle_data function and use it for two epochs
    - Verify that different epochs have different orderings

    Returns dict with:
    - 'permutation': permutation of 0-9
    - 'shuffled_data': shuffled version of [10,20,30,40,50]
    - 'X_shuffled_shape': shape of shuffled X
    - 'different_epochs': bool, True if two epochs have different orderings
    """
    # TODO: Implement this function
    key = None
    perm = None
    data = jnp.array([10, 20, 30, 40, 50])
    shuffled = None
    X = jnp.arange(100).reshape(20, 5)
    y = jnp.arange(20)
    X_shuffled = None
    y_shuffled = None

    def shuffle_data(key, X, y):
        # TODO: Implement this helper function
        pass

    y_epoch1 = None
    y_epoch2 = None

    return {
        'permutation': None,
        'shuffled_data': None,
        'X_shuffled_shape': None,
        'different_epochs': None
    }


# =============================================================================
# Exercise 6: Reproducibility Patterns
# =============================================================================
def exercise_reproducibility():
    """
    Patterns for reproducible random number generation.

    TODO:
    - Create experiment function that takes a seed and returns random arrays
    - Run experiment with seed 42 twice to verify same results
    - Run experiment with seed 123 to verify different results
    - Implement checkpoint/resume pattern: save key, generate value,
      resume from saved key and verify same value is generated

    Returns dict with:
    - 'same_seed_reproduces': bool, True if same seed gives same results
    - 'different_seed_differs': bool, True if different seed gives different results
    - 'checkpoint_works': bool, True if checkpoint/resume produces same value
    """
    # TODO: Implement this function

    def experiment(seed):
        # TODO: Implement experiment function
        # Should return tuple of (normal_array, uniform_array)
        pass

    result1 = None
    result2 = None
    result3 = None

    same_seed_same = None
    different_seed_different = None

    # Checkpoint/resume pattern
    val2 = None
    val2_reproduced = None

    return {
        'same_seed_reproduces': None,
        'different_seed_differs': None,
        'checkpoint_works': None
    }


# =============================================================================
# Exercise 7: Random in vmap (Unique Keys per Batch)
# =============================================================================
def exercise_random_vmap():
    """
    Using random in vmap requires one key per batch element.

    TODO:
    - Create sample_fn that takes a key and returns 3 normal values
    - Split key into 5 keys (one per batch element)
    - Use vmap to apply sample_fn to all keys
    - Verify all samples are different
    - Create batched_dropout function that applies dropout to a batch

    Returns dict with:
    - 'batch_samples_shape': shape of batched samples (should be (5, 3))
    - 'samples_different': bool, True if batch samples are different
    - 'dropout_output_shape': shape of dropout output
    """
    # TODO: Implement this function

    def sample_fn(key):
        """Function that uses randomness."""
        # TODO: Implement - return 3 normal values
        pass

    key = None
    batch_size = 5
    keys = None
    correct_samples = None
    all_different = None

    def batched_dropout(key, x, rate=0.5):
        """Apply dropout to a batch."""
        # TODO: Implement batched dropout using vmap
        pass

    dropped = None

    return {
        'batch_samples_shape': None,
        'samples_different': None,
        'dropout_output_shape': None
    }


# =============================================================================
# Exercise 8: Random in JIT (Passing Keys Explicitly)
# =============================================================================
def exercise_random_jit():
    """
    JIT-compiled functions must receive keys as arguments.
    Keys cannot be created inside JIT (would break reproducibility).

    TODO:
    - Create a JIT-compiled function that adds noise to input x
    - The function should take key and x as arguments
    - Call the function twice with different subkeys
    - Verify results are different

    Returns dict with:
    - 'results_different': bool, True if different keys give different results
    - 'jit_works': bool, True (just to verify JIT compilation worked)
    """
    # TODO: Implement this function

    @jax.jit
    def good_fn(key, x):
        # TODO: Implement - add noise to x
        pass

    key = None
    x = jnp.ones(5)
    result1 = None
    result2 = None

    return {
        'results_different': None,
        'jit_works': None
    }


# =============================================================================
# Exercise 9: Dropout Implementation
# =============================================================================
def exercise_dropout():
    """
    Implement dropout using JAX random.
    Dropout randomly zeros elements during training.

    TODO:
    - Implement dropout function that:
      - Returns x unchanged if not training or rate == 0
      - Creates a bernoulli mask with keep_rate = 1 - rate
      - Zeros masked elements and scales remaining by 1/(1-rate)
    - Apply dropout with rate=0.3 to x = ones((10, 20))
    - Verify inference mode returns unchanged input
    - Verify scaling is correct (non-zero values should be 1/(1-rate))

    Returns dict with:
    - 'dropped_shape': shape of dropped output
    - 'approx_drop_rate': approximate fraction of zeros
    - 'scale_correct': bool, True if scaling is correct
    - 'inference_unchanged': bool, True if inference mode unchanged
    """
    # TODO: Implement this function

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
        # TODO: Implement dropout
        pass

    x = jnp.ones((10, 20))
    key = random.key(42)

    dropped = None
    no_drop = None
    non_zero_vals = None
    expected_scale = 1 / (1 - 0.3)
    dropped_count = None
    total_elements = x.size

    return {
        'dropped_shape': None,
        'approx_drop_rate': None,
        'scale_correct': None,
        'inference_unchanged': None
    }


# =============================================================================
# Exercise 10: Weight Initialization Patterns
# =============================================================================
def exercise_weight_init():
    """
    Common weight initialization patterns for neural networks.

    TODO:
    - Implement xavier_normal initialization:
      std = sqrt(2 / (fan_in + fan_out))
    - Initialize MLP with layers: (784, 256), (256, 128), (128, 10)
    - Each layer needs weights (xavier_normal) and biases (zeros)
    - Check that layer0 weights have correct statistics

    Returns dict with:
    - 'layer0_w_shape': shape of first layer weights
    - 'layer0_w_mean': mean of first layer weights (should be ~0)
    - 'layer0_w_std': std of first layer weights
    - 'expected_std': expected std for Xavier normal
    - 'std_close': bool, True if actual std is close to expected
    """
    # TODO: Implement this function

    def xavier_normal(key, shape):
        """Xavier/Glorot normal initialization."""
        # TODO: Implement Xavier normal initialization
        pass

    key = None
    layer_shapes = [(784, 256), (256, 128), (128, 10)]
    mlp_params = {}

    # TODO: Initialize MLP layers

    w0 = None
    w0_mean = None
    w0_std = None
    expected_std = None

    return {
        'layer0_w_shape': None,
        'layer0_w_mean': None,
        'layer0_w_std': None,
        'expected_std': None,
        'std_close': None
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Random Number Generation Exercises")
    print("=" * 60)

    exercises = [
        ("1. Key Creation", exercise_key_creation),
        ("2. Key Splitting", exercise_key_splitting),
        ("3. Distributions", exercise_distributions),
        ("4. Random Choice", exercise_choice),
        ("5. Permutation", exercise_permutation),
        ("6. Reproducibility", exercise_reproducibility),
        ("7. Random in vmap", exercise_random_vmap),
        ("8. Random in JIT", exercise_random_jit),
        ("9. Dropout", exercise_dropout),
        ("10. Weight Init", exercise_weight_init),
    ]

    for name, func in exercises:
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
