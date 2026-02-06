"""
JAX vmap (Vectorizing Map) - 10 Exercises
=========================================

vmap automatically vectorizes functions to work on batches.
Write code for single examples, vmap makes it work on batches.

Key concepts:
- in_axes: which axis is the batch axis for each input
- out_axes: where to put the batch axis in output
- Composes with jit and grad
- Much cleaner than manual batch handling

Reference: https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, grad


# =============================================================================
# Exercise 1: Basic vmap for Batching
# =============================================================================
def exercise_basic_vmap():
    """
    vmap adds a batch dimension to a function.
    The function is applied independently to each element.

    Task:
    1. Compute the L2 norm of a single vector v
    2. Use vmap to compute norms for a batch of vectors
    3. Verify the results match a manual loop computation
    """
    # Function for single vector
    def vector_norm(x):
        """Compute L2 norm of a single vector."""
        return jnp.sqrt(jnp.sum(x ** 2))

    # Single vector
    v = jnp.array([3.0, 4.0])  # norm = 5

    # TODO: Compute norm of single vector v
    single_norm = vmap(vector_norm)

    # Batch of vectors - manual way
    batch = jnp.array([
        [3.0, 4.0],   # norm = 5
        [1.0, 0.0],   # norm = 1
        [0.0, 2.0],   # norm = 2
    ])

    # TODO: Use vmap to create a batched version of vector_norm
    # Then apply it to the batch
    batched_norm = None  # Create the vmapped function
    norms = None  # Apply it to batch

    # TODO: Compute norms manually using a list comprehension for comparison
    manual_norms = None

    return {
        'single_norm': single_norm,
        'batch_norms': norms,
        'manual_norms': manual_norms,
        'results_match': jnp.allclose(norms, manual_norms) if norms is not None and manual_norms is not None else False
    }


# =============================================================================
# Exercise 2: in_axes and out_axes Specification
# =============================================================================
def exercise_in_out_axes():
    """
    in_axes specifies which axis is the batch axis for each input.
    out_axes specifies where the batch axis appears in output.

    in_axes=0 means first dimension is batch
    in_axes=None means don't batch this input (broadcast)

    Task:
    1. Use vmap with in_axes to batch x while broadcasting bias
    2. Use vmap with in_axes=(0, 0) for outer product
    3. Use out_axes to put batch dimension last
    """
    def add_bias(x, bias):
        """Add bias to vector x."""
        return x + bias

    # x is batched, bias is not (broadcast)
    batch_x = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    bias = jnp.array([10, 20])  # (2,)

    # TODO: Create vmap with in_axes=(0, None) to batch over x, broadcast bias
    # Then apply to batch_x and bias
    batched_add = None
    result = None

    # Different batching patterns
    def outer_product(a, b):
        return jnp.outer(a, b)

    a_batch = jnp.ones((3, 4))  # 3 vectors of length 4
    b_batch = jnp.ones((3, 5))  # 3 vectors of length 5

    # TODO: Batch both inputs along axis 0
    batched_outer = None
    outer_results = None  # Should have shape (3, 4, 5)

    # TODO: Use out_axes to put batch dimension last
    batched_outer_last = None
    outer_last = None  # Should have shape (4, 5, 3)

    return {
        'broadcast_result': result,
        'outer_shape': outer_results.shape if outer_results is not None else None,
        'outer_last_shape': outer_last.shape if outer_last is not None else None
    }


# =============================================================================
# Exercise 3: vmap Over Multiple Arguments
# =============================================================================
def exercise_multiple_args():
    """
    vmap can batch different arguments along different axes.

    Task:
    1. Batch all three arguments
    2. Batch x and y, but keep weight fixed
    3. Use mixed batching: x and weight batched, y broadcast
    """
    def weighted_sum(x, y, weight):
        """Compute weight * x + (1-weight) * y."""
        return weight * x + (1 - weight) * y

    # Scenario 1: All inputs batched
    x_batch = jnp.array([1.0, 2.0, 3.0])
    y_batch = jnp.array([10.0, 20.0, 30.0])
    w_batch = jnp.array([0.5, 0.3, 0.7])

    # TODO: Use vmap to batch all arguments (default behavior)
    all_batched = None
    result1 = None

    # Scenario 2: x and y batched, weight fixed
    fixed_weight = 0.5

    # TODO: Use vmap with in_axes=(0, 0, None) to batch x, y but not weight
    xy_batched = None
    result2 = None

    # Scenario 3: Different batch axes
    # x: (batch, features), y: (features,) broadcast
    x_2d = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    y_1d = jnp.array([10.0, 20.0])
    w_1d = jnp.array([0.5, 0.5, 0.5])

    # TODO: Use vmap with in_axes=(0, None, 0) for mixed batching
    mixed_batch = None
    result3 = None

    return {
        'all_batched': result1,
        'fixed_weight': result2,
        'mixed_batch': result3
    }


# =============================================================================
# Exercise 4: Nested vmap for Higher-Rank Batching
# =============================================================================
def exercise_nested_vmap():
    """
    Nested vmap applies batching multiple times.
    Useful for operating over multiple batch dimensions.

    Task:
    1. Compute dot product of single vectors
    2. Use vmap for batched dot products
    3. Use nested vmap for 2D batch
    4. Compute pairwise dot products using nested vmap
    """
    def dot_product(a, b):
        """Dot product of two vectors."""
        return jnp.sum(a * b)

    # Single vectors
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])

    # TODO: Compute single dot product
    single = None  # Should be 32

    # Batch of vectors: (batch, features)
    a_batch = jnp.ones((4, 3))
    b_batch = jnp.ones((4, 3)) * 2

    # TODO: Use vmap for batched dot products
    batched = None  # Shape: (4,)

    # 2D batch: (batch1, batch2, features)
    a_2d = jnp.ones((2, 4, 3))
    b_2d = jnp.ones((2, 4, 3)) * 3

    # TODO: Use nested vmap (vmap(vmap(...))) for double batching
    double_batched = None  # Shape: (2, 4)

    # Create pairwise dot products: all pairs of a[i] and b[j]
    # a: (M, D), b: (N, D) -> output: (M, N)
    a_set = jnp.array([[1, 0], [0, 1], [1, 1]], dtype=jnp.float32)  # (3, 2)
    b_set = jnp.array([[1, 0], [0, 1]], dtype=jnp.float32)  # (2, 2)

    # TODO: Use nested vmap to compute pairwise dot products
    # For each a[i], compute dot with all b[j]
    # Result: (3, 2) - pairwise[i,j] = dot(a_set[i], b_set[j])
    pairwise = None

    return {
        'single': single,
        'batched': batched,
        'double_batched': double_batched,
        'double_batched_shape': double_batched.shape if double_batched is not None else None,
        'pairwise': pairwise,
        'pairwise_shape': pairwise.shape if pairwise is not None else None
    }


# =============================================================================
# Exercise 5: vmap Combined with jit
# =============================================================================
def exercise_vmap_jit():
    """
    vmap and jit compose naturally.
    Order usually doesn't matter, but jit(vmap(f)) is typical.

    Task:
    1. Create jit(vmap(f)) version
    2. Create vmap(jit(f)) version
    3. Verify they produce the same results
    """
    def complex_fn(x):
        """Some computation we want fast and batched."""
        return jnp.sin(x) * jnp.exp(-x ** 2)

    # TODO: Create vmap then jit version: jit(vmap(complex_fn))
    batched_jitted = None

    # TODO: Create jit then vmap version: vmap(jit(complex_fn))
    jitted_batched = None

    x_batch = jnp.linspace(0, 5, 1000)

    # TODO: Apply both versions to x_batch
    result1 = None
    result2 = None

    # Timing comparison (jit(vmap) is usually faster)
    import time

    # Warm up
    if batched_jitted is not None:
        _ = batched_jitted(x_batch).block_until_ready()

        start = time.perf_counter()
        for _ in range(100):
            _ = batched_jitted(x_batch).block_until_ready()
        time1 = time.perf_counter() - start
    else:
        time1 = None

    return {
        'results_match': jnp.allclose(result1, result2) if result1 is not None and result2 is not None else False,
        'jit_vmap_time': time1,
        'output_shape': result1.shape if result1 is not None else None
    }


# =============================================================================
# Exercise 6: vmap Combined with grad (Per-Example Gradients)
# =============================================================================
def exercise_vmap_grad():
    """
    vmap(grad(f)) computes per-example gradients.
    This is useful for:
    - Differentially private training
    - Understanding which examples contribute most to gradient
    - Second-order optimization

    Task:
    1. Compute gradient for a single example
    2. Use vmap(grad(...)) to compute per-example gradients for a batch
    3. Compute mean gradient from per-example gradients
    """
    def loss_single(params, x, y):
        """Loss for a single example."""
        pred = params['w'] @ x + params['b']
        return jnp.sum((pred - y) ** 2)

    params = {
        'w': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'b': jnp.array([0.0, 0.0])
    }

    # Single example gradient
    x_single = jnp.array([1.0, 1.0])
    y_single = jnp.array([3.0, 7.0])

    # TODO: Compute gradient for single example using grad(loss_single)
    grad_single = None

    # Per-example gradients for a batch
    x_batch = jnp.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y_batch = jnp.array([[3.0, 7.0], [2.0, 6.0], [4.0, 8.0]])

    # TODO: Use vmap(grad(...), in_axes=(None, 0, 0)) to compute per-example gradients
    # vmap over examples (axis 0 of x and y), but not over params
    per_example_grad = None
    grads_batch = None

    # TODO: Compute mean gradients from per-example gradients
    # grads_batch['w'] has shape (3, 2, 2) - one gradient per example
    # grads_batch['b'] has shape (3, 2)
    mean_grad_w = None
    mean_grad_b = None

    return {
        'single_grad_w_shape': grad_single['w'].shape if grad_single is not None else None,
        'batch_grad_w_shape': grads_batch['w'].shape if grads_batch is not None else None,
        'batch_grad_b_shape': grads_batch['b'].shape if grads_batch is not None else None,
        'mean_grad_w': mean_grad_w,
        'mean_grad_b': mean_grad_b
    }


# =============================================================================
# Exercise 7: vmap for Matrix-Vector Products
# =============================================================================
def exercise_matrix_vector():
    """
    Efficient batched matrix-vector products using vmap.

    Task:
    1. Compute single matrix-vector product
    2. Batch of vectors, same matrix (in_axes=(None, 0))
    3. Batch of matrices, same vector (in_axes=(0, None))
    4. Both batched (in_axes=(0, 0))
    """
    # Single matrix-vector product
    def matvec(A, x):
        return A @ x

    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    x = jnp.array([1.0, 1.0])

    # TODO: Compute single matrix-vector product
    single = None  # Should be [3, 7]

    # Batch of vectors, same matrix
    x_batch = jnp.array([[1, 1], [2, 0], [0, 2]], dtype=jnp.float32)

    # TODO: Use vmap with in_axes=(None, 0) for same A, batch over x
    batch_matvec = None
    results = None  # Shape: (3, 2)

    # Batch of matrices, same vector
    A_batch = jnp.stack([A, A * 2, A * 3])  # (3, 2, 2)

    # TODO: Use vmap with in_axes=(0, None) for batch over A, same x
    batch_mat = None
    results_A = None  # Shape: (3, 2)

    # TODO: Both batched with in_axes=(0, 0)
    both_batch = None
    results_both = None  # Shape: (3, 2)

    return {
        'single': single,
        'batch_x': results,
        'batch_A': results_A,
        'batch_both': results_both
    }


# =============================================================================
# Exercise 8: vmap for Pairwise Distances
# =============================================================================
def exercise_pairwise_distances():
    """
    Compute pairwise distances between two sets of points.
    This is a common operation in clustering, kNN, etc.

    Task:
    1. Use nested vmap to compute pairwise distances
    2. Verify against broadcasting implementation
    3. Compute self-distances (symmetric matrix)
    """
    def euclidean_distance(a, b):
        """L2 distance between two vectors."""
        return jnp.sqrt(jnp.sum((a - b) ** 2))

    # Two sets of points
    points_a = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=jnp.float32)  # (3, 2)
    points_b = jnp.array([[1, 1], [2, 2]], dtype=jnp.float32)  # (2, 2)

    # Pairwise distances: (3, 2) matrix
    # dist[i, j] = distance(points_a[i], points_b[j])

    # TODO: Use nested vmap to compute pairwise distances
    # Outer vmap over points_a (in_axes=(0, None))
    # Inner vmap over points_b (in_axes=(None, 0))
    pairwise_dist = None
    distances = None

    # Alternative: compute all at once using broadcasting
    # diff[i, j, k] = points_a[i, k] - points_b[j, k]
    diff = points_a[:, None, :] - points_b[None, :, :]  # (3, 2, 2)
    distances_broadcast = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (3, 2)

    # TODO: Compute self-distances (symmetric matrix)
    # Use the same nested vmap pattern but with points_a for both inputs
    self_dist = None
    self_distances = None  # Shape: (3, 3)

    return {
        'pairwise_distances': distances,
        'broadcast_distances': distances_broadcast,
        'methods_match': jnp.allclose(distances, distances_broadcast) if distances is not None else False,
        'self_distances': self_distances,
        'self_diagonal_zero': jnp.allclose(jnp.diag(self_distances), 0) if self_distances is not None else False
    }


# =============================================================================
# Exercise 9: vmap with None Axes (Broadcasting)
# =============================================================================
def exercise_none_axes():
    """
    Using None in in_axes broadcasts that input across the batch.
    This is useful for shared parameters, constants, etc.

    Task:
    1. Transform a batch of data with shared scale and offset
    2. Apply different scales per batch element with same offset
    """
    def apply_transform(x, scale, offset):
        """Transform: scale * x + offset."""
        return scale * x + offset

    # Batch of data, single scale and offset
    x_batch = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    scale = 2.0
    offset = jnp.array([10.0, 20.0])

    # TODO: Use vmap with in_axes=(0, None, None) to batch x, broadcast scale and offset
    batched_transform = None
    transformed = None

    # Different scales per batch element, same offset
    scales = jnp.array([1.0, 2.0, 3.0])

    # TODO: Use vmap with in_axes=(0, 0, None) for batched x and scales, broadcast offset
    # Note: scales needs to be reshaped to (3, 1) for broadcasting with x
    mixed = None
    mixed_result = None

    return {
        'transformed': transformed,
        'mixed_result': mixed_result,
        'expected_first_row': jnp.array([12.0, 24.0])  # 2*[1,2] + [10,20]
    }


# =============================================================================
# Exercise 10: vmap for Attention Score Computation
# =============================================================================
def exercise_attention_scores():
    """
    Compute attention scores: softmax(Q @ K^T / sqrt(d_k)) @ V
    This is the core of transformer attention.

    Task:
    1. Implement single head attention
    2. Use vmap to create multi-head attention
    3. Use nested vmap for batch + multi-head
    """
    def single_head_attention(Q, K, V):
        """
        Single attention head.
        Q: (seq_len, d_k)
        K: (seq_len, d_k)
        V: (seq_len, d_v)
        Returns: (seq_len, d_v)
        """
        d_k = Q.shape[-1]
        # TODO: Implement attention computation
        # 1. Compute scores = Q @ K.T / sqrt(d_k)
        # 2. Apply softmax along axis=-1
        # 3. Return weights @ V
        scores = None
        weights = None
        return None

    # Single head
    seq_len, d_k, d_v = 4, 8, 8
    Q = jnp.ones((seq_len, d_k))
    K = jnp.ones((seq_len, d_k))
    V = jnp.ones((seq_len, d_v))

    # TODO: Apply single head attention
    single_output = None

    # Multi-head attention: batch over heads
    num_heads = 4
    Q_multi = jnp.ones((num_heads, seq_len, d_k))
    K_multi = jnp.ones((num_heads, seq_len, d_k))
    V_multi = jnp.ones((num_heads, seq_len, d_v))

    # TODO: Use vmap to create multi-head attention
    multi_head = None
    multi_output = None  # Shape: (num_heads, seq_len, d_v)

    # Batch + multi-head: batch of sequences with multiple heads
    batch_size = 2
    Q_batch = jnp.ones((batch_size, num_heads, seq_len, d_k))
    K_batch = jnp.ones((batch_size, num_heads, seq_len, d_k))
    V_batch = jnp.ones((batch_size, num_heads, seq_len, d_v))

    # TODO: Use nested vmap for batch + multi-head
    batch_multi_head = None
    batch_output = None  # Shape: (batch_size, num_heads, seq_len, d_v)

    return {
        'single_head_shape': single_output.shape if single_output is not None else None,
        'multi_head_shape': multi_output.shape if multi_output is not None else None,
        'batch_multi_head_shape': batch_output.shape if batch_output is not None else None,
        'expected_batch_shape': (batch_size, num_heads, seq_len, d_v)
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX vmap Exercises")
    print("=" * 60)

    exercises = [
        ("1. Basic vmap", exercise_basic_vmap),
        ("2. in_axes and out_axes", exercise_in_out_axes),
        ("3. Multiple Arguments", exercise_multiple_args),
        ("4. Nested vmap", exercise_nested_vmap),
        ("5. vmap + jit", exercise_vmap_jit),
        ("6. vmap + grad", exercise_vmap_grad),
        ("7. Matrix-Vector Products", exercise_matrix_vector),
        ("8. Pairwise Distances", exercise_pairwise_distances),
        ("9. None Axes", exercise_none_axes),
        ("10. Attention Scores", exercise_attention_scores),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
