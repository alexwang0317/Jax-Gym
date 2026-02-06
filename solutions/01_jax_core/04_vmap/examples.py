"""
JAX vmap (Vectorizing Map) - 10 Examples
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
# Example 1: Basic vmap for Batching
# =============================================================================
def example_basic_vmap():
    """
    vmap adds a batch dimension to a function.
    The function is applied independently to each element.
    """
    # Function for single vector
    def vector_norm(x):
        """Compute L2 norm of a single vector."""
        return jnp.sqrt(jnp.sum(x ** 2))

    # Single vector
    v = jnp.array([3.0, 4.0])  # norm = 5
    single_norm = vector_norm(v)

    # Batch of vectors - manual way
    batch = jnp.array([
        [3.0, 4.0],   # norm = 5
        [1.0, 0.0],   # norm = 1
        [0.0, 2.0],   # norm = 2
    ])

    # vmap way - much cleaner!
    batched_norm = vmap(vector_norm)
    norms = batched_norm(batch)

    # Equivalent to manual loop:
    manual_norms = jnp.array([vector_norm(v) for v in batch])

    return {
        'single_norm': single_norm,
        'batch_norms': norms,
        'manual_norms': manual_norms,
        'results_match': jnp.allclose(norms, manual_norms)
    }


# =============================================================================
# Example 2: in_axes and out_axes Specification
# =============================================================================
def example_in_out_axes():
    """
    in_axes specifies which axis is the batch axis for each input.
    out_axes specifies where the batch axis appears in output.

    in_axes=0 means first dimension is batch
    in_axes=None means don't batch this input (broadcast)
    """
    def add_bias(x, bias):
        """Add bias to vector x."""
        return x + bias

    # x is batched, bias is not (broadcast)
    batch_x = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    bias = jnp.array([10, 20])  # (2,)

    # in_axes=(0, None) means: batch over axis 0 of x, don't batch bias
    batched_add = vmap(add_bias, in_axes=(0, None))
    result = batched_add(batch_x, bias)

    # Different batching patterns
    def outer_product(a, b):
        return jnp.outer(a, b)

    a_batch = jnp.ones((3, 4))  # 3 vectors of length 4
    b_batch = jnp.ones((3, 5))  # 3 vectors of length 5

    # Batch both inputs along axis 0
    batched_outer = vmap(outer_product, in_axes=(0, 0))
    outer_results = batched_outer(a_batch, b_batch)  # (3, 4, 5)

    # out_axes example: put batch dimension last
    batched_outer_last = vmap(outer_product, in_axes=(0, 0), out_axes=-1)
    outer_last = batched_outer_last(a_batch, b_batch)  # (4, 5, 3)

    return {
        'broadcast_result': result,
        'outer_shape': outer_results.shape,
        'outer_last_shape': outer_last.shape
    }


# =============================================================================
# Example 3: vmap Over Multiple Arguments
# =============================================================================
def example_multiple_args():
    """
    vmap can batch different arguments along different axes.
    """
    def weighted_sum(x, y, weight):
        """Compute weight * x + (1-weight) * y."""
        return weight * x + (1 - weight) * y

    # Scenario 1: All inputs batched
    x_batch = jnp.array([1.0, 2.0, 3.0])
    y_batch = jnp.array([10.0, 20.0, 30.0])
    w_batch = jnp.array([0.5, 0.3, 0.7])

    all_batched = vmap(weighted_sum)
    result1 = all_batched(x_batch, y_batch, w_batch)

    # Scenario 2: x and y batched, weight fixed
    fixed_weight = 0.5
    xy_batched = vmap(weighted_sum, in_axes=(0, 0, None))
    result2 = xy_batched(x_batch, y_batch, fixed_weight)

    # Scenario 3: Different batch axes
    # x: (batch, features), y: (features,) broadcast
    x_2d = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    y_1d = jnp.array([10.0, 20.0])
    w_1d = jnp.array([0.5, 0.5, 0.5])

    mixed_batch = vmap(weighted_sum, in_axes=(0, None, 0))
    result3 = mixed_batch(x_2d, y_1d, w_1d)

    return {
        'all_batched': result1,
        'fixed_weight': result2,
        'mixed_batch': result3
    }


# =============================================================================
# Example 4: Nested vmap for Higher-Rank Batching
# =============================================================================
def example_nested_vmap():
    """
    Nested vmap applies batching multiple times.
    Useful for operating over multiple batch dimensions.
    """
    def dot_product(a, b):
        """Dot product of two vectors."""
        return jnp.sum(a * b)

    # Single vectors
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    single = dot_product(a, b)  # 32

    # Batch of vectors: (batch, features)
    a_batch = jnp.ones((4, 3))
    b_batch = jnp.ones((4, 3)) * 2
    batched = vmap(dot_product)(a_batch, b_batch)  # (4,)

    # 2D batch: (batch1, batch2, features)
    a_2d = jnp.ones((2, 4, 3))
    b_2d = jnp.ones((2, 4, 3)) * 3
    # Outer vmap batches over axis 0, inner vmap over what becomes axis 0 after
    double_batched = vmap(vmap(dot_product))(a_2d, b_2d)  # (2, 4)

    # Create pairwise dot products: all pairs of a[i] and b[j]
    # a: (M, D), b: (N, D) -> output: (M, N)
    a_set = jnp.array([[1, 0], [0, 1], [1, 1]], dtype=jnp.float32)  # (3, 2)
    b_set = jnp.array([[1, 0], [0, 1]], dtype=jnp.float32)  # (2, 2)

    # For each a[i], compute dot with all b[j]
    pairwise = vmap(lambda a: vmap(lambda b: dot_product(a, b))(b_set))(a_set)
    # Result: (3, 2) - pairwise[i,j] = dot(a_set[i], b_set[j])

    return {
        'single': single,
        'batched': batched,
        'double_batched': double_batched,
        'double_batched_shape': double_batched.shape,
        'pairwise': pairwise,
        'pairwise_shape': pairwise.shape
    }


# =============================================================================
# Example 5: vmap Combined with jit
# =============================================================================
def example_vmap_jit():
    """
    vmap and jit compose naturally.
    Order usually doesn't matter, but jit(vmap(f)) is typical.
    """
    def complex_fn(x):
        """Some computation we want fast and batched."""
        return jnp.sin(x) * jnp.exp(-x ** 2)

    # vmap then jit
    batched_jitted = jit(vmap(complex_fn))

    # jit then vmap (same result, different compilation)
    jitted_batched = vmap(jit(complex_fn))

    x_batch = jnp.linspace(0, 5, 1000)

    result1 = batched_jitted(x_batch)
    result2 = jitted_batched(x_batch)

    # Timing comparison (jit(vmap) is usually faster)
    import time

    # Warm up
    _ = batched_jitted(x_batch).block_until_ready()

    start = time.perf_counter()
    for _ in range(100):
        _ = batched_jitted(x_batch).block_until_ready()
    time1 = time.perf_counter() - start

    return {
        'results_match': jnp.allclose(result1, result2),
        'jit_vmap_time': time1,
        'output_shape': result1.shape
    }


# =============================================================================
# Example 6: vmap Combined with grad (Per-Example Gradients)
# =============================================================================
def example_vmap_grad():
    """
    vmap(grad(f)) computes per-example gradients.
    This is useful for:
    - Differentially private training
    - Understanding which examples contribute most to gradient
    - Second-order optimization
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

    grad_single = grad(loss_single)(params, x_single, y_single)

    # Per-example gradients for a batch
    x_batch = jnp.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y_batch = jnp.array([[3.0, 7.0], [2.0, 6.0], [4.0, 8.0]])

    # vmap over examples (axis 0 of x and y)
    per_example_grad = vmap(grad(loss_single), in_axes=(None, 0, 0))
    grads_batch = per_example_grad(params, x_batch, y_batch)

    # grads_batch['w'] has shape (3, 2, 2) - one gradient per example
    # grads_batch['b'] has shape (3, 2)

    # Mean gradient (what normal training computes)
    mean_grad_w = jnp.mean(grads_batch['w'], axis=0)
    mean_grad_b = jnp.mean(grads_batch['b'], axis=0)

    return {
        'single_grad_w_shape': grad_single['w'].shape,
        'batch_grad_w_shape': grads_batch['w'].shape,
        'batch_grad_b_shape': grads_batch['b'].shape,
        'mean_grad_w': mean_grad_w,
        'mean_grad_b': mean_grad_b
    }


# =============================================================================
# Example 7: vmap for Matrix-Vector Products
# =============================================================================
def example_matrix_vector():
    """
    Efficient batched matrix-vector products using vmap.
    """
    # Single matrix-vector product
    def matvec(A, x):
        return A @ x

    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    x = jnp.array([1.0, 1.0])

    single = matvec(A, x)  # [3, 7]

    # Batch of vectors, same matrix
    x_batch = jnp.array([[1, 1], [2, 0], [0, 2]], dtype=jnp.float32)

    # in_axes=(None, 0): same A, batch over x
    batch_matvec = vmap(matvec, in_axes=(None, 0))
    results = batch_matvec(A, x_batch)  # (3, 2)

    # Batch of matrices, same vector
    A_batch = jnp.stack([A, A * 2, A * 3])  # (3, 2, 2)

    # in_axes=(0, None): batch over A, same x
    batch_mat = vmap(matvec, in_axes=(0, None))
    results_A = batch_mat(A_batch, x)  # (3, 2)

    # Both batched
    both_batch = vmap(matvec, in_axes=(0, 0))
    results_both = both_batch(A_batch, x_batch)  # (3, 2)

    return {
        'single': single,
        'batch_x': results,
        'batch_A': results_A,
        'batch_both': results_both
    }


# =============================================================================
# Example 8: vmap for Pairwise Distances
# =============================================================================
def example_pairwise_distances():
    """
    Compute pairwise distances between two sets of points.
    This is a common operation in clustering, kNN, etc.
    """
    def euclidean_distance(a, b):
        """L2 distance between two vectors."""
        return jnp.sqrt(jnp.sum((a - b) ** 2))

    # Two sets of points
    points_a = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=jnp.float32)  # (3, 2)
    points_b = jnp.array([[1, 1], [2, 2]], dtype=jnp.float32)  # (2, 2)

    # Pairwise distances: (3, 2) matrix
    # dist[i, j] = distance(points_a[i], points_b[j])

    # Nested vmap approach
    pairwise_dist = vmap(
        vmap(euclidean_distance, in_axes=(None, 0)),
        in_axes=(0, None)
    )
    distances = pairwise_dist(points_a, points_b)

    # Alternative: compute all at once using broadcasting
    # diff[i, j, k] = points_a[i, k] - points_b[j, k]
    diff = points_a[:, None, :] - points_b[None, :, :]  # (3, 2, 2)
    distances_broadcast = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (3, 2)

    # Self-distances (symmetric matrix)
    self_dist = vmap(
        vmap(euclidean_distance, in_axes=(None, 0)),
        in_axes=(0, None)
    )
    self_distances = self_dist(points_a, points_a)  # (3, 3)

    return {
        'pairwise_distances': distances,
        'broadcast_distances': distances_broadcast,
        'methods_match': jnp.allclose(distances, distances_broadcast),
        'self_distances': self_distances,
        'self_diagonal_zero': jnp.allclose(jnp.diag(self_distances), 0)
    }


# =============================================================================
# Example 9: vmap with None Axes (Broadcasting)
# =============================================================================
def example_none_axes():
    """
    Using None in in_axes broadcasts that input across the batch.
    This is useful for shared parameters, constants, etc.
    """
    def apply_transform(x, scale, offset):
        """Transform: scale * x + offset."""
        return scale * x + offset

    # Batch of data, single scale and offset
    x_batch = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    scale = 2.0
    offset = jnp.array([10.0, 20.0])

    # x is batched, scale and offset are broadcast
    batched_transform = vmap(apply_transform, in_axes=(0, None, None))
    transformed = batched_transform(x_batch, scale, offset)

    # Different scales per batch element, same offset
    scales = jnp.array([1.0, 2.0, 3.0])
    mixed = vmap(apply_transform, in_axes=(0, 0, None))
    mixed_result = mixed(x_batch, scales[:, None], offset)  # need to broadcast scales

    # Actually, for element-wise operations, can be simpler:
    def apply_scalar_scale(x, scale, offset):
        return scale * x + offset

    mixed2 = vmap(apply_scalar_scale, in_axes=(0, 0, None))
    # But scale needs to match x's shape per-element

    return {
        'transformed': transformed,
        'mixed_result': mixed_result,
        'expected_first_row': jnp.array([12.0, 24.0])  # 2*[1,2] + [10,20]
    }


# =============================================================================
# Example 10: vmap for Attention Score Computation
# =============================================================================
def example_attention_scores():
    """
    Compute attention scores: softmax(Q @ K^T / sqrt(d_k)) @ V
    This is the core of transformer attention.
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
        scores = Q @ K.T / jnp.sqrt(d_k)  # (seq_len, seq_len)
        weights = jax.nn.softmax(scores, axis=-1)
        return weights @ V  # (seq_len, d_v)

    # Single head
    seq_len, d_k, d_v = 4, 8, 8
    Q = jnp.ones((seq_len, d_k))
    K = jnp.ones((seq_len, d_k))
    V = jnp.ones((seq_len, d_v))

    single_output = single_head_attention(Q, K, V)

    # Multi-head attention: batch over heads
    num_heads = 4
    Q_multi = jnp.ones((num_heads, seq_len, d_k))
    K_multi = jnp.ones((num_heads, seq_len, d_k))
    V_multi = jnp.ones((num_heads, seq_len, d_v))

    # vmap over head dimension
    multi_head = vmap(single_head_attention)
    multi_output = multi_head(Q_multi, K_multi, V_multi)  # (num_heads, seq_len, d_v)

    # Batch + multi-head: batch of sequences with multiple heads
    batch_size = 2
    Q_batch = jnp.ones((batch_size, num_heads, seq_len, d_k))
    K_batch = jnp.ones((batch_size, num_heads, seq_len, d_k))
    V_batch = jnp.ones((batch_size, num_heads, seq_len, d_v))

    # Nested vmap: outer for batch, inner for heads
    batch_multi_head = vmap(vmap(single_head_attention))
    batch_output = batch_multi_head(Q_batch, K_batch, V_batch)
    # Shape: (batch_size, num_heads, seq_len, d_v)

    return {
        'single_head_shape': single_output.shape,
        'multi_head_shape': multi_output.shape,
        'batch_multi_head_shape': batch_output.shape,
        'expected_batch_shape': (batch_size, num_heads, seq_len, d_v)
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX vmap Examples")
    print("=" * 60)

    examples = [
        ("1. Basic vmap", example_basic_vmap),
        ("2. in_axes and out_axes", example_in_out_axes),
        ("3. Multiple Arguments", example_multiple_args),
        ("4. Nested vmap", example_nested_vmap),
        ("5. vmap + jit", example_vmap_jit),
        ("6. vmap + grad", example_vmap_grad),
        ("7. Matrix-Vector Products", example_matrix_vector),
        ("8. Pairwise Distances", example_pairwise_distances),
        ("9. None Axes", example_none_axes),
        ("10. Attention Scores", example_attention_scores),
    ]

    for name, func in examples:
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
