"""
JAX NumPy Basics - 10 Examples
==============================

jax.numpy (jnp) provides a NumPy-like API that works with JAX transformations.
Key difference: JAX arrays are IMMUTABLE - no in-place operations.

Reference: https://jax.readthedocs.io/en/latest/jax.numpy.html
"""

import jax
import jax.numpy as jnp


# =============================================================================
# Example 1: Array Creation
# =============================================================================
def example_array_creation():
    """
    Create arrays using various jnp functions.
    Similar to NumPy but returns JAX DeviceArrays.
    """
    # From Python list
    arr1 = jnp.array([1, 2, 3, 4, 5])

    # Zeros and ones
    zeros = jnp.zeros((3, 4))  # 3x4 matrix of zeros
    ones = jnp.ones((2, 3), dtype=jnp.float32)

    # Ranges
    range_arr = jnp.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
    linspace_arr = jnp.linspace(0, 1, 5)  # 5 points from 0 to 1

    # Identity matrix
    eye = jnp.eye(3)

    # Full array with specific value
    full = jnp.full((2, 2), fill_value=7.0)

    return {
        'from_list': arr1,
        'zeros': zeros,
        'ones': ones,
        'arange': range_arr,
        'linspace': linspace_arr,
        'eye': eye,
        'full': full
    }


# =============================================================================
# Example 2: Array Reshaping and Slicing
# =============================================================================
def example_reshaping_slicing():
    """
    Reshape arrays and extract slices.
    Slicing works like NumPy but returns views that are still immutable.
    """
    arr = jnp.arange(12)

    # Reshape to 3x4
    reshaped = arr.reshape(3, 4)

    # Alternative: use -1 for auto-dimension
    reshaped_auto = arr.reshape(3, -1)  # same as (3, 4)

    # Slicing
    row = reshaped[0]          # First row
    col = reshaped[:, 0]       # First column
    submat = reshaped[1:, 1:3] # Submatrix

    # Flatten back
    flat = reshaped.flatten()

    # Transpose
    transposed = reshaped.T

    # Expand and squeeze dimensions
    expanded = jnp.expand_dims(arr, axis=0)  # (12,) -> (1, 12)
    squeezed = jnp.squeeze(expanded)         # (1, 12) -> (12,)

    return {
        'original': arr,
        'reshaped': reshaped,
        'row_0': row,
        'col_0': col,
        'submatrix': submat,
        'transposed': transposed,
        'expanded': expanded,
        'squeezed': squeezed
    }


# =============================================================================
# Example 3: Mathematical Operations
# =============================================================================
def example_math_operations():
    """
    Element-wise mathematical functions.
    These are automatically differentiable with jax.grad.
    """
    x = jnp.linspace(0, 2 * jnp.pi, 100)

    # Trigonometric
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    tan_x = jnp.tan(x)

    # Exponential and logarithm
    y = jnp.linspace(0.1, 5, 50)
    exp_y = jnp.exp(y)
    log_y = jnp.log(y)
    log10_y = jnp.log10(y)

    # Power and roots
    z = jnp.array([1, 4, 9, 16, 25])
    sqrt_z = jnp.sqrt(z)
    power_z = jnp.power(z, 2)  # z^2

    # Absolute value and sign
    w = jnp.array([-3, -1, 0, 1, 3])
    abs_w = jnp.abs(w)
    sign_w = jnp.sign(w)

    return {
        'sin': sin_x,
        'cos': cos_x,
        'exp': exp_y,
        'log': log_y,
        'sqrt': sqrt_z,
        'abs': abs_w,
        'sign': sign_w
    }


# =============================================================================
# Example 4: Linear Algebra
# =============================================================================
def example_linear_algebra():
    """
    Linear algebra operations: dot products, matrix multiplication, etc.
    """
    # Vectors
    v1 = jnp.array([1.0, 2.0, 3.0])
    v2 = jnp.array([4.0, 5.0, 6.0])

    # Dot product (inner product)
    dot_product = jnp.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

    # Matrices
    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    B = jnp.array([[5, 6], [7, 8]], dtype=jnp.float32)

    # Matrix multiplication
    matmul_result = jnp.matmul(A, B)  # or A @ B
    matmul_operator = A @ B

    # Outer product
    outer = jnp.outer(v1, v2)

    # Matrix-vector product
    x = jnp.array([1.0, 2.0])
    Ax = A @ x

    # Norms
    vec_norm = jnp.linalg.norm(v1)
    mat_norm = jnp.linalg.norm(A, ord='fro')  # Frobenius norm

    # Determinant and trace
    det_A = jnp.linalg.det(A)
    trace_A = jnp.trace(A)

    # Matrix inverse
    A_inv = jnp.linalg.inv(A)

    return {
        'dot_product': dot_product,
        'matmul': matmul_result,
        'outer_product': outer,
        'matrix_vector': Ax,
        'vec_norm': vec_norm,
        'determinant': det_A,
        'trace': trace_A,
        'inverse': A_inv
    }


# =============================================================================
# Example 5: Broadcasting Rules
# =============================================================================
def example_broadcasting():
    """
    Broadcasting allows operations on arrays with different shapes.
    Rules: dimensions are compatible if they're equal or one of them is 1.
    """
    # Scalar broadcast
    arr = jnp.array([1, 2, 3])
    scalar_add = arr + 10  # [11, 12, 13]
    scalar_mul = arr * 2   # [2, 4, 6]

    # Row and column broadcast
    matrix = jnp.ones((3, 4))
    row_vec = jnp.array([1, 2, 3, 4])      # shape (4,)
    col_vec = jnp.array([[10], [20], [30]]) # shape (3, 1)

    # Broadcasting adds dimensions from the right
    with_row = matrix + row_vec  # (3,4) + (4,) -> (3,4)
    with_col = matrix + col_vec  # (3,4) + (3,1) -> (3,4)

    # Outer product via broadcasting
    a = jnp.array([1, 2, 3])[:, None]  # (3, 1)
    b = jnp.array([4, 5, 6])[None, :]  # (1, 3)
    outer_broadcast = a * b  # (3, 3)

    # 3D broadcasting
    batch = jnp.ones((2, 3, 4))  # batch of matrices
    bias = jnp.array([1, 2, 3, 4])  # shape (4,)
    batch_with_bias = batch + bias  # broadcasts to all batches

    return {
        'scalar_add': scalar_add,
        'with_row_vec': with_row,
        'with_col_vec': with_col,
        'outer_via_broadcast': outer_broadcast,
        'batch_broadcast_shape': batch_with_bias.shape
    }


# =============================================================================
# Example 6: Reduction Operations
# =============================================================================
def example_reductions():
    """
    Reduction operations collapse dimensions.
    Use 'axis' to specify which dimension(s) to reduce.
    """
    arr = jnp.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=jnp.float32)

    # Global reductions (collapse all dimensions)
    total_sum = jnp.sum(arr)      # 45
    total_mean = jnp.mean(arr)    # 5.0
    total_max = jnp.max(arr)      # 9
    total_min = jnp.min(arr)      # 1
    total_prod = jnp.prod(arr)    # 362880

    # Axis-specific reductions
    row_sum = jnp.sum(arr, axis=1)    # Sum each row: [6, 15, 24]
    col_sum = jnp.sum(arr, axis=0)    # Sum each column: [12, 15, 18]

    # Keep dimensions for broadcasting
    row_mean_keepdims = jnp.mean(arr, axis=1, keepdims=True)  # shape (3, 1)

    # Multiple axes
    sum_both = jnp.sum(arr, axis=(0, 1))  # Same as global sum

    # Argmax/Argmin (returns indices)
    argmax_flat = jnp.argmax(arr)         # 8 (index of 9 in flattened)
    argmax_row = jnp.argmax(arr, axis=1)  # [2, 2, 2] (index of max in each row)

    # Cumulative operations
    cumsum = jnp.cumsum(arr.flatten())  # Running sum

    return {
        'sum': total_sum,
        'mean': total_mean,
        'max': total_max,
        'row_sum': row_sum,
        'col_sum': col_sum,
        'argmax': argmax_flat,
        'cumsum': cumsum
    }


# =============================================================================
# Example 7: Boolean Indexing and jnp.where
# =============================================================================
def example_boolean_indexing():
    """
    Boolean operations and conditional selection.
    jnp.where is the main tool for conditional operations in JAX.
    """
    arr = jnp.array([1, -2, 3, -4, 5, -6])

    # Boolean mask
    positive_mask = arr > 0  # [True, False, True, False, True, False]

    # Count true values
    num_positive = jnp.sum(positive_mask)

    # jnp.where with condition only (returns indices)
    positive_indices = jnp.where(positive_mask, size=3)  # need size for JIT

    # jnp.where with x and y (like ternary operator)
    # where(condition, x, y) -> x if condition else y
    abs_arr = jnp.where(arr > 0, arr, -arr)  # absolute value

    # Clipping values
    clipped = jnp.where(arr > 3, 3, arr)  # cap at 3
    clipped = jnp.where(clipped < -3, -3, clipped)  # floor at -3
    # Or use jnp.clip
    clipped_builtin = jnp.clip(arr, -3, 3)

    # Comparison operations
    matrix = jnp.array([[1, 2], [3, 4]])
    greater_than_2 = matrix > 2
    all_positive = jnp.all(matrix > 0)
    any_greater_3 = jnp.any(matrix > 3)

    return {
        'positive_mask': positive_mask,
        'num_positive': num_positive,
        'abs_via_where': abs_arr,
        'clipped': clipped_builtin,
        'all_positive': all_positive,
        'any_greater_3': any_greater_3
    }


# =============================================================================
# Example 8: Immutability and .at[].set() Pattern
# =============================================================================
def example_immutability():
    """
    JAX arrays are IMMUTABLE. You cannot modify them in place.
    Use .at[].set() to create a new array with modifications.

    This is crucial for JAX's functional paradigm and enables
    automatic differentiation and JIT compilation.
    """
    arr = jnp.array([1, 2, 3, 4, 5])

    # WRONG (would raise error in JAX):
    # arr[0] = 10  # TypeError!

    # CORRECT: .at[].set() returns a NEW array
    arr_modified = arr.at[0].set(10)  # [10, 2, 3, 4, 5]
    # Original arr is unchanged: [1, 2, 3, 4, 5]

    # Multiple updates (chained)
    arr_multi = arr.at[0].set(10).at[4].set(50)

    # Slice updates
    arr_slice = arr.at[1:4].set(0)  # [1, 0, 0, 0, 5]

    # Add/multiply at index (useful for scatter operations)
    arr_add = arr.at[0].add(100)  # [101, 2, 3, 4, 5]
    arr_mul = arr.at[0].mul(10)   # [10, 2, 3, 4, 5]

    # 2D array updates
    matrix = jnp.zeros((3, 3))
    matrix_updated = matrix.at[1, 1].set(1.0)  # Set center to 1

    # Set entire row/column
    matrix_row = matrix.at[0, :].set(jnp.array([1, 2, 3]))
    matrix_col = matrix.at[:, 0].set(jnp.array([1, 2, 3]))

    # Scatter operation: add values at indices
    indices = jnp.array([0, 2, 2])  # Note: index 2 appears twice
    values = jnp.array([1.0, 2.0, 3.0])
    scattered = jnp.zeros(5).at[indices].add(values)  # [1, 0, 5, 0, 0]

    return {
        'original': arr,
        'modified': arr_modified,
        'multi_update': arr_multi,
        'slice_update': arr_slice,
        'add_at_index': arr_add,
        'matrix_center': matrix_updated,
        'scattered': scattered
    }


# =============================================================================
# Example 9: Einstein Summation (einsum)
# =============================================================================
def example_einsum():
    """
    jnp.einsum provides a powerful, flexible way to express tensor operations.
    The notation describes which indices are summed over.

    Format: 'input_indices -> output_indices'
    - Repeated indices are summed over
    - Output indices determine the result shape
    """
    # Vectors
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])

    # Dot product: sum over i
    # 'i,i->' means: multiply element-wise, sum all
    dot = jnp.einsum('i,i->', a, b)  # 32

    # Outer product: no summation
    # 'i,j->ij' means: create matrix with a[i]*b[j]
    outer = jnp.einsum('i,j->ij', a, b)

    # Matrices
    A = jnp.array([[1, 2], [3, 4]])
    B = jnp.array([[5, 6], [7, 8]])

    # Matrix multiplication
    # 'ik,kj->ij' means: sum over k (the shared dimension)
    matmul = jnp.einsum('ik,kj->ij', A, B)

    # Transpose
    transpose = jnp.einsum('ij->ji', A)

    # Trace (sum of diagonal)
    trace = jnp.einsum('ii->', A)  # 1 + 4 = 5

    # Batch matrix multiplication
    # Batch of 2 matrices, each 3x4 and 4x5
    batch_A = jnp.ones((2, 3, 4))
    batch_B = jnp.ones((2, 4, 5))
    batch_matmul = jnp.einsum('bij,bjk->bik', batch_A, batch_B)

    # Attention-style: batch, heads, seq, dim
    Q = jnp.ones((2, 4, 8, 16))  # batch=2, heads=4, seq=8, dim=16
    K = jnp.ones((2, 4, 8, 16))
    # Attention scores: Q @ K^T for each batch and head
    scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K)  # (2, 4, 8, 8)

    return {
        'dot_product': dot,
        'outer_product': outer,
        'matmul': matmul,
        'transpose': transpose,
        'trace': trace,
        'batch_matmul_shape': batch_matmul.shape,
        'attention_scores_shape': scores.shape
    }


# =============================================================================
# Example 10: Device Placement
# =============================================================================
def example_device_placement():
    """
    JAX can run on CPU, GPU, or TPU.
    Use jax.devices() to see available devices.
    Use jax.device_put() to explicitly place arrays.
    """
    # List available devices
    devices = jax.devices()
    default_device = jax.devices()[0]

    # Check device types
    cpu_devices = jax.devices('cpu')
    # gpu_devices = jax.devices('gpu')  # If GPU available

    # Create array (goes to default device)
    arr = jnp.array([1, 2, 3])

    # Check where array is
    arr_device = arr.devices()  # Returns set of devices

    # Explicit placement
    arr_on_cpu = jax.device_put(arr, jax.devices('cpu')[0])

    # Arrays are created on default backend
    # Operations between arrays on different devices will transfer data

    # Check current default backend
    default_backend = jax.default_backend()  # 'cpu', 'gpu', or 'tpu'

    # Block until computation is done (useful for benchmarking)
    result = arr * 2
    result.block_until_ready()

    return {
        'devices': [str(d) for d in devices],
        'default_backend': default_backend,
        'array_device': str(list(arr_device)[0]),
        'cpu_device': str(cpu_devices[0])
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX NumPy Basics Examples")
    print("=" * 60)

    examples = [
        ("1. Array Creation", example_array_creation),
        ("2. Reshaping & Slicing", example_reshaping_slicing),
        ("3. Math Operations", example_math_operations),
        ("4. Linear Algebra", example_linear_algebra),
        ("5. Broadcasting", example_broadcasting),
        ("6. Reductions", example_reductions),
        ("7. Boolean Indexing", example_boolean_indexing),
        ("8. Immutability & .at[]", example_immutability),
        ("9. Einsum", example_einsum),
        ("10. Device Placement", example_device_placement),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        result = func()
        for key, value in result.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
