"""
JAX NumPy Basics - 10 Exercises
===============================

Fill in the TODO sections to complete each exercise.
Run `pytest test_exercises.py` to check your implementations.

Reference: https://jax.readthedocs.io/en/latest/jax.numpy.html
"""

import jax
import jax.numpy as jnp


# =============================================================================
# Exercise 1: Array Creation
# =============================================================================
def exercise_array_creation():
    """
    Create arrays using various jnp functions.
    Similar to NumPy but returns JAX DeviceArrays.

    TODO: Create the following arrays:
    - arr1: A 1D array from the list [1, 2, 3, 4, 5]
    - zeros: A 3x4 matrix of zeros
    - ones: A 2x3 matrix of ones with dtype float32
    - range_arr: Values [0, 2, 4, 6, 8] using jnp.arange
    - linspace_arr: 5 evenly spaced points from 0 to 1
    - eye: A 3x3 identity matrix
    - full: A 2x2 matrix filled with 7.0
    """
    arr1 = jnp.array([1,2,3,4,5])    # arange(start, end, interval)
    zeros = jnp.zeros((3,4))  # zeros(shape, dtype...)
    ones = jnp.ones((2, 3))
    range_arr = jnp.arange(0, 10, 2) 
    linspace_arr = jnp.linspace(0, 1, 5)   # linearly/evenly spaced positions, from 0 to 1. 
    eye = jnp.eye(3) 
    full = jnp.full((2,2), 7.0)
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
# Exercise 2: Array Reshaping and Slicing
# =============================================================================
def exercise_reshaping_slicing():
    """
    Reshape arrays and extract slices.
    Slicing works like NumPy but returns views that are still immutable.

    TODO: Starting with arr = jnp.arange(12):
    - reshaped: Reshape to 3x4
    - row: Extract first row (shape should be (4,))
    - col: Extract first column (shape should be (3,))
    - submat: Extract rows 1-2 (inclusive), columns 1-2 (shape should be (2,2))
    - transposed: Transpose of reshaped
    - expanded: Add a dimension at axis 0 (shape should be (1, 12))
    - squeezed: Remove the added dimension
    """
    arr = jnp.arange(12)

    # TODO: Implement this function
    reshaped = arr.reshape((3,4))
    row = reshaped[0] 

    assert row.shape == (4,)

    col = reshaped[:, 0]
    assert col.shape == (3,)
    

    submat = reshaped[1:3, 1:3]
    transposed = jnp.transpose(reshaped)
    expanded = jnp.expand_dims(arr, 0)
    squeezed = jnp.squeeze(expanded, 0)

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
# Exercise 3: Mathematical Operations
# =============================================================================
def exercise_math_operations():
    """
    Element-wise mathematical functions.
    These are automatically differentiable with jax.grad.

    TODO:
    - sin: Compute sin of x where x = jnp.linspace(0, 2*pi, 100)
    - cos: Compute cos of same x
    - exp: Compute exp of y where y = jnp.linspace(0.1, 5, 50)
    - log: Compute natural log of same y
    - sqrt: Compute sqrt of z = jnp.array([1, 4, 9, 16, 25])
    - abs: Compute absolute value of w = jnp.array([-3, -1, 0, 1, 3])
    - sign: Compute sign of same w
    """
    x = jnp.linspace(0, 2 * jnp.pi, 100)
    y = jnp.linspace(0.1, 5, 50)
    z = jnp.array([1, 4, 9, 16, 25])
    w = jnp.array([-3, -1, 0, 1, 3])

    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    exp_y = jnp.exp(y)
    log_y = jnp.log(y)
    sqrt_z = jnp.sqrt(z)
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
# Exercise 4: Linear Algebra
# =============================================================================
def exercise_linear_algebra():
    """
    Linear algebra operations: dot products, matrix multiplication, etc.

    TODO: Given v1 = [1, 2, 3], v2 = [4, 5, 6], A = [[1,2],[3,4]], B = [[5,6],[7,8]]:
    - dot_product: Compute v1 . v2 (should be 32)
    - matmul: Compute A @ B
    - outer_product: Compute outer product of v1 and v2
    - matrix_vector: Compute A @ [1, 2]
    - vec_norm: L2 norm of v1
    - determinant: Determinant of A (should be -2)
    - trace: Trace of A (should be 5)
    - inverse: Inverse of A
    """
    v1 = jnp.array([1.0, 2.0, 3.0])
    v2 = jnp.array([4.0, 5.0, 6.0])
    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    B = jnp.array([[5, 6], [7, 8]], dtype=jnp.float32)
    x = jnp.array([1.0, 2.0])

    # TODO: Implement this function
    dot_product = jnp.dot(v1, v2)
    matmul_result = A @ B

    outer = jnp.linalg.outer(v1, v2)   # outer product to create array from 2 1-ds
    Ax = A @ x
    vec_norm = jnp.linalg.norm(v1)
    det_A = jnp.linalg.det(A)
    trace_A = jnp.linalg.trace(A)
    A_inv = jnp.linalg.pinv(A)

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
# Exercise 5: Broadcasting Rules
# =============================================================================
def exercise_broadcasting():
    """
    Broadcasting allows operations on arrays with different shapes.
    Rules: dimensions are compatible if they're equal or one of them is 1.

    TODO:
    - scalar_add: Add 10 to arr = [1, 2, 3]
    - with_row_vec: Add row_vec = [1,2,3,4] to matrix = ones((3,4))
    - with_col_vec: Add col_vec = [[10],[20],[30]] to same matrix
    - outer_via_broadcast: Create 3x3 outer product of [1,2,3] and [4,5,6] using broadcasting
      Hint: reshape one to (3,1) and other to (1,3), then multiply
    - batch_broadcast_shape: Shape after adding bias=[1,2,3,4] to batch=ones((2,3,4))
    """
    arr = jnp.array([1, 2, 3])
    matrix = jnp.ones((3, 4))
    row_vec = jnp.array([1, 2, 3, 4])
    col_vec = jnp.array([[10], [20], [30]])
    batch = jnp.ones((2, 3, 4))
    bias = jnp.array([1, 2, 3, 4])

    # TODO: Implement this function
    scalar_add = 10 + arr
    with_row = row_vec + matrix 
    with_col = col_vec + matrix
    outer_broadcast = arr.reshape((3, 1)) * (3 + arr)   # don't need reshape, note that 1's get added to the left. 
    batch_with_bias = bias + batch   # 

    return {
        'scalar_add': scalar_add,
        'with_row_vec': with_row,
        'with_col_vec': with_col,
        'outer_via_broadcast': outer_broadcast,
        'batch_broadcast_shape': batch_with_bias.shape if batch_with_bias is not None else None
    }


# =============================================================================
# Exercise 6: Reduction Operations
# =============================================================================
def exercise_reductions():
    """
    Reduction operations collapse dimensions.
    Use 'axis' to specify which dimension(s) to reduce.

    TODO: Given arr = [[1,2,3],[4,5,6],[7,8,9]] (float32):
    - sum: Sum of all elements (should be 45)
    - mean: Mean of all elements (should be 5.0)
    - max: Maximum value (should be 9)
    - row_sum: Sum of each row -> [6, 15, 24]
    - col_sum: Sum of each column -> [12, 15, 18]
    - argmax: Index of max in flattened array (should be 8)
    - cumsum: Cumulative sum of flattened array
    """
    arr = jnp.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=jnp.float32)

    # TODO: Implement this function
    total_sum = arr.sum()
    total_mean = arr.mean()
    total_max = arr.max()
    row_sum = arr.sum(axis=1)
    col_sum = arr.sum(axis=0)
    argmax_flat = arr.argmax()
    cumsum = arr.cumsum()

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
# Exercise 7: Boolean Indexing and jnp.where
# =============================================================================
def exercise_boolean_indexing():
    """
    Boolean operations and conditional selection.
    jnp.where is the main tool for conditional operations in JAX.

    TODO: Given arr = [1, -2, 3, -4, 5, -6]:
    - positive_mask: Boolean mask where arr > 0
    - num_positive: Count of positive values (should be 3)
    - abs_via_where: Absolute value using jnp.where(condition, x, y)
      Hint: where(arr > 0, arr, -arr)
    - clipped: Clip values to range [-3, 3] using jnp.clip
    - all_positive: Are all values in [[1,2],[3,4]] positive? (True)
    - any_greater_3: Is any value in [[1,2],[3,4]] > 3? (True)
    """
    arr = jnp.array([1, -2, 3, -4, 5, -6])
    matrix = jnp.array([[1, 2], [3, 4]])

    # TODO: Implement this function
    positive_mask = arr > 0  
    num_positive = positive_mask.sum()
    abs_arr = jnp.where(arr > 0, arr, -arr)
    clipped_builtin = arr.clip(-3, 3)
    all_positive = (matrix > 0).all()
    any_greater_3 = (matrix > 3).any()

    return {
        'positive_mask': positive_mask,
        'num_positive': num_positive,
        'abs_via_where': abs_arr,
        'clipped': clipped_builtin,
        'all_positive': all_positive,
        'any_greater_3': any_greater_3
    }


# =============================================================================
# Exercise 8: Immutability and .at[].set() Pattern
# =============================================================================
def exercise_immutability():
    """
    JAX arrays are IMMUTABLE. You cannot modify them in place.
    Use .at[].set() to create a new array with modifications.

    TODO: Given arr = [1, 2, 3, 4, 5]:
    - original: Keep original unchanged
    - modified: Set index 0 to 10 -> [10, 2, 3, 4, 5]
    - multi_update: Set index 0 to 10 AND index 4 to 50 -> [10, 2, 3, 4, 50]
    - slice_update: Set indices 1:4 to 0 -> [1, 0, 0, 0, 5]
    - add_at_index: Add 100 to index 0 -> [101, 2, 3, 4, 5]
    - matrix_center: Set center of 3x3 zeros to 1.0
    - scattered: Add values [1, 2, 3] at indices [0, 2, 2] to zeros(5)
      Result should be [1, 0, 5, 0, 0] (index 2 gets 2+3=5)
    """
    arr = jnp.array([1, 2, 3, 4, 5])
    matrix = jnp.zeros((3, 3))

    # TODO: Implement this function
    arr_modified = arr.at[0].set(10)
    arr_multi = arr.at[jnp.array([0, 4])].set(jnp.array([10, 50]))
    arr_slice = arr.at[1:4].set(0)
    arr_add = arr.at[0].add(100)
    matrix_updated = matrix.at[(1,1)].set(1.0)

    indices = jnp.array([0, 2, 2])
    values = jnp.array([1.0, 2.0, 3.0])
    scattered = jnp.zeros(5).at[jnp.array([0, 2, 2])].add([1,2,3])

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
# Exercise 9: Einstein Summation (einsum)
# =============================================================================
def exercise_einsum():
    """
    jnp.einsum provides a powerful, flexible way to express tensor operations.

    Format: 'input_indices -> output_indices'
    - Repeated indices are summed over
    - Output indices determine the result shape

    TODO:
    - dot_product: Dot product of a=[1,2,3] and b=[4,5,6] using einsum
      Hint: 'i,i->' means multiply elementwise and sum all
    - outer_product: Outer product using 'i,j->ij'
    - matmul: Matrix multiply A=[[1,2],[3,4]] and B=[[5,6],[7,8]] using 'ik,kj->ij'
    - transpose: Transpose A using 'ij->ji'
    - trace: Trace of A using 'ii->' (should be 5)
    - batch_matmul_shape: Shape of batch matmul of ones((2,3,4)) @ ones((2,4,5))
      Use 'bij,bjk->bik'
    - attention_scores_shape: Shape of Q @ K^T where Q,K are (2,4,8,16)
      Use 'bhqd,bhkd->bhqk' for batch, heads, seq, dim
    """
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    A = jnp.array([[1, 2], [3, 4]])
    B = jnp.array([[5, 6], [7, 8]])
    batch_A = jnp.ones((2, 3, 4))
    batch_B = jnp.ones((2, 4, 5))
    Q = jnp.ones((2, 4, 8, 16))
    K = jnp.ones((2, 4, 8, 16))

    # TODO: Implement this function
    dot = jnp.einsum("i, i ->", a, b)
    outer = jnp.einsum("i, j -> ij", a, b)
    matmul = jnp.einsum("ik, kj -> ij ", A, B)
    transpose = jnp.einsum("ij -> ji", A)
    trace = jnp.einsum("ii -> ", A)
    batch_matmul = jnp.einsum("bij, bjk -> bik", batch_A, batch_B)
    scores = jnp.einsum("bhqd, bhkd -> bhqk", Q, K)

    return {
        'dot_product': dot,
        'outer_product': outer,
        'matmul': matmul,
        'transpose': transpose,
        'trace': trace,
        'batch_matmul_shape': batch_matmul.shape if batch_matmul is not None else None,
        'attention_scores_shape': scores.shape if scores is not None else None
    }


# =============================================================================
# Exercise 10: Device Placement
# =============================================================================
def exercise_device_placement():
    """
    JAX can run on CPU, GPU, or TPU.

    TODO:
    - devices: List of available devices as strings (use jax.devices())
    - default_backend: Current default backend ('cpu', 'gpu', or 'tpu')
    - array_device: Device where arr=[1,2,3] is placed (as string)
    - cpu_device: First CPU device as string

    Hints:
    - jax.devices() returns list of devices
    - jax.default_backend() returns 'cpu', 'gpu', or 'tpu'
    - arr.devices() returns set of devices where array is placed
    """
    arr = jnp.array([1, 2, 3])

    # TODO: Implement this function
    devices = jax.devices()
    default_backend = jax.default_backend()
    arr_device = arr.devices()
    cpu_device = str(jax.devices("cpu")[0]) 

    return {
        'devices': devices,
        'default_backend': default_backend,
        'array_device': arr_device,
        'cpu_device': cpu_device
    }


# =============================================================================
# Run exercises to test your implementations
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX NumPy Basics Exercises")
    print("=" * 60)
    print("\nRun 'pytest test_exercises.py -v' to check your solutions!")
    print("\nOr run individual exercises below:\n")

    exercises = [
        ("1. Array Creation", exercise_array_creation),
        ("2. Reshaping & Slicing", exercise_reshaping_slicing),
        ("3. Math Operations", exercise_math_operations),
        ("4. Linear Algebra", exercise_linear_algebra),
        ("5. Broadcasting", exercise_broadcasting),
        ("6. Reductions", exercise_reductions),
        ("7. Boolean Indexing", exercise_boolean_indexing),
        ("8. Immutability & .at[]", exercise_immutability),
        ("9. Einsum", exercise_einsum),
        ("10. Device Placement", exercise_device_placement),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if value is None:
                    print(f"  {key}: TODO")
                elif hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
