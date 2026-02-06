"""
JAX Pytrees - 10 Exercises
==========================

Fill in the TODO sections to complete each exercise.
Run `pytest test_exercises.py` to check your implementations.

Pytrees are JAX's way of handling nested data structures.
Neural network parameters are pytrees (nested dicts of arrays).

Key concepts:
- Pytree = any nested structure of containers (dict, list, tuple)
- Leaves = the arrays at the end of the nesting
- jax.tree.* functions operate on all leaves

Reference: https://jax.readthedocs.io/en/latest/pytrees.html
"""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import NamedTuple


# =============================================================================
# Exercise 1: Basic Pytree Structure
# =============================================================================
def exercise_basic_pytree():
    """
    A pytree is any nested combination of:
    - dicts, lists, tuples (containers)
    - JAX arrays, scalars (leaves)

    Common use: neural network parameters.

    TODO:
    - Create simple_params: a dict with 'weights' (2x2 array [[1,2],[3,4]])
      and 'bias' (array [0, 0])
    - Create mlp_params: nested dict with 3 layers (layer1, layer2, layer3),
      each having 'w' and 'b':
        - layer1: w=ones((784, 256)), b=zeros(256)
        - layer2: w=ones((256, 128)), b=zeros(128)
        - layer3: w=ones((128, 10)), b=zeros(10)
    - Create list_tree: list of two arrays [array([1,2]), array([3,4,5])]
    - Create mixed_tree: dict with 'data' (list of [ones((2,2)), zeros((3,))])
      and 'config' (tuple of (array(1.0), array(2.0)))

    Return dict with:
    - 'simple_leaves': jax.tree.leaves(simple_params)
    - 'mlp_num_params': count of leaf arrays in mlp_params
    - 'list_leaves': jax.tree.leaves(list_tree)
    - 'mixed_leaves': jax.tree.leaves(mixed_tree)
    """
    # TODO: Implement this function
    simple_params = None
    mlp_params = None
    list_tree = None
    mixed_tree = None

    return {
        'simple_leaves': jax.tree.leaves(simple_params) if simple_params is not None else None,
        'mlp_num_params': len(jax.tree.leaves(mlp_params)) if mlp_params is not None else None,
        'list_leaves': jax.tree.leaves(list_tree) if list_tree is not None else None,
        'mixed_leaves': jax.tree.leaves(mixed_tree) if mixed_tree is not None else None
    }


# =============================================================================
# Exercise 2: jax.tree.map for Element-wise Operations
# =============================================================================
def exercise_tree_map():
    """
    jax.tree.map applies a function to every leaf.
    This is how you do element-wise operations on parameters.

    TODO: Given params with layer1 and layer2 (each having 'w' and 'b'):
    - doubled: Double all parameters using jax.tree.map(lambda x: x * 2, params)
    - grads: Create dummy gradients using jax.tree.map(jnp.ones_like, params)
    - updated: Apply gradient update: p - lr * g where lr=0.1
      Use jax.tree.map with two pytrees: params and grads
    - norms: Compute L2 norm of each leaf using jax.tree.map

    Hints:
    - jax.tree.map(fn, tree) applies fn to each leaf
    - jax.tree.map(fn, tree1, tree2) applies fn(leaf1, leaf2) for matching leaves
    """
    params = {
        'layer1': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
        'layer2': {'w': jnp.ones((4, 2)), 'b': jnp.zeros(2)}
    }
    lr = 0.1

    # TODO: Implement this function
    doubled = None
    grads = None
    updated = None
    norms = None

    return {
        'doubled_layer1_w_sum': jnp.sum(doubled['layer1']['w']) if doubled is not None else None,
        'updated_layer1_w_sum': jnp.sum(updated['layer1']['w']) if updated is not None else None,
        'layer1_w_norm': norms['layer1']['w'] if norms is not None else None,
        'structure_preserved': list(doubled.keys()) == list(params.keys()) if doubled is not None else None
    }


# =============================================================================
# Exercise 3: jax.tree.leaves and jax.tree.structure
# =============================================================================
def exercise_leaves_structure():
    """
    - jax.tree.leaves: extract all arrays as a flat list
    - jax.tree.structure: get the "shape" of the tree

    TODO: Given params (encoder with conv1, conv2; classifier with dense):
    - leaves: Get all leaf arrays using jax.tree.leaves
    - total_params: Sum of x.size for all leaves
    - structure: Get tree structure using jax.tree.structure
    - other_leaves: Create new leaves with same shapes but values * 2
    - reconstructed: Rebuild pytree using jax.tree.unflatten(structure, other_leaves)

    The params structure:
    - encoder/conv1: w=(3,3,3,64), b=(64,)
    - encoder/conv2: w=(3,3,64,128), b=(128,)
    - classifier/dense: w=(128,10), b=(10,)
    """
    params = {
        'encoder': {
            'conv1': {'w': jnp.ones((3, 3, 3, 64)), 'b': jnp.zeros(64)},
            'conv2': {'w': jnp.ones((3, 3, 64, 128)), 'b': jnp.zeros(128)}
        },
        'classifier': {
            'dense': {'w': jnp.ones((128, 10)), 'b': jnp.zeros(10)}
        }
    }

    # TODO: Implement this function
    leaves = None
    total_params = None
    structure = None
    other_leaves = None
    reconstructed = None

    return {
        'num_arrays': len(leaves) if leaves is not None else None,
        'total_params': total_params,
        'structure': str(structure)[:100] + '...' if structure is not None else None,
        'reconstructed_matches': list(reconstructed.keys()) == list(params.keys()) if reconstructed is not None else None
    }


# =============================================================================
# Exercise 4: jax.tree.reduce for Aggregation
# =============================================================================
def exercise_tree_reduce():
    """
    jax.tree.reduce combines all leaves using a function.
    Useful for computing total norms, sums, etc.

    TODO: Given params:
    - total_count: Count total parameters using jax.tree.reduce
      Hint: lambda acc, x: acc + x.size, initializer=0
    - l2_norm_sq: Sum of squared values across all leaves
      Hint: lambda acc, x: acc + jnp.sum(x ** 2), initializer=0.0
    - max_val: Maximum absolute value across all leaves
    - total_via_leaves: Same count using jax.tree.leaves and Python sum

    Params structure:
    - layer1: w=(10,20), b=(20,)
    - layer2: w=(20,5), b=(5,)
    """
    params = {
        'layer1': {'w': jnp.ones((10, 20)), 'b': jnp.zeros(20)},
        'layer2': {'w': jnp.ones((20, 5)), 'b': jnp.zeros(5)}
    }

    # TODO: Implement this function
    total_count = None
    l2_norm_sq = None
    max_val = None
    total_via_leaves = None

    return {
        'total_params': total_count,
        'l2_norm_squared': l2_norm_sq,
        'max_absolute': max_val,
        'total_via_leaves': total_via_leaves
    }


# =============================================================================
# Exercise 5: Custom Pytree Nodes with register_pytree_node
# =============================================================================
def exercise_custom_pytree():
    """
    Register custom classes as pytree nodes.
    This lets JAX transformations work with your own types.

    TODO:
    1. NamedTuple approach (already works as pytree):
       - Create LayerParams NamedTuple with fields 'w' and 'b'
       - Create layer instance with w=ones((3,4)), b=zeros(4)
       - Use jax.tree.map to double all values

    2. Custom class approach:
       - CustomLayer class is provided with w, b, name attributes
       - Register it using jax.tree_util.register_pytree_node
       - flatten_custom: return ((w, b), name) - children and aux_data
       - unflatten_custom: reconstruct from aux_data and children
       - Create custom instance and scale by 3 using tree.map

    Hints:
    - flatten returns (children_tuple, aux_data)
    - unflatten receives (aux_data, children_tuple)
    """
    # Method 1: NamedTuple
    class LayerParams(NamedTuple):
        w: jnp.ndarray
        b: jnp.ndarray

    # TODO: Create layer and doubled_layer
    layer = None
    doubled_layer = None

    # Method 2: Custom class (provided)
    class CustomLayer:
        def __init__(self, w, b, name):
            self.w = w
            self.b = b
            self.name = name

    # TODO: Define flatten and unflatten functions, register the class
    def flatten_custom(layer):
        """Return (children, aux_data)."""
        # TODO: Implement
        children = None
        aux_data = None
        return children, aux_data

    def unflatten_custom(aux_data, children):
        """Reconstruct from children and aux_data."""
        # TODO: Implement
        return None

    # TODO: Register CustomLayer and create scaled instance
    # jax.tree_util.register_pytree_node(CustomLayer, flatten_custom, unflatten_custom)
    custom = None
    scaled = None

    return {
        'namedtuple_works': isinstance(doubled_layer, LayerParams) if doubled_layer is not None else None,
        'namedtuple_w_sum': jnp.sum(doubled_layer.w) if doubled_layer is not None else None,
        'custom_name_preserved': scaled.name == "dense1" if scaled is not None else None,
        'custom_w_sum': jnp.sum(scaled.w) if scaled is not None else None
    }


# =============================================================================
# Exercise 6: Pytrees for Neural Network Parameters
# =============================================================================
def exercise_nn_params():
    """
    Real-world pattern: managing neural network parameters.

    TODO:
    1. Implement init_mlp(key, layer_sizes) that returns params dict:
       - For each pair of consecutive layer sizes, create layer{i} with:
         - 'w': random normal * sqrt(2/(fan_in + fan_out)) (Xavier init)
         - 'b': zeros of fan_out size
       - Use jax.random.split to get new keys

    2. Forward pass is provided - just understand it

    3. Compute gradients of loss w.r.t. params using jax.grad

    4. Update params using tree_map: new_p = p - lr * g

    layer_sizes = [784, 256, 128, 10]
    """
    def init_mlp(key, layer_sizes):
        """Initialize MLP parameters."""
        # TODO: Implement this function
        params = {}
        return params

    def forward(params, x):
        """MLP forward pass (provided)."""
        for name, layer in params.items():
            x = x @ layer['w'] + layer['b']
            if name != list(params.keys())[-1]:
                x = jnp.maximum(0, x)  # ReLU
        return x

    def loss_fn(params, x, y):
        """MSE loss (provided)."""
        pred = forward(params, x)
        return jnp.mean((pred - y) ** 2)

    # Initialize
    key = jax.random.key(42)
    layer_sizes = [784, 256, 128, 10]

    # TODO: Initialize params
    params = None

    # Forward pass
    x = jnp.ones((32, 784))
    y = jnp.zeros((32, 10))

    # TODO: Compute output, gradients, and updated params
    output = None
    grads = None
    lr = 0.01
    new_params = None

    return {
        'output_shape': output.shape if output is not None else None,
        'grad_layer0_w_shape': grads['layer0']['w'].shape if grads is not None else None,
        'params_updated': not jnp.allclose(params['layer0']['w'], new_params['layer0']['w']) if (params is not None and new_params is not None) else None
    }


# =============================================================================
# Exercise 7: jax.tree.unflatten and jax.tree.flatten
# =============================================================================
def exercise_flatten_unflatten():
    """
    Flatten converts pytree to (leaves, treedef).
    Unflatten reconstructs from (leaves, treedef).

    Useful for:
    - Interfacing with optimizers
    - Serialization
    - Custom operations on all parameters

    TODO: Given params:
    - Use jax.tree_util.tree_flatten to get (leaves, treedef)
    - Get shapes of all leaves
    - Concatenate all flattened leaves into one 1D array
    - Modify leaves by adding 1 to each
    - Unflatten back to pytree using jax.tree_util.tree_unflatten

    Also verify roundtrip:
    - Use jax.tree.leaves and jax.tree.structure
    - Reconstruct with jax.tree.unflatten
    """
    params = {
        'encoder': {'w': jnp.ones((10, 20)), 'b': jnp.zeros(20)},
        'decoder': {'w': jnp.ones((20, 10)), 'b': jnp.zeros(10)}
    }

    # TODO: Implement this function
    leaves = None
    treedef = None
    shapes = None
    flat_params = None
    modified_leaves = None
    modified_params = None

    # Roundtrip verification
    leaves2 = None
    struct2 = None
    reconstructed = None

    return {
        'num_leaves': len(leaves) if leaves is not None else None,
        'leaf_shapes': shapes,
        'flat_params_size': flat_params.size if flat_params is not None else None,
        'modified_sum': jnp.sum(modified_params['encoder']['w']) if modified_params is not None else None,
        'roundtrip_works': jnp.allclose(reconstructed['encoder']['w'], params['encoder']['w']) if reconstructed is not None else None
    }


# =============================================================================
# Exercise 8: Pytree Transformations with jit and grad
# =============================================================================
def exercise_pytree_transforms():
    """
    JAX transformations (jit, grad, vmap) work seamlessly with pytrees.

    TODO:
    1. Define model(params, x): h = tanh(x @ w1 + b1), out = h @ w2 + b2
    2. Define loss(params, x, y): MSE between model output and y
    3. JIT compile the model
    4. Compute gradients w.r.t. params
    5. Use value_and_grad to get both loss and gradients

    params = {'w1': ones((4,8)), 'b1': zeros(8), 'w2': ones((8,2)), 'b2': zeros(2)}
    x = ones((16, 4)), y = ones((16, 2))
    """
    def model(params, x):
        """Simple model with pytree params."""
        # TODO: Implement
        return None

    def loss(params, x, y):
        """MSE loss."""
        # TODO: Implement
        return None

    params = {
        'w1': jnp.ones((4, 8)),
        'b1': jnp.zeros(8),
        'w2': jnp.ones((8, 2)),
        'b2': jnp.zeros(2)
    }

    x = jnp.ones((16, 4))
    y = jnp.ones((16, 2))

    # TODO: JIT, grad, value_and_grad
    jitted_model = None
    output = None
    grads = None
    loss_val = None

    return {
        'output_shape': output.shape if output is not None else None,
        'grad_keys_match': set(params.keys()) == set(grads.keys()) if grads is not None else None,
        'grad_w1_shape': grads['w1'].shape if grads is not None else None,
        'loss_value': loss_val
    }


# =============================================================================
# Exercise 9: Combining Multiple Pytrees
# =============================================================================
def exercise_combine_pytrees():
    """
    Operations involving multiple pytrees with matching structure.

    TODO: Implement SGD with momentum:
    1. new_momentum = beta * momentum + grads
    2. new_params = params - lr * new_momentum

    Then implement simplified Adam moment updates:
    1. new_m = beta1 * m + (1 - beta1) * grads
    2. new_v = beta2 * v + (1 - beta2) * grads^2

    Use jax.tree.map with multiple pytrees.

    params = {'w': ones((3,4)), 'b': zeros(4)}
    grads = {'w': ones((3,4)) * 0.1, 'b': ones(4) * 0.01}
    momentum = {'w': zeros((3,4)), 'b': zeros(4)}
    """
    params = {
        'w': jnp.ones((3, 4)),
        'b': jnp.zeros(4)
    }
    grads = {
        'w': jnp.ones((3, 4)) * 0.1,
        'b': jnp.ones(4) * 0.01
    }
    momentum = {
        'w': jnp.zeros((3, 4)),
        'b': jnp.zeros(4)
    }

    beta = 0.9
    lr = 0.01

    # TODO: SGD with momentum
    new_momentum = None
    new_params = None

    # TODO: Adam moments (simplified)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    beta1, beta2 = 0.9, 0.999

    new_m = None
    new_v = None

    return {
        'momentum_w_sum': jnp.sum(new_momentum['w']) if new_momentum is not None else None,
        'params_updated_w_sum': jnp.sum(new_params['w']) if new_params is not None else None,
        'adam_m_w_sum': jnp.sum(new_m['w']) if new_m is not None else None,
        'adam_v_w_sum': jnp.sum(new_v['w']) if new_v is not None else None
    }


# =============================================================================
# Exercise 10: Pytree Utilities - tree_all, tree_any, etc.
# =============================================================================
def exercise_pytree_utilities():
    """
    Additional pytree utilities for checking conditions,
    comparing structures, and more.

    TODO: Given params:
    - all_finite: Check if all leaves are finite (no NaN/Inf)
      Hint: all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params))
    - shapes: Get shape of each leaf using tree_map
    - counts: Get size of each leaf using tree_map
    - same_structure: Compare structure of params and params*2
      Hint: jax.tree.structure(a) == jax.tree.structure(b)
    - leaves_with_path: Use jax.tree_util.tree_leaves_with_path
    - stacked: Stack 3 param copies along axis 0
      Hint: jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batched_params)
    """
    params = {
        'layer1': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
        'layer2': {'w': jnp.ones((4, 2)) * 0.5, 'b': jnp.zeros(2)}
    }

    # TODO: Implement this function
    all_finite = None
    shapes = None
    counts = None
    params2 = jax.tree.map(lambda x: x * 2, params)
    same_structure = None
    leaves_with_path = None
    paths = None

    # Stack multiple param copies
    batched_params = [
        {'w': jnp.ones((3, 4)) * i, 'b': jnp.zeros(4)} for i in range(3)
    ]
    stacked = None

    return {
        'all_finite': all_finite,
        'shapes': shapes,
        'param_counts': counts,
        'same_structure': same_structure,
        'paths': paths[:2] if paths is not None else None,
        'stacked_w_shape': stacked['w'].shape if stacked is not None else None
    }


# =============================================================================
# Run exercises to test your implementations
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Pytrees Exercises")
    print("=" * 60)
    print("\nRun 'pytest test_exercises.py -v' to check your solutions!")
    print("\nOr run individual exercises below:\n")

    exercises = [
        ("1. Basic Pytree", exercise_basic_pytree),
        ("2. tree_map", exercise_tree_map),
        ("3. leaves and structure", exercise_leaves_structure),
        ("4. tree_reduce", exercise_tree_reduce),
        ("5. Custom Pytree", exercise_custom_pytree),
        ("6. NN Parameters", exercise_nn_params),
        ("7. flatten/unflatten", exercise_flatten_unflatten),
        ("8. Pytree Transforms", exercise_pytree_transforms),
        ("9. Combining Pytrees", exercise_combine_pytrees),
        ("10. Utilities", exercise_pytree_utilities),
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
