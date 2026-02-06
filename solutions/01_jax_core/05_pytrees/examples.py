"""
JAX Pytrees - 10 Examples
==========================

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
# Example 1: Basic Pytree Structure
# =============================================================================
def example_basic_pytree():
    """
    A pytree is any nested combination of:
    - dicts, lists, tuples (containers)
    - JAX arrays, scalars (leaves)

    Common use: neural network parameters.
    """
    # Simple pytree: dict of arrays
    simple_params = {
        'weights': jnp.array([[1, 2], [3, 4]]),
        'bias': jnp.array([0, 0])
    }

    # Nested pytree: MLP parameters
    mlp_params = {
        'layer1': {
            'w': jnp.ones((784, 256)),
            'b': jnp.zeros(256)
        },
        'layer2': {
            'w': jnp.ones((256, 128)),
            'b': jnp.zeros(128)
        },
        'layer3': {
            'w': jnp.ones((128, 10)),
            'b': jnp.zeros(10)
        }
    }

    # List/tuple pytrees
    list_tree = [jnp.array([1, 2]), jnp.array([3, 4, 5])]
    tuple_tree = (jnp.array([1.0]), jnp.array([2.0, 3.0]))

    # Mixed containers
    mixed_tree = {
        'data': [jnp.ones((2, 2)), jnp.zeros((3,))],
        'config': (jnp.array(1.0), jnp.array(2.0))
    }

    return {
        'simple_leaves': jax.tree.leaves(simple_params),
        'mlp_num_params': len(jax.tree.leaves(mlp_params)),
        'list_leaves': jax.tree.leaves(list_tree),
        'mixed_leaves': jax.tree.leaves(mixed_tree)
    }


# =============================================================================
# Example 2: jax.tree.map for Element-wise Operations
# =============================================================================
def example_tree_map():
    """
    jax.tree.map applies a function to every leaf.
    This is how you do element-wise operations on parameters.
    """
    params = {
        'layer1': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
        'layer2': {'w': jnp.ones((4, 2)), 'b': jnp.zeros(2)}
    }

    # Double all parameters
    doubled = jax.tree.map(lambda x: x * 2, params)

    # Apply different functions
    squared = jax.tree.map(lambda x: x ** 2, params)

    # Common operations in training:
    # 1. Apply gradients
    grads = jax.tree.map(jnp.ones_like, params)  # Dummy gradients
    lr = 0.1
    updated = jax.tree.map(lambda p, g: p - lr * g, params, grads)

    # 2. Initialize with zeros like existing params
    zeros = jax.tree.map(jnp.zeros_like, params)

    # 3. Compute norms
    norms = jax.tree.map(lambda x: jnp.linalg.norm(x), params)

    return {
        'doubled_layer1_w_sum': jnp.sum(doubled['layer1']['w']),
        'updated_layer1_w_sum': jnp.sum(updated['layer1']['w']),
        'layer1_w_norm': norms['layer1']['w'],
        'structure_preserved': list(doubled.keys()) == list(params.keys())
    }


# =============================================================================
# Example 3: jax.tree.leaves and jax.tree.structure
# =============================================================================
def example_leaves_structure():
    """
    - jax.tree.leaves: extract all arrays as a flat list
    - jax.tree.structure: get the "shape" of the tree
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

    # Get all leaves
    leaves = jax.tree.leaves(params)

    # Count total parameters
    total_params = sum(x.size for x in leaves)

    # Get structure (treedef)
    structure = jax.tree.structure(params)

    # Structure can be reused
    other_leaves = [jnp.ones(x.shape) * 2 for x in leaves]
    reconstructed = jax.tree.unflatten(structure, other_leaves)

    return {
        'num_arrays': len(leaves),
        'total_params': total_params,
        'structure': str(structure)[:100] + '...',
        'reconstructed_matches': list(reconstructed.keys()) == list(params.keys())
    }


# =============================================================================
# Example 4: jax.tree.reduce for Aggregation
# =============================================================================
def example_tree_reduce():
    """
    jax.tree.reduce combines all leaves using a function.
    Useful for computing total norms, sums, etc.
    """
    params = {
        'layer1': {'w': jnp.ones((10, 20)), 'b': jnp.zeros(20)},
        'layer2': {'w': jnp.ones((20, 5)), 'b': jnp.zeros(5)}
    }

    # Total number of parameters
    total_count = jax.tree.reduce(
        lambda acc, x: acc + x.size,
        params,
        initializer=0
    )

    # Total L2 norm squared (for regularization)
    l2_norm_sq = jax.tree.reduce(
        lambda acc, x: acc + jnp.sum(x ** 2),
        params,
        initializer=0.0
    )

    # Max absolute value
    max_val = jax.tree.reduce(
        lambda acc, x: jnp.maximum(acc, jnp.max(jnp.abs(x))),
        params,
        initializer=0.0
    )

    # Alternative: use tree_util.tree_flatten and regular Python
    leaves, _ = jax.tree_util.tree_flatten(params)
    total_via_leaves = sum(x.size for x in leaves)

    return {
        'total_params': total_count,
        'l2_norm_squared': l2_norm_sq,
        'max_absolute': max_val,
        'total_via_leaves': total_via_leaves
    }


# =============================================================================
# Example 5: Custom Pytree Nodes with register_pytree_node
# =============================================================================
def example_custom_pytree():
    """
    Register custom classes as pytree nodes.
    This lets JAX transformations work with your own types.
    """
    # Method 1: Using NamedTuple (automatically a pytree!)
    class LayerParams(NamedTuple):
        w: jnp.ndarray
        b: jnp.ndarray

    layer = LayerParams(w=jnp.ones((3, 4)), b=jnp.zeros(4))

    # NamedTuples work automatically
    doubled_layer = jax.tree.map(lambda x: x * 2, layer)

    # Method 2: Custom class with explicit registration
    class CustomLayer:
        def __init__(self, w, b, name):
            self.w = w
            self.b = b
            self.name = name  # Non-array attribute (metadata)

    # Register the class
    def flatten_custom(layer):
        """Return (children, aux_data)."""
        children = (layer.w, layer.b)
        aux_data = layer.name  # Metadata
        return children, aux_data

    def unflatten_custom(aux_data, children):
        """Reconstruct from children and aux_data."""
        w, b = children
        return CustomLayer(w, b, aux_data)

    jax.tree_util.register_pytree_node(
        CustomLayer,
        flatten_custom,
        unflatten_custom
    )

    # Now CustomLayer works with JAX!
    custom = CustomLayer(jnp.ones((3, 4)), jnp.zeros(4), "dense1")
    scaled = jax.tree.map(lambda x: x * 3, custom)

    return {
        'namedtuple_works': isinstance(doubled_layer, LayerParams),
        'namedtuple_w_sum': jnp.sum(doubled_layer.w),
        'custom_name_preserved': scaled.name == "dense1",
        'custom_w_sum': jnp.sum(scaled.w)
    }


# =============================================================================
# Example 6: Pytrees for Neural Network Parameters
# =============================================================================
def example_nn_params():
    """
    Real-world pattern: managing neural network parameters.
    """
    def init_mlp(key, layer_sizes):
        """Initialize MLP parameters."""
        params = {}
        for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            key, subkey = jax.random.split(key)
            # Xavier initialization
            scale = jnp.sqrt(2.0 / (fan_in + fan_out))
            params[f'layer{i}'] = {
                'w': jax.random.normal(subkey, (fan_in, fan_out)) * scale,
                'b': jnp.zeros(fan_out)
            }
        return params

    def forward(params, x):
        """MLP forward pass."""
        for name, layer in params.items():
            x = x @ layer['w'] + layer['b']
            if name != list(params.keys())[-1]:  # ReLU except last
                x = jnp.maximum(0, x)
        return x

    # Initialize
    key = jax.random.key(42)
    layer_sizes = [784, 256, 128, 10]
    params = init_mlp(key, layer_sizes)

    # Forward pass
    x = jnp.ones((32, 784))  # Batch of 32
    output = forward(params, x)

    # Compute gradients
    def loss_fn(params, x, y):
        pred = forward(params, x)
        return jnp.mean((pred - y) ** 2)

    y = jnp.zeros((32, 10))
    grads = jax.grad(loss_fn)(params, x, y)

    # Update with tree_map
    lr = 0.01
    new_params = jax.tree.map(
        lambda p, g: p - lr * g,
        params, grads
    )

    return {
        'output_shape': output.shape,
        'grad_layer0_w_shape': grads['layer0']['w'].shape,
        'params_updated': not jnp.allclose(
            params['layer0']['w'],
            new_params['layer0']['w']
        )
    }


# =============================================================================
# Example 7: jax.tree.unflatten and jax.tree.flatten
# =============================================================================
def example_flatten_unflatten():
    """
    Flatten converts pytree to (leaves, treedef).
    Unflatten reconstructs from (leaves, treedef).

    Useful for:
    - Interfacing with optimizers
    - Serialization
    - Custom operations on all parameters
    """
    params = {
        'encoder': {'w': jnp.ones((10, 20)), 'b': jnp.zeros(20)},
        'decoder': {'w': jnp.ones((20, 10)), 'b': jnp.zeros(10)}
    }

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(params)

    # leaves is a list, treedef remembers structure
    num_leaves = len(leaves)
    shapes = [x.shape for x in leaves]

    # Concatenate all params (useful for some optimizers)
    flat_params = jnp.concatenate([x.flatten() for x in leaves])

    # Modify leaves
    modified_leaves = [x + 1 for x in leaves]

    # Unflatten back to pytree
    modified_params = jax.tree_util.tree_unflatten(treedef, modified_leaves)

    # Alternative: tree.flatten and tree.unflatten
    leaves2 = jax.tree.leaves(params)
    struct2 = jax.tree.structure(params)
    reconstructed = jax.tree.unflatten(struct2, leaves2)

    return {
        'num_leaves': num_leaves,
        'leaf_shapes': shapes,
        'flat_params_size': flat_params.size,
        'modified_sum': jnp.sum(modified_params['encoder']['w']),
        'roundtrip_works': jnp.allclose(
            reconstructed['encoder']['w'],
            params['encoder']['w']
        )
    }


# =============================================================================
# Example 8: Pytree Transformations with jit and grad
# =============================================================================
def example_pytree_transforms():
    """
    JAX transformations (jit, grad, vmap) work seamlessly with pytrees.
    """
    def model(params, x):
        """Simple model with pytree params."""
        h = x @ params['w1'] + params['b1']
        h = jnp.tanh(h)
        out = h @ params['w2'] + params['b2']
        return out

    def loss(params, x, y):
        pred = model(params, x)
        return jnp.mean((pred - y) ** 2)

    # Initialize
    params = {
        'w1': jnp.ones((4, 8)),
        'b1': jnp.zeros(8),
        'w2': jnp.ones((8, 2)),
        'b2': jnp.zeros(2)
    }

    x = jnp.ones((16, 4))
    y = jnp.ones((16, 2))

    # JIT works with pytree inputs/outputs
    jitted_model = jax.jit(model)
    output = jitted_model(params, x)

    # Grad returns a pytree with same structure
    grads = jax.grad(loss)(params, x, y)

    # Verify gradient structure matches params
    param_keys = set(params.keys())
    grad_keys = set(grads.keys())

    # Value and grad
    loss_val, grads = jax.value_and_grad(loss)(params, x, y)

    return {
        'output_shape': output.shape,
        'grad_keys_match': param_keys == grad_keys,
        'grad_w1_shape': grads['w1'].shape,
        'loss_value': loss_val
    }


# =============================================================================
# Example 9: Combining Multiple Pytrees
# =============================================================================
def example_combine_pytrees():
    """
    Operations involving multiple pytrees with matching structure.
    """
    # Two pytrees with same structure
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

    # tree_map with multiple pytrees
    # SGD with momentum
    beta = 0.9
    lr = 0.01

    new_momentum = jax.tree.map(
        lambda m, g: beta * m + g,
        momentum, grads
    )

    new_params = jax.tree.map(
        lambda p, m: p - lr * m,
        params, new_momentum
    )

    # Adam-style update (simplified)
    m = jax.tree.map(jnp.zeros_like, params)  # First moment
    v = jax.tree.map(jnp.zeros_like, params)  # Second moment

    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Update moments
    new_m = jax.tree.map(
        lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i,
        m, grads
    )
    new_v = jax.tree.map(
        lambda v_i, g_i: beta2 * v_i + (1 - beta2) * g_i ** 2,
        v, grads
    )

    # Compute update
    updates = jax.tree.map(
        lambda m_i, v_i: m_i / (jnp.sqrt(v_i) + eps),
        new_m, new_v
    )

    return {
        'momentum_w_sum': jnp.sum(new_momentum['w']),
        'params_updated_w_sum': jnp.sum(new_params['w']),
        'adam_m_w_sum': jnp.sum(new_m['w']),
        'adam_v_w_sum': jnp.sum(new_v['w'])
    }


# =============================================================================
# Example 10: Pytree Utilities - tree_all, tree_any, etc.
# =============================================================================
def example_pytree_utilities():
    """
    Additional pytree utilities for checking conditions,
    comparing structures, and more.
    """
    params = {
        'layer1': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
        'layer2': {'w': jnp.ones((4, 2)) * 0.5, 'b': jnp.zeros(2)}
    }

    # Check if all leaves satisfy a condition
    all_finite = all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params))

    # Check shapes
    shapes = jax.tree.map(lambda x: x.shape, params)

    # Count parameters per layer
    counts = jax.tree.map(lambda x: x.size, params)

    # Check structure equality
    params2 = jax.tree.map(lambda x: x * 2, params)
    same_structure = (
        jax.tree.structure(params) == jax.tree.structure(params2)
    )

    # Get leaf paths (for debugging)
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(params))
    paths = [str(path) for path, _ in leaves_with_path]

    # Transpose a list of pytrees to a pytree of lists
    batched_params = [
        {'w': jnp.ones((3, 4)) * i, 'b': jnp.zeros(4)} for i in range(3)
    ]

    # Stack leaves across the batch
    stacked = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *batched_params
    )

    return {
        'all_finite': all_finite,
        'shapes': shapes,
        'param_counts': counts,
        'same_structure': same_structure,
        'paths': paths[:2],  # First two paths
        'stacked_w_shape': stacked['w'].shape
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX Pytrees Examples")
    print("=" * 60)

    examples = [
        ("1. Basic Pytree", example_basic_pytree),
        ("2. tree_map", example_tree_map),
        ("3. leaves and structure", example_leaves_structure),
        ("4. tree_reduce", example_tree_reduce),
        ("5. Custom Pytree", example_custom_pytree),
        ("6. NN Parameters", example_nn_params),
        ("7. flatten/unflatten", example_flatten_unflatten),
        ("8. Pytree Transforms", example_pytree_transforms),
        ("9. Combining Pytrees", example_combine_pytrees),
        ("10. Utilities", example_pytree_utilities),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
