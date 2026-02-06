"""
Einops with JAX - 10 Exercises
==============================

Fill in the TODO sections to complete each exercise.
Run `pytest test_exercises.py` to check your implementations.

Einops provides readable tensor operations:
- rearrange: reshape, transpose, split, merge axes
- reduce: reduce dimensions (mean, sum, max, etc.)
- repeat: tile/repeat along new or existing axes

Reference: https://einops.rocks/
Install: pip install einops
"""

import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat, einsum


# =============================================================================
# Exercise 1: Basic Rearrange - Transpose and Reshape
# =============================================================================
def exercise_basic_rearrange():
    """
    rearrange is the core operation - combines reshape, transpose, squeeze, etc.
    Pattern: 'input_axes -> output_axes'

    TODO:
    - transpose_shape: Transpose (H, W, C) -> (C, H, W)
      Use: rearrange(image_hwc, 'h w c -> c h w')
    - flatten_shape: Flatten (4, 5) -> (20,)
      Use: rearrange(matrix, 'h w -> (h w)')
    - unflatten_shape: Unflatten (20,) -> (4, 5)
      Use: rearrange(flat, '(h w) -> h w', h=4, w=5)
    - batch_transpose_shape: (B, H, W, C) -> (B, C, H, W)
    """
    image_hwc = jnp.ones((32, 48, 3))
    matrix = jnp.ones((4, 5))
    batch_images = jnp.ones((16, 32, 32, 3))

    # TODO: Implement this function
    image_chw = None
    flat = None
    unflat = None
    batch_chw = None

    return {
        'transpose_shape': image_chw.shape if image_chw is not None else None,
        'flatten_shape': flat.shape if flat is not None else None,
        'unflatten_shape': unflat.shape if unflat is not None else None,
        'batch_transpose_shape': batch_chw.shape if batch_chw is not None else None,
    }


# =============================================================================
# Exercise 2: Splitting and Merging Axes
# =============================================================================
def exercise_split_merge():
    """
    Split one axis into multiple, or merge multiple into one.
    Use parentheses to group axes.

    TODO:
    - grid_shape: Split 12 samples into 3x4 grid
      rearrange(batch, '(rows cols) f -> rows cols f', rows=3, cols=4)
    - flat_spatial_shape: Merge H,W: (B, H, W, C) -> (B, H*W, C)
      rearrange(images, 'b h w c -> b (h w) c')
    - grouped_shape: Split channels into 8 groups: (16, 128) -> (16, 8, 16)
      rearrange(features, 'b (g c) -> b g c', g=8)
    - merged_shape: Merge batch and time: (4, 100, 256) -> (400, 256)
    """
    batch = jnp.arange(12).reshape(12, 4)
    images = jnp.ones((8, 32, 32, 64))
    features = jnp.ones((16, 128))
    sequences = jnp.ones((4, 100, 256))

    # TODO: Implement this function
    grid = None
    flat_spatial = None
    grouped = None
    merged = None

    return {
        'grid_shape': grid.shape if grid is not None else None,
        'flat_spatial_shape': flat_spatial.shape if flat_spatial is not None else None,
        'grouped_shape': grouped.shape if grouped is not None else None,
        'merged_shape': merged.shape if merged is not None else None,
    }


# =============================================================================
# Exercise 3: Image Patching (Vision Transformer Style)
# =============================================================================
def exercise_image_patching():
    """
    Extract patches from images - essential for Vision Transformers.

    TODO:
    - patches_shape: Extract 16x16 patches from 224x224 image
      Pattern: '(h ph) (w pw) c -> (h w) (ph pw c)' with ph=16, pw=16
      Result: 196 patches (14*14), each 768 dims (16*16*3)
    - batch_patches_shape: Same but for batch of images
      Add 'b' at the start of pattern
    - reconstruction_correct: Reverse the patching to reconstruct image
      Pattern: '(h w) (ph pw c) -> (h ph) (w pw) c' with h=14, w=14, ph=16, pw=16, c=3
    """
    image = jnp.ones((224, 224, 3))
    batch_images = jnp.ones((8, 224, 224, 3))
    patch_size = 16

    # TODO: Implement this function
    patches = None
    batch_patches = None
    reconstructed = None

    return {
        'patches_shape': patches.shape if patches is not None else None,
        'batch_patches_shape': batch_patches.shape if batch_patches is not None else None,
        'reconstructed_shape': reconstructed.shape if reconstructed is not None else None,
        'reconstruction_correct': jnp.allclose(image, reconstructed) if reconstructed is not None else None,
    }


# =============================================================================
# Exercise 4: Attention Reshaping (Multi-Head Attention)
# =============================================================================
def exercise_attention_reshape():
    """
    Reshape for multi-head attention: split embed_dim into heads.

    TODO:
    - multi_head_shape: Split (B, T, D) -> (B, heads, T, head_dim)
      Pattern: 'b t (h d) -> b h t d' with h=num_heads
    - merged_shape: Merge back (B, heads, T, head_dim) -> (B, T, D)
      Pattern: 'b h t d -> b t (h d)'
    - q_shape, k_shape, v_shape: Split QKV projection
      Input (B, T, 3*D), output 3 tensors of (B, heads, T, head_dim)
      Pattern: 'b t (qkv h d) -> qkv b h t d' with qkv=3, h=num_heads
    """
    batch_size, seq_len, embed_dim = 4, 128, 512
    num_heads = 8
    head_dim = embed_dim // num_heads  # 64

    x = jnp.ones((batch_size, seq_len, embed_dim))
    qkv = jnp.ones((batch_size, seq_len, 3 * embed_dim))

    # TODO: Implement this function
    multi_head = None
    merged = None
    q, k, v = None, None, None

    return {
        'multi_head_shape': multi_head.shape if multi_head is not None else None,
        'merged_shape': merged.shape if merged is not None else None,
        'q_shape': q.shape if q is not None else None,
        'k_shape': k.shape if k is not None else None,
        'v_shape': v.shape if v is not None else None,
    }


# =============================================================================
# Exercise 5: Reduce Operations
# =============================================================================
def exercise_reduce():
    """
    reduce combines reduction with rearrangement.

    TODO:
    - global_pool_shape: Global average pooling (B, H, W, C) -> (B, C)
      reduce(images, 'b h w c -> b c', 'mean')
    - seq_mean_shape: Mean over sequence (B, T, D) -> (B, D)
    - max_pool_shape: 2x2 max pooling (B, H, W, C) -> (B, H/2, W/2, C)
      reduce(images, 'b (h h2) (w w2) c -> b h w c', 'max', h2=2, w2=2)
    - variance: Compute variance using mean of squares - square of mean
    """
    images = jnp.ones((8, 32, 32, 64)) * jnp.arange(64)
    sequences = jnp.arange(24).reshape(2, 3, 4).astype(float)
    data = jnp.array([[1., 2., 3.], [4., 5., 6.]])

    # TODO: Implement this function
    pooled = None
    seq_mean = None
    max_pooled = None
    variance = None

    return {
        'global_pool_shape': pooled.shape if pooled is not None else None,
        'seq_mean_shape': seq_mean.shape if seq_mean is not None else None,
        'max_pool_shape': max_pooled.shape if max_pooled is not None else None,
        'group_sum_shape': (16, 4),  # Provided
        'variance': variance,
    }


# =============================================================================
# Exercise 6: Repeat Operations
# =============================================================================
def exercise_repeat():
    """
    repeat tiles/broadcasts arrays along new or existing axes.

    TODO:
    - batched_shape: Add batch dimension (H, W, C) -> (B, H, W, C)
      repeat(image, 'h w c -> b h w c', b=8)
    - tiled_shape: Tile channels 3x: (T, D) -> (T, D*3)
      repeat(seq, 't d -> t (repeat d)', repeat=3)
    - broadcast_mask_shape: Broadcast mask for attention (B, T) -> (B, heads, T, T)
      repeat(mask, 'b t -> b h t t2', h=8, t2=128)
    - interleaved: [1,2,3] -> [1,1,2,2,3,3]
      repeat(arr, 'n -> (n repeat)', repeat=2)
    """
    image = jnp.ones((32, 32, 3))
    seq = jnp.ones((10, 64))
    mask = jnp.ones((4, 128))
    arr = jnp.array([1, 2, 3])

    # TODO: Implement this function
    batched = None
    tiled = None
    broadcast_mask = None
    interleaved = None

    return {
        'batched_shape': batched.shape if batched is not None else None,
        'tiled_shape': tiled.shape if tiled is not None else None,
        'broadcast_mask_shape': broadcast_mask.shape if broadcast_mask is not None else None,
        'pos_expanded_shape': (100, 64),  # Provided
        'interleaved': interleaved,
    }


# =============================================================================
# Exercise 7: Einops with JAX Transformations
# =============================================================================
def exercise_jax_transforms():
    """
    Einops works seamlessly with jit, grad, and vmap.

    TODO:
    - jit_result_shape: JIT a function that uses rearrange
      @jax.jit def fn(x): return rearrange(x, 'b t (h d) -> b h t d', h=8)
    - grad_shape: Use grad with a function that uses rearrange + reduce
    - vmap_result_shape: Use vmap with a function that uses rearrange
    """
    x = jnp.ones((4, 128, 512))
    params = jnp.ones((64, 32))
    x_img = jnp.ones((2, 8, 8, 64))
    batch_images = jnp.ones((16, 32, 32, 3))

    # TODO: Implement this function
    # Hint for JIT:
    # @jax.jit
    # def attention_reshape_jit(x, num_heads):
    #     return rearrange(x, 'b t (h d) -> b h t d', h=num_heads)
    # result_jit = attention_reshape_jit(x, 8)

    result_jit = None
    grads = None
    vmap_result = None

    return {
        'jit_result_shape': result_jit.shape if result_jit is not None else None,
        'grad_shape': grads.shape if grads is not None else None,
        'vmap_result_shape': vmap_result.shape if vmap_result is not None else None,
        'all_transforms_work': result_jit is not None and grads is not None and vmap_result is not None,
    }


# =============================================================================
# Exercise 8: Einsum with Einops
# =============================================================================
def exercise_einops_einsum():
    """
    einops.einsum uses named dimensions instead of single letters.

    TODO:
    - matmul_shape: Matrix multiply A (3,4) @ B (4,5)
      einsum(A, B, 'i j, j k -> i k')
    - batch_matmul_shape: Batch matrix multiply
      einsum(batch_A, batch_B, 'b i j, b j k -> b i k')
    - attention_scores_shape: Q @ K^T for attention
      einsum(Q, K, 'b h q d, b h k d -> b h q k')
    - attention_output_shape: scores @ V
      einsum(attn_scores, V, 'b h q k, b h k d -> b h q d')
    """
    A = jnp.ones((3, 4))
    B = jnp.ones((4, 5))
    batch_A = jnp.ones((8, 3, 4))
    batch_B = jnp.ones((8, 4, 5))
    Q = jnp.ones((4, 8, 128, 64))
    K = jnp.ones((4, 8, 128, 64))
    V = jnp.ones((4, 8, 128, 64))
    attn_scores = jnp.ones((4, 8, 128, 128))

    # TODO: Implement this function
    C = None
    batch_C = None
    scores = None
    output = None

    return {
        'matmul_shape': C.shape if C is not None else None,
        'batch_matmul_shape': batch_C.shape if batch_C is not None else None,
        'attention_scores_shape': scores.shape if scores is not None else None,
        'attention_output_shape': output.shape if output is not None else None,
        'bilinear_shape': (8,),  # Provided
    }


# =============================================================================
# Exercise 9: Space to Depth and Depth to Space
# =============================================================================
def exercise_space_depth():
    """
    Space-to-depth downsamples spatially, increases channels.
    Depth-to-space upsamples spatially, decreases channels.

    TODO:
    - space_to_depth_shape: (B, H, W, C) -> (B, H/2, W/2, C*4)
      rearrange(images, 'b (h h2) (w w2) c -> b h w (h2 w2 c)', h2=2, w2=2)
    - depth_to_space_shape: (B, H, W, C*4) -> (B, H*2, W*2, C)
      rearrange(low_res, 'b h w (h2 w2 c) -> b (h h2) (w w2) c', h2=2, w2=2)
    - super_res_shape: Pixel shuffle 4x: (B, H, W, C*16) -> (B, H*4, W*4, C)
    """
    images = jnp.ones((8, 32, 32, 64))
    low_res = jnp.ones((8, 16, 16, 256))
    low_res_sr = jnp.ones((8, 64, 64, 48))  # 48 = 3 * 4^2
    scale_factor = 4

    # TODO: Implement this function
    space_to_depth = None
    depth_to_space = None
    high_res = None

    return {
        'space_to_depth_shape': space_to_depth.shape if space_to_depth is not None else None,
        'depth_to_space_shape': depth_to_space.shape if depth_to_space is not None else None,
        'super_res_shape': high_res.shape if high_res is not None else None,
    }


# =============================================================================
# Exercise 10: Building Blocks for Modern Architectures
# =============================================================================
def exercise_architecture_blocks():
    """
    Common einops patterns used in modern architectures.

    TODO:
    - swin_window_shape: Swin Transformer window partition
      (B, H, W, C) -> (B, num_windows, window_size^2, C)
      rearrange(features, 'b (h wh) (w ww) c -> b (h w) (wh ww) c', wh=7, ww=7)
    - token_mix_shape: MLP-Mixer token transpose
      (B, tokens, channels) -> (B, channels, tokens)
    - nchw_shape: NHWC to NCHW format
    - video_frames_shape: (B, T, H, W, C) -> (B*T, H, W, C)
    - group_norm_shape: Reshape for group norm
      (B, C, H, W) -> (B, G, C//G, H, W)
    """
    features = jnp.ones((4, 56, 56, 96))
    window_size = 7
    tokens = jnp.ones((8, 196, 768))
    features_nhwc = jnp.ones((8, 32, 32, 256))
    video = jnp.ones((4, 16, 224, 224, 3))
    features_gn = jnp.ones((8, 64, 32, 32))
    num_groups = 8

    # TODO: Implement this function
    windowed = None
    for_token_mix = None
    features_nchw = None
    frames = None
    grouped = None

    return {
        'swin_window_shape': windowed.shape if windowed is not None else None,
        'token_mix_shape': for_token_mix.shape if for_token_mix is not None else None,
        'nchw_shape': features_nchw.shape if features_nchw is not None else None,
        'video_frames_shape': frames.shape if frames is not None else None,
        'group_norm_shape': grouped.shape if grouped is not None else None,
    }


# =============================================================================
# Run exercises to test your implementations
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Einops with JAX Exercises")
    print("=" * 60)
    print("\nRun 'pytest test_exercises.py -v' to check your solutions!")
    print("\nOr run individual exercises below:\n")

    exercises = [
        ("1. Basic Rearrange", exercise_basic_rearrange),
        ("2. Split and Merge", exercise_split_merge),
        ("3. Image Patching (ViT)", exercise_image_patching),
        ("4. Attention Reshape", exercise_attention_reshape),
        ("5. Reduce Operations", exercise_reduce),
        ("6. Repeat Operations", exercise_repeat),
        ("7. JAX Transforms", exercise_jax_transforms),
        ("8. Einops Einsum", exercise_einops_einsum),
        ("9. Space/Depth Operations", exercise_space_depth),
        ("10. Architecture Blocks", exercise_architecture_blocks),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if value is None:
                    print(f"  {key}: TODO")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
