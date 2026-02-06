"""
Einops with JAX - 10 Examples
=============================

Einops provides readable, flexible tensor operations using Einstein notation.
Works seamlessly with JAX arrays and is fully compatible with jit/grad/vmap.

Key operations:
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
# Example 1: Basic Rearrange - Transpose and Reshape
# =============================================================================
def example_basic_rearrange():
    """
    rearrange is the core operation - combines reshape, transpose, squeeze, etc.
    Pattern: 'input_axes -> output_axes'
    """
    # Simple transpose: (H, W, C) -> (C, H, W)
    image_hwc = jnp.ones((32, 48, 3))
    image_chw = rearrange(image_hwc, 'h w c -> c h w')

    # Flatten: (H, W) -> (H*W,)
    matrix = jnp.ones((4, 5))
    flat = rearrange(matrix, 'h w -> (h w)')

    # Unflatten: (H*W,) -> (H, W) with explicit sizes
    unflat = rearrange(flat, '(h w) -> h w', h=4, w=5)

    # Batch transpose: (B, H, W, C) -> (B, C, H, W)
    batch_images = jnp.ones((16, 32, 32, 3))
    batch_chw = rearrange(batch_images, 'b h w c -> b c h w')

    return {
        'transpose_shape': image_chw.shape,  # (3, 32, 48)
        'flatten_shape': flat.shape,  # (20,)
        'unflatten_shape': unflat.shape,  # (4, 5)
        'batch_transpose_shape': batch_chw.shape,  # (16, 3, 32, 32)
    }


# =============================================================================
# Example 2: Splitting and Merging Axes
# =============================================================================
def example_split_merge():
    """
    Split one axis into multiple, or merge multiple into one.
    Use parentheses to group axes.
    """
    # Split batch into grid: (B,) -> (rows, cols)
    batch = jnp.arange(12).reshape(12, 4)  # 12 samples, 4 features
    grid = rearrange(batch, '(rows cols) f -> rows cols f', rows=3, cols=4)

    # Merge spatial dims: (B, H, W, C) -> (B, H*W, C)
    images = jnp.ones((8, 32, 32, 64))
    flat_spatial = rearrange(images, 'b h w c -> b (h w) c')

    # Split channels into groups: (B, C) -> (B, groups, C//groups)
    features = jnp.ones((16, 128))
    grouped = rearrange(features, 'b (g c) -> b g c', g=8)

    # Merge batch and sequence: (B, T, D) -> (B*T, D)
    sequences = jnp.ones((4, 100, 256))
    merged = rearrange(sequences, 'b t d -> (b t) d')

    return {
        'grid_shape': grid.shape,  # (3, 4, 4)
        'flat_spatial_shape': flat_spatial.shape,  # (8, 1024, 64)
        'grouped_shape': grouped.shape,  # (16, 8, 16)
        'merged_shape': merged.shape,  # (400, 256)
    }


# =============================================================================
# Example 3: Image Patching (Vision Transformer Style)
# =============================================================================
def example_image_patching():
    """
    Extract patches from images - essential for Vision Transformers.
    Split H and W into (num_patches, patch_size).
    """
    # Single image: (H, W, C) -> (num_patches, patch_dim)
    image = jnp.ones((224, 224, 3))
    patch_size = 16

    # Extract patches and flatten each patch
    patches = rearrange(
        image,
        '(h ph) (w pw) c -> (h w) (ph pw c)',
        ph=patch_size, pw=patch_size
    )
    # h = 224/16 = 14, w = 14, so 196 patches of size 16*16*3 = 768

    # Batch version: (B, H, W, C) -> (B, num_patches, patch_dim)
    batch_images = jnp.ones((8, 224, 224, 3))
    batch_patches = rearrange(
        batch_images,
        'b (h ph) (w pw) c -> b (h w) (ph pw c)',
        ph=patch_size, pw=patch_size
    )

    # Reverse: patches back to image
    reconstructed = rearrange(
        patches,
        '(h w) (ph pw c) -> (h ph) (w pw) c',
        h=14, w=14, ph=16, pw=16, c=3
    )

    return {
        'patches_shape': patches.shape,  # (196, 768)
        'batch_patches_shape': batch_patches.shape,  # (8, 196, 768)
        'reconstructed_shape': reconstructed.shape,  # (224, 224, 3)
        'reconstruction_correct': jnp.allclose(image, reconstructed),
    }


# =============================================================================
# Example 4: Attention Reshaping (Multi-Head Attention)
# =============================================================================
def example_attention_reshape():
    """
    Reshape for multi-head attention: split embed_dim into heads.
    Pattern: (B, T, D) -> (B, heads, T, head_dim)
    """
    batch_size, seq_len, embed_dim = 4, 128, 512
    num_heads = 8
    head_dim = embed_dim // num_heads  # 64

    # Input: (B, T, D)
    x = jnp.ones((batch_size, seq_len, embed_dim))

    # Split into heads: (B, T, D) -> (B, heads, T, head_dim)
    multi_head = rearrange(x, 'b t (h d) -> b h t d', h=num_heads)

    # After attention, merge heads back: (B, heads, T, head_dim) -> (B, T, D)
    merged = rearrange(multi_head, 'b h t d -> b t (h d)')

    # For QKV projection: (B, T, 3*D) -> 3 x (B, heads, T, head_dim)
    qkv = jnp.ones((batch_size, seq_len, 3 * embed_dim))
    q, k, v = rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=num_heads)

    return {
        'multi_head_shape': multi_head.shape,  # (4, 8, 128, 64)
        'merged_shape': merged.shape,  # (4, 128, 512)
        'q_shape': q.shape,  # (4, 8, 128, 64)
        'k_shape': k.shape,  # (4, 8, 128, 64)
        'v_shape': v.shape,  # (4, 8, 128, 64)
    }


# =============================================================================
# Example 5: Reduce Operations
# =============================================================================
def example_reduce():
    """
    reduce combines reduction with rearrangement.
    Specify which axes to reduce and how (mean, sum, max, min, prod).
    """
    # Global average pooling: (B, H, W, C) -> (B, C)
    images = jnp.ones((8, 32, 32, 64)) * jnp.arange(64)
    pooled = reduce(images, 'b h w c -> b c', 'mean')

    # Reduce specific axis: mean over sequence
    sequences = jnp.arange(24).reshape(2, 3, 4).astype(float)  # (B, T, D)
    seq_mean = reduce(sequences, 'b t d -> b d', 'mean')

    # Max pooling with stride: (B, H, W, C) -> (B, H/2, W/2, C)
    max_pooled = reduce(images, 'b (h h2) (w w2) c -> b h w c', 'max', h2=2, w2=2)

    # Sum over groups
    grouped = jnp.ones((16, 8, 4))  # (B, groups, features)
    group_sum = reduce(grouped, 'b g f -> b f', 'sum')

    # Variance via reduce (using mean of squares - square of mean)
    data = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    mean_sq = reduce(data ** 2, 'b d -> b', 'mean')
    sq_mean = reduce(data, 'b d -> b', 'mean') ** 2
    variance = mean_sq - sq_mean

    return {
        'global_pool_shape': pooled.shape,  # (8, 64)
        'seq_mean_shape': seq_mean.shape,  # (2, 4)
        'max_pool_shape': max_pooled.shape,  # (8, 16, 16, 64)
        'group_sum_shape': group_sum.shape,  # (16, 4)
        'variance': variance,  # [0.667, 0.667]
    }


# =============================================================================
# Example 6: Repeat Operations
# =============================================================================
def example_repeat():
    """
    repeat tiles/broadcasts arrays along new or existing axes.
    """
    # Add batch dimension: (H, W, C) -> (B, H, W, C)
    image = jnp.ones((32, 32, 3))
    batched = repeat(image, 'h w c -> b h w c', b=8)

    # Tile along existing axis: (T, D) -> (T, D*3)
    seq = jnp.ones((10, 64))
    tiled = repeat(seq, 't d -> t (repeat d)', repeat=3)

    # Broadcast for attention: (B, T) -> (B, heads, T, T)
    # Useful for attention masks
    mask = jnp.ones((4, 128))
    broadcast_mask = repeat(mask, 'b t -> b h t t2', h=8, t2=128)

    # Create positional encoding base: (D,) -> (T, D)
    dim = 64
    pos_base = jnp.arange(dim)
    pos_expanded = repeat(pos_base, 'd -> t d', t=100)

    # Repeat interleaved: [1,2,3] -> [1,1,2,2,3,3]
    arr = jnp.array([1, 2, 3])
    interleaved = repeat(arr, 'n -> (n repeat)', repeat=2)

    return {
        'batched_shape': batched.shape,  # (8, 32, 32, 3)
        'tiled_shape': tiled.shape,  # (10, 192)
        'broadcast_mask_shape': broadcast_mask.shape,  # (4, 8, 128, 128)
        'pos_expanded_shape': pos_expanded.shape,  # (100, 64)
        'interleaved': interleaved,  # [1, 1, 2, 2, 3, 3]
    }


# =============================================================================
# Example 7: Einops with JAX Transformations
# =============================================================================
def example_jax_transforms():
    """
    Einops works seamlessly with jit, grad, and vmap.
    """
    # JIT compilation
    @jax.jit
    def attention_reshape_jit(x, num_heads):
        return rearrange(x, 'b t (h d) -> b h t d', h=num_heads)

    x = jnp.ones((4, 128, 512))
    result_jit = attention_reshape_jit(x, 8)

    # Works with grad
    def loss_with_rearrange(params, x):
        # Reshape, apply params, reduce
        reshaped = rearrange(x, 'b h w c -> b (h w) c')
        output = reshaped @ params
        return reduce(output, 'b t d -> ', 'mean')

    params = jnp.ones((64, 32))
    x_img = jnp.ones((2, 8, 8, 64))
    grads = jax.grad(loss_with_rearrange)(params, x_img)

    # Works with vmap
    def process_single(img):
        patches = rearrange(img, '(h ph) (w pw) c -> (h w) (ph pw c)', ph=8, pw=8)
        return reduce(patches, 'n d -> n', 'mean')

    batch_images = jnp.ones((16, 32, 32, 3))
    vmap_result = jax.vmap(process_single)(batch_images)

    return {
        'jit_result_shape': result_jit.shape,
        'grad_shape': grads.shape,
        'vmap_result_shape': vmap_result.shape,
        'all_transforms_work': True,
    }


# =============================================================================
# Example 8: Einsum with Einops
# =============================================================================
def example_einops_einsum():
    """
    einops.einsum is a more readable wrapper around einsum.
    Uses named dimensions instead of single letters.
    """
    # Matrix multiplication with named dims
    A = jnp.ones((3, 4))
    B = jnp.ones((4, 5))
    C = einsum(A, B, 'i j, j k -> i k')

    # Batch matrix multiply
    batch_A = jnp.ones((8, 3, 4))
    batch_B = jnp.ones((8, 4, 5))
    batch_C = einsum(batch_A, batch_B, 'b i j, b j k -> b i k')

    # Attention scores: Q @ K^T
    Q = jnp.ones((4, 8, 128, 64))  # (batch, heads, seq, dim)
    K = jnp.ones((4, 8, 128, 64))
    scores = einsum(Q, K, 'b h q d, b h k d -> b h q k')

    # Dot product attention output: scores @ V
    V = jnp.ones((4, 8, 128, 64))
    attn_scores = jnp.ones((4, 8, 128, 128))  # after softmax
    output = einsum(attn_scores, V, 'b h q k, b h k d -> b h q d')

    # Bilinear form: x^T A y
    x = jnp.ones((8, 64))
    y = jnp.ones((8, 64))
    A_mat = jnp.ones((64, 64))
    bilinear = einsum(x, A_mat, y, 'b i, i j, b j -> b')

    return {
        'matmul_shape': C.shape,  # (3, 5)
        'batch_matmul_shape': batch_C.shape,  # (8, 3, 5)
        'attention_scores_shape': scores.shape,  # (4, 8, 128, 128)
        'attention_output_shape': output.shape,  # (4, 8, 128, 64)
        'bilinear_shape': bilinear.shape,  # (8,)
    }


# =============================================================================
# Example 9: Common Patterns - Space to Depth and Depth to Space
# =============================================================================
def example_space_depth():
    """
    Space-to-depth and depth-to-space operations.
    Useful for efficient downsampling/upsampling without pooling.
    """
    # Space to depth: (B, H, W, C) -> (B, H/2, W/2, C*4)
    # Downsamples spatially, increases channels
    images = jnp.ones((8, 32, 32, 64))
    space_to_depth = rearrange(
        images,
        'b (h h2) (w w2) c -> b h w (h2 w2 c)',
        h2=2, w2=2
    )

    # Depth to space: (B, H, W, C*4) -> (B, H*2, W*2, C)
    # Upsamples spatially, decreases channels
    low_res = jnp.ones((8, 16, 16, 256))
    depth_to_space = rearrange(
        low_res,
        'b h w (h2 w2 c) -> b (h h2) (w w2) c',
        h2=2, w2=2
    )

    # Pixel shuffle for super-resolution: (B, H, W, C*r^2) -> (B, H*r, W*r, C)
    scale_factor = 4
    low_res_sr = jnp.ones((8, 64, 64, 48))  # 48 = 3 * 4^2
    high_res = rearrange(
        low_res_sr,
        'b h w (h2 w2 c) -> b (h h2) (w w2) c',
        h2=scale_factor, w2=scale_factor
    )

    return {
        'space_to_depth_shape': space_to_depth.shape,  # (8, 16, 16, 256)
        'depth_to_space_shape': depth_to_space.shape,  # (8, 32, 32, 64)
        'super_res_shape': high_res.shape,  # (8, 256, 256, 3)
    }


# =============================================================================
# Example 10: Building Blocks for Modern Architectures
# =============================================================================
def example_architecture_blocks():
    """
    Common einops patterns used in modern architectures.
    """
    # 1. Swin Transformer: Window partition
    # (B, H, W, C) -> (B, num_windows, window_size^2, C)
    features = jnp.ones((4, 56, 56, 96))
    window_size = 7
    windowed = rearrange(
        features,
        'b (h wh) (w ww) c -> b (h w) (wh ww) c',
        wh=window_size, ww=window_size
    )

    # 2. MLP-Mixer: Token mixing
    # Transpose to mix across tokens: (B, tokens, channels) -> (B, channels, tokens)
    tokens = jnp.ones((8, 196, 768))
    for_token_mix = rearrange(tokens, 'b t c -> b c t')

    # 3. ConvNeXt: Depthwise conv reshape
    # (B, H, W, C) -> (B, C, H, W) for PyTorch-style conv
    features_nhwc = jnp.ones((8, 32, 32, 256))
    features_nchw = rearrange(features_nhwc, 'b h w c -> b c h w')

    # 4. 3D video: (B, T, H, W, C) -> (B*T, H, W, C) for 2D processing
    video = jnp.ones((4, 16, 224, 224, 3))
    frames = rearrange(video, 'b t h w c -> (b t) h w c')

    # Reconstruct: (B*T, H, W, C) -> (B, T, H, W, C)
    reconstructed = rearrange(frames, '(b t) h w c -> b t h w c', b=4, t=16)

    # 5. Group normalization reshape: (B, C, H, W) -> (B, G, C//G, H, W)
    features_gn = jnp.ones((8, 64, 32, 32))
    num_groups = 8
    grouped = rearrange(features_gn, 'b (g c) h w -> b g c h w', g=num_groups)

    return {
        'swin_window_shape': windowed.shape,  # (4, 64, 49, 96)
        'token_mix_shape': for_token_mix.shape,  # (8, 768, 196)
        'nchw_shape': features_nchw.shape,  # (8, 256, 32, 32)
        'video_frames_shape': frames.shape,  # (64, 224, 224, 3)
        'group_norm_shape': grouped.shape,  # (8, 8, 8, 32, 32)
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Einops with JAX Examples")
    print("=" * 60)

    examples = [
        ("1. Basic Rearrange", example_basic_rearrange),
        ("2. Split and Merge", example_split_merge),
        ("3. Image Patching (ViT)", example_image_patching),
        ("4. Attention Reshape", example_attention_reshape),
        ("5. Reduce Operations", example_reduce),
        ("6. Repeat Operations", example_repeat),
        ("7. JAX Transforms", example_jax_transforms),
        ("8. Einops Einsum", example_einops_einsum),
        ("9. Space/Depth Operations", example_space_depth),
        ("10. Architecture Blocks", example_architecture_blocks),
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
