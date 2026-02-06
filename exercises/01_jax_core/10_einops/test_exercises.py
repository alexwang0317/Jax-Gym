"""
Tests for Einops with JAX Exercises
===================================

Run with: pytest test_exercises.py -v
"""

import pytest
import jax
import jax.numpy as jnp

from exercises import (
    exercise_basic_rearrange,
    exercise_split_merge,
    exercise_image_patching,
    exercise_attention_reshape,
    exercise_reduce,
    exercise_repeat,
    exercise_jax_transforms,
    exercise_einops_einsum,
    exercise_space_depth,
    exercise_architecture_blocks,
)


class TestBasicRearrange:
    """Exercise 1: Test basic rearrange operations."""

    def test_transpose_shape(self):
        result = exercise_basic_rearrange()
        assert result['transpose_shape'] is not None, \
            "TODO: Use rearrange(image_hwc, 'h w c -> c h w')"
        assert result['transpose_shape'] == (3, 32, 48), \
            f"Expected (3, 32, 48), got {result['transpose_shape']}"

    def test_flatten_shape(self):
        result = exercise_basic_rearrange()
        assert result['flatten_shape'] == (20,), \
            f"Expected (20,), got {result['flatten_shape']}"

    def test_unflatten_shape(self):
        result = exercise_basic_rearrange()
        assert result['unflatten_shape'] == (4, 5), \
            f"Expected (4, 5), got {result['unflatten_shape']}"

    def test_batch_transpose_shape(self):
        result = exercise_basic_rearrange()
        assert result['batch_transpose_shape'] == (16, 3, 32, 32), \
            f"Expected (16, 3, 32, 32), got {result['batch_transpose_shape']}"


class TestSplitMerge:
    """Exercise 2: Test split and merge operations."""

    def test_grid_shape(self):
        result = exercise_split_merge()
        assert result['grid_shape'] is not None, \
            "TODO: Use rearrange(batch, '(rows cols) f -> rows cols f', rows=3, cols=4)"
        assert result['grid_shape'] == (3, 4, 4), \
            f"Expected (3, 4, 4), got {result['grid_shape']}"

    def test_flat_spatial_shape(self):
        result = exercise_split_merge()
        assert result['flat_spatial_shape'] == (8, 1024, 64), \
            f"Expected (8, 1024, 64), got {result['flat_spatial_shape']}"

    def test_grouped_shape(self):
        result = exercise_split_merge()
        assert result['grouped_shape'] == (16, 8, 16), \
            f"Expected (16, 8, 16), got {result['grouped_shape']}"

    def test_merged_shape(self):
        result = exercise_split_merge()
        assert result['merged_shape'] == (400, 256), \
            f"Expected (400, 256), got {result['merged_shape']}"


class TestImagePatching:
    """Exercise 3: Test ViT-style image patching."""

    def test_patches_shape(self):
        result = exercise_image_patching()
        assert result['patches_shape'] is not None, \
            "TODO: Use rearrange with '(h ph) (w pw) c -> (h w) (ph pw c)'"
        # 224/16 = 14, 14*14 = 196 patches, 16*16*3 = 768 dims
        assert result['patches_shape'] == (196, 768), \
            f"Expected (196, 768), got {result['patches_shape']}"

    def test_batch_patches_shape(self):
        result = exercise_image_patching()
        assert result['batch_patches_shape'] == (8, 196, 768), \
            f"Expected (8, 196, 768), got {result['batch_patches_shape']}"

    def test_reconstruction(self):
        result = exercise_image_patching()
        assert result['reconstruction_correct'] == True, \
            "Reconstruction should match original image"


class TestAttentionReshape:
    """Exercise 4: Test multi-head attention reshaping."""

    def test_multi_head_shape(self):
        result = exercise_attention_reshape()
        assert result['multi_head_shape'] is not None, \
            "TODO: Use rearrange(x, 'b t (h d) -> b h t d', h=num_heads)"
        assert result['multi_head_shape'] == (4, 8, 128, 64), \
            f"Expected (4, 8, 128, 64), got {result['multi_head_shape']}"

    def test_merged_shape(self):
        result = exercise_attention_reshape()
        assert result['merged_shape'] == (4, 128, 512), \
            f"Expected (4, 128, 512), got {result['merged_shape']}"

    def test_qkv_split(self):
        result = exercise_attention_reshape()
        assert result['q_shape'] == (4, 8, 128, 64), \
            f"Q shape: expected (4, 8, 128, 64), got {result['q_shape']}"
        assert result['k_shape'] == (4, 8, 128, 64), \
            f"K shape: expected (4, 8, 128, 64), got {result['k_shape']}"
        assert result['v_shape'] == (4, 8, 128, 64), \
            f"V shape: expected (4, 8, 128, 64), got {result['v_shape']}"


class TestReduce:
    """Exercise 5: Test reduce operations."""

    def test_global_pool_shape(self):
        result = exercise_reduce()
        assert result['global_pool_shape'] is not None, \
            "TODO: Use reduce(images, 'b h w c -> b c', 'mean')"
        assert result['global_pool_shape'] == (8, 64), \
            f"Expected (8, 64), got {result['global_pool_shape']}"

    def test_seq_mean_shape(self):
        result = exercise_reduce()
        assert result['seq_mean_shape'] == (2, 4), \
            f"Expected (2, 4), got {result['seq_mean_shape']}"

    def test_max_pool_shape(self):
        result = exercise_reduce()
        assert result['max_pool_shape'] == (8, 16, 16, 64), \
            f"Expected (8, 16, 16, 64), got {result['max_pool_shape']}"

    def test_variance(self):
        result = exercise_reduce()
        if result['variance'] is not None:
            expected_var = jnp.array([2/3, 2/3])
            assert jnp.allclose(result['variance'], expected_var, atol=1e-5), \
                f"Variance should be ~[0.667, 0.667], got {result['variance']}"


class TestRepeat:
    """Exercise 6: Test repeat operations."""

    def test_batched_shape(self):
        result = exercise_repeat()
        assert result['batched_shape'] is not None, \
            "TODO: Use repeat(image, 'h w c -> b h w c', b=8)"
        assert result['batched_shape'] == (8, 32, 32, 3), \
            f"Expected (8, 32, 32, 3), got {result['batched_shape']}"

    def test_tiled_shape(self):
        result = exercise_repeat()
        assert result['tiled_shape'] == (10, 192), \
            f"Expected (10, 192), got {result['tiled_shape']}"

    def test_broadcast_mask_shape(self):
        result = exercise_repeat()
        assert result['broadcast_mask_shape'] == (4, 8, 128, 128), \
            f"Expected (4, 8, 128, 128), got {result['broadcast_mask_shape']}"

    def test_interleaved(self):
        result = exercise_repeat()
        if result['interleaved'] is not None:
            expected = jnp.array([1, 1, 2, 2, 3, 3])
            assert jnp.allclose(result['interleaved'], expected), \
                f"Expected [1,1,2,2,3,3], got {result['interleaved']}"


class TestJaxTransforms:
    """Exercise 7: Test einops with JAX transformations."""

    def test_jit_works(self):
        result = exercise_jax_transforms()
        assert result['jit_result_shape'] is not None, \
            "TODO: JIT a function that uses rearrange"
        assert result['jit_result_shape'] == (4, 8, 128, 64), \
            f"Expected (4, 8, 128, 64), got {result['jit_result_shape']}"

    def test_grad_works(self):
        result = exercise_jax_transforms()
        assert result['grad_shape'] is not None, \
            "TODO: Use grad with a function that uses rearrange"
        assert result['grad_shape'] == (64, 32), \
            f"Expected (64, 32), got {result['grad_shape']}"

    def test_vmap_works(self):
        result = exercise_jax_transforms()
        assert result['vmap_result_shape'] is not None, \
            "TODO: Use vmap with a function that uses rearrange"
        assert result['vmap_result_shape'] == (16, 16), \
            f"Expected (16, 16), got {result['vmap_result_shape']}"

    def test_all_transforms(self):
        result = exercise_jax_transforms()
        assert result['all_transforms_work'] == True, \
            "All JAX transforms should work with einops"


class TestEinopsEinsum:
    """Exercise 8: Test einops einsum."""

    def test_matmul_shape(self):
        result = exercise_einops_einsum()
        assert result['matmul_shape'] is not None, \
            "TODO: Use einsum(A, B, 'i j, j k -> i k')"
        assert result['matmul_shape'] == (3, 5), \
            f"Expected (3, 5), got {result['matmul_shape']}"

    def test_batch_matmul_shape(self):
        result = exercise_einops_einsum()
        assert result['batch_matmul_shape'] == (8, 3, 5), \
            f"Expected (8, 3, 5), got {result['batch_matmul_shape']}"

    def test_attention_scores_shape(self):
        result = exercise_einops_einsum()
        assert result['attention_scores_shape'] == (4, 8, 128, 128), \
            f"Expected (4, 8, 128, 128), got {result['attention_scores_shape']}"

    def test_attention_output_shape(self):
        result = exercise_einops_einsum()
        assert result['attention_output_shape'] == (4, 8, 128, 64), \
            f"Expected (4, 8, 128, 64), got {result['attention_output_shape']}"


class TestSpaceDepth:
    """Exercise 9: Test space-to-depth and depth-to-space."""

    def test_space_to_depth(self):
        result = exercise_space_depth()
        assert result['space_to_depth_shape'] is not None, \
            "TODO: Use rearrange with 'b (h h2) (w w2) c -> b h w (h2 w2 c)'"
        assert result['space_to_depth_shape'] == (8, 16, 16, 256), \
            f"Expected (8, 16, 16, 256), got {result['space_to_depth_shape']}"

    def test_depth_to_space(self):
        result = exercise_space_depth()
        assert result['depth_to_space_shape'] == (8, 32, 32, 64), \
            f"Expected (8, 32, 32, 64), got {result['depth_to_space_shape']}"

    def test_super_res(self):
        result = exercise_space_depth()
        assert result['super_res_shape'] == (8, 256, 256, 3), \
            f"Expected (8, 256, 256, 3), got {result['super_res_shape']}"


class TestArchitectureBlocks:
    """Exercise 10: Test modern architecture patterns."""

    def test_swin_window(self):
        result = exercise_architecture_blocks()
        assert result['swin_window_shape'] is not None, \
            "TODO: Window partition for Swin Transformer"
        # 56/7 = 8, 8*8 = 64 windows, 7*7 = 49 tokens per window
        assert result['swin_window_shape'] == (4, 64, 49, 96), \
            f"Expected (4, 64, 49, 96), got {result['swin_window_shape']}"

    def test_token_mix(self):
        result = exercise_architecture_blocks()
        assert result['token_mix_shape'] == (8, 768, 196), \
            f"Expected (8, 768, 196), got {result['token_mix_shape']}"

    def test_nchw(self):
        result = exercise_architecture_blocks()
        assert result['nchw_shape'] == (8, 256, 32, 32), \
            f"Expected (8, 256, 32, 32), got {result['nchw_shape']}"

    def test_video_frames(self):
        result = exercise_architecture_blocks()
        assert result['video_frames_shape'] == (64, 224, 224, 3), \
            f"Expected (64, 224, 224, 3), got {result['video_frames_shape']}"

    def test_group_norm(self):
        result = exercise_architecture_blocks()
        assert result['group_norm_shape'] == (8, 8, 8, 32, 32), \
            f"Expected (8, 8, 8, 32, 32), got {result['group_norm_shape']}"


class TestEinopsEdgeCases:
    """Additional edge case tests."""

    def test_identity_rearrange(self):
        """Rearranging to same pattern should be identity."""
        from einops import rearrange
        x = jnp.ones((4, 8, 16))
        y = rearrange(x, 'a b c -> a b c')
        assert jnp.allclose(x, y)

    def test_single_element(self):
        """Should work with single-element tensors."""
        from einops import rearrange
        x = jnp.array([[[1.0]]])
        y = rearrange(x, 'a b c -> (a b c)')
        assert y.shape == (1,)

    def test_reduce_to_scalar(self):
        """Reduce to scalar."""
        from einops import reduce
        x = jnp.arange(12).reshape(3, 4).astype(float)
        scalar = reduce(x, 'h w -> ', 'mean')
        assert scalar.shape == ()
        assert jnp.allclose(scalar, 5.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
