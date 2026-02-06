"""
Tests for Einops with JAX Examples
==================================
"""

import pytest
import jax
import jax.numpy as jnp

from examples import (
    example_basic_rearrange,
    example_split_merge,
    example_image_patching,
    example_attention_reshape,
    example_reduce,
    example_repeat,
    example_jax_transforms,
    example_einops_einsum,
    example_space_depth,
    example_architecture_blocks,
)


class TestBasicRearrange:
    def test_transpose_shape(self):
        result = example_basic_rearrange()
        assert result['transpose_shape'] == (3, 32, 48)

    def test_flatten_shape(self):
        result = example_basic_rearrange()
        assert result['flatten_shape'] == (20,)

    def test_unflatten_shape(self):
        result = example_basic_rearrange()
        assert result['unflatten_shape'] == (4, 5)

    def test_batch_transpose_shape(self):
        result = example_basic_rearrange()
        assert result['batch_transpose_shape'] == (16, 3, 32, 32)


class TestSplitMerge:
    def test_grid_shape(self):
        result = example_split_merge()
        assert result['grid_shape'] == (3, 4, 4)

    def test_flat_spatial_shape(self):
        result = example_split_merge()
        assert result['flat_spatial_shape'] == (8, 1024, 64)

    def test_grouped_shape(self):
        result = example_split_merge()
        assert result['grouped_shape'] == (16, 8, 16)

    def test_merged_shape(self):
        result = example_split_merge()
        assert result['merged_shape'] == (400, 256)


class TestImagePatching:
    def test_patches_shape(self):
        result = example_image_patching()
        # 224/16 = 14, 14*14 = 196 patches, 16*16*3 = 768 dims
        assert result['patches_shape'] == (196, 768)

    def test_batch_patches_shape(self):
        result = example_image_patching()
        assert result['batch_patches_shape'] == (8, 196, 768)

    def test_reconstruction(self):
        result = example_image_patching()
        assert result['reconstruction_correct'] == True


class TestAttentionReshape:
    def test_multi_head_shape(self):
        result = example_attention_reshape()
        assert result['multi_head_shape'] == (4, 8, 128, 64)

    def test_merged_shape(self):
        result = example_attention_reshape()
        assert result['merged_shape'] == (4, 128, 512)

    def test_qkv_split(self):
        result = example_attention_reshape()
        assert result['q_shape'] == (4, 8, 128, 64)
        assert result['k_shape'] == (4, 8, 128, 64)
        assert result['v_shape'] == (4, 8, 128, 64)


class TestReduce:
    def test_global_pool_shape(self):
        result = example_reduce()
        assert result['global_pool_shape'] == (8, 64)

    def test_seq_mean_shape(self):
        result = example_reduce()
        assert result['seq_mean_shape'] == (2, 4)

    def test_max_pool_shape(self):
        result = example_reduce()
        assert result['max_pool_shape'] == (8, 16, 16, 64)

    def test_variance(self):
        result = example_reduce()
        expected_var = jnp.array([2/3, 2/3])
        assert jnp.allclose(result['variance'], expected_var, atol=1e-5)


class TestRepeat:
    def test_batched_shape(self):
        result = example_repeat()
        assert result['batched_shape'] == (8, 32, 32, 3)

    def test_tiled_shape(self):
        result = example_repeat()
        assert result['tiled_shape'] == (10, 192)

    def test_broadcast_mask_shape(self):
        result = example_repeat()
        assert result['broadcast_mask_shape'] == (4, 8, 128, 128)

    def test_interleaved(self):
        result = example_repeat()
        expected = jnp.array([1, 1, 2, 2, 3, 3])
        assert jnp.allclose(result['interleaved'], expected)


class TestJaxTransforms:
    def test_jit_works(self):
        result = example_jax_transforms()
        assert result['jit_result_shape'] == (4, 8, 128, 64)

    def test_grad_works(self):
        result = example_jax_transforms()
        assert result['grad_shape'] == (64, 32)

    def test_vmap_works(self):
        result = example_jax_transforms()
        assert result['vmap_result_shape'] == (16, 16)

    def test_all_transforms(self):
        result = example_jax_transforms()
        assert result['all_transforms_work'] == True


class TestEinopsEinsum:
    def test_matmul_shape(self):
        result = example_einops_einsum()
        assert result['matmul_shape'] == (3, 5)

    def test_batch_matmul_shape(self):
        result = example_einops_einsum()
        assert result['batch_matmul_shape'] == (8, 3, 5)

    def test_attention_scores_shape(self):
        result = example_einops_einsum()
        assert result['attention_scores_shape'] == (4, 8, 128, 128)

    def test_attention_output_shape(self):
        result = example_einops_einsum()
        assert result['attention_output_shape'] == (4, 8, 128, 64)


class TestSpaceDepth:
    def test_space_to_depth(self):
        result = example_space_depth()
        assert result['space_to_depth_shape'] == (8, 16, 16, 256)

    def test_depth_to_space(self):
        result = example_space_depth()
        assert result['depth_to_space_shape'] == (8, 32, 32, 64)

    def test_super_res(self):
        result = example_space_depth()
        assert result['super_res_shape'] == (8, 256, 256, 3)


class TestArchitectureBlocks:
    def test_swin_window(self):
        result = example_architecture_blocks()
        # 56/7 = 8, 8*8 = 64 windows, 7*7 = 49 tokens per window
        assert result['swin_window_shape'] == (4, 64, 49, 96)

    def test_token_mix(self):
        result = example_architecture_blocks()
        assert result['token_mix_shape'] == (8, 768, 196)

    def test_nchw(self):
        result = example_architecture_blocks()
        assert result['nchw_shape'] == (8, 256, 32, 32)

    def test_video_frames(self):
        result = example_architecture_blocks()
        assert result['video_frames_shape'] == (64, 224, 224, 3)

    def test_group_norm(self):
        result = example_architecture_blocks()
        assert result['group_norm_shape'] == (8, 8, 8, 32, 32)


class TestEinopsEdgeCases:
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
