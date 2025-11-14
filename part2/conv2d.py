import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from tqdm import tqdm

"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
def nki_matmul_tiled_(lhsT, rhs, result):
    """NKI helper to compute a matrix multiplication operation in a tiled manner"""

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

    # Maximum partition dimension of a tile
    TILE_K = nl.tile_size.pmax  # 128

    # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Use affine_range to loop over tiles
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Declare the tiles on SBUF
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                # Load tiles from lhsT and rhs
                nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

                # Accumulate partial-sums into PSUM
                res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)

        # Handle remainder in N dimension if N is not divisible by TILE_N
        n_remainder = N % TILE_N
        if n_remainder > 0:
            n_tiles = N // TILE_N
            # Allocate full-sized tile
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                # Only copy the remainder portion
                nisa.dma_copy(dst=rhs_tile[:, :n_remainder], src=rhs[k * TILE_K:(k + 1) * TILE_K, n_tiles * TILE_N:N])

                res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

            res_sb = nl.copy(res_psum, dtype=result.dtype)
            # Only copy back the remainder portion
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n_tiles * TILE_N:N], src=res_sb[:, :n_remainder])

@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    H_out = 1 + (input_height - filter_height)
    W_out = 1 + (input_width - filter_width)

    # Process the images in batches
    # for b in range(batch_size):
    #     for c in range(out_channels):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 # Step 1: Allocate tiles
    #                 x_tile = nl.ndarray((in_channels, filter_height, filter_width), dtype=X.dtype, buffer=nl.sbuf)
    #                 w_tile = nl.ndarray((in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)

    #                 # Step 2: Load data
    #                 nisa.dma_copy(src=X[b, :, i * pool_size : i * pool_size + filter_height, j * pool_size : j * pool_size + filter_width], dst=x_tile)
    #                 nisa.dma_copy(src=W[c], dst=w_tile)

    #                 out_tile = x_tile * w_tile

    #                 temp = nisa.tensor_reduce(nl.add, out_tile, axis=(1, 2), keepdims=True)

    #                 temp_flat = temp[:, 0, 0:1]

    #                 ones = nl.ones((in_channels, 1), dtype=temp_flat.dtype, buffer=nl.sbuf)
    #                 result_psum = nisa.nc_matmul(temp_flat, ones)
    #                 result = nl.copy(result_psum, dtype=X.dtype)

    #                 # Add bias
    #                 bias_tile = nl.ndarray((1, 1), dtype=bias.dtype, buffer=nl.sbuf)
    #                 nisa.dma_copy(src=bias[c:c+1], dst=bias_tile)
    #                 result = nisa.tensor_scalar(result, nl.add, bias_tile)
    #                 nisa.dma_copy(src=result, dst=X_out[b, c, i, j])


    # (col_offset, row_offset)

    # reshaped_X = X.reshape((batch_size, in_channels, input_height * input_width))
    # output = nl.ndarray((batch_size, out_channels, out_pool_height, out_pool_width), dtype=X.dtype, buffer=nl.hbm)
    # for batch_idx in nl.affine_range(batch_size):
    #     output_tile = nl.ndarray((out_channels, out_pool_height * out_pool_width), dtype=X.dtype, buffer=nl.sbuf)
    #     for row_offset in range(filter_height):
    #         for col_offset in range(filter_width):
    #             total_offset = row_offset * input_width + col_offset
    #             x_tile = nl.ndarray((in_channels, input_width *input_height), dtype=X.dtype, buffer=nl.sbuf)
    #             nisa.dma_copy(src=reshaped_X[batch_idx, :, :], dst=x_tile)
    #             w_tile = nl.ndarray((in_channels, out_channels), dtype=W.dtype, buffer=nl.sbuf)
    #             nisa.dma_copy(src=W[:,:, col_offset, row_offset], dst=w_tile)
    #             out = nisa.nc_matmul(w_tile, x_tile)

    #             output_tile += out[total_offset:]
    #     nisa.dma_copy(src=output_tile, dst=output[batch_idx, :, :, :])

    PARTITION_SIZE = 128
    SPATIAL_PARTITION_SIZE = 512

    for batch_idx in nl.affine_range(batch_size):
        for oc_idx in nl.affine_range(out_channels // PARTITION_SIZE):
            # Load ALL weights for this output channel ONCE
            w_all = nl.ndarray((PARTITION_SIZE, in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=W[oc_idx * PARTITION_SIZE: (oc_idx + 1) * PARTITION_SIZE, :, :, :], dst=w_all)

            # Load bias once per output channel tile
            bias_tile = nl.ndarray((PARTITION_SIZE, 1), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=bias[oc_idx * PARTITION_SIZE: (oc_idx + 1) * PARTITION_SIZE], dst=bias_tile[:, 0])

            num_spatial_tiles = (out_height * out_width + SPATIAL_PARTITION_SIZE - 1) // SPATIAL_PARTITION_SIZE
            for spatial_partition_idx in nl.affine_range(num_spatial_tiles):
                spatial_start = spatial_partition_idx * SPATIAL_PARTITION_SIZE
                spatial_end = min(spatial_start + SPATIAL_PARTITION_SIZE, out_height * out_width)
                spatial_size = spatial_end - spatial_start

                # Allocate PSUM accumulator for this spatial tile
                output_psum = nl.zeros((PARTITION_SIZE, SPATIAL_PARTITION_SIZE), nl.float32, buffer=nl.psum)

                # Accumulate contributions from all input channels and filter positions
                for ic_idx in range(in_channels // PARTITION_SIZE):
                    # Load entire input plane for this ic_idx ONCE
                    x_all = nl.ndarray((PARTITION_SIZE, input_height, input_width), dtype=X.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=X[batch_idx, ic_idx * PARTITION_SIZE: (ic_idx + 1) * PARTITION_SIZE, :, :], dst=x_all)

                    # Extract weights for this ic_idx from pre-loaded buffer (SBUF -> SBUF)
                    w_ic = nl.ndarray((PARTITION_SIZE, PARTITION_SIZE, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    w_ic[:, :, :, :] = w_all[:, ic_idx * PARTITION_SIZE: (ic_idx + 1) * PARTITION_SIZE, :, :]

                    for fh in range(filter_height):
                        for fw in range(filter_width):
                            # Extract weight tile from SBUF (no DMA)
                            w_tile = w_ic[:, :, fh, fw]
                            w_transposed = nl.copy(nisa.nc_transpose(w_tile), dtype=W.dtype)

                            # Extract the plane for this filter position from pre-loaded X (SBUF -> SBUF)
                            x_plane = nl.ndarray((PARTITION_SIZE, out_height, out_width), dtype=X.dtype, buffer=nl.sbuf)
                            x_plane[:, :, :] = x_all[:, fh:fh + out_height, fw:fw + out_width]

                            # Flatten and extract spatial tile
                            x_plane_flat = x_plane.reshape((PARTITION_SIZE, out_height * out_width))
                            x_tile = nl.ndarray((PARTITION_SIZE, SPATIAL_PARTITION_SIZE), dtype=X.dtype, buffer=nl.sbuf)
                            for i in nl.affine_range(SPATIAL_PARTITION_SIZE):
                                if i < spatial_size:
                                    x_tile[:, i] = x_plane_flat[:, spatial_start + i]

                            # Accumulate in PSUM - NO intermediate conversion
                            output_psum += nisa.nc_matmul(w_transposed, x_tile)

                # Convert from PSUM to SBUF once at the end
                output_sbuf = nl.copy(output_psum, dtype=X.dtype)

                # Add bias
                for i in nl.affine_range(SPATIAL_PARTITION_SIZE):
                    if i < spatial_size:
                        output_sbuf[:, i] = nisa.tensor_tensor(output_sbuf[:, i], bias_tile[:, 0], nl.add)

                # Write back element-by-element
                for i in nl.affine_range(SPATIAL_PARTITION_SIZE):
                    if i < spatial_size:
                        global_i = spatial_start + i
                        oh = global_i // out_width
                        ow = global_i % out_width
                        nisa.dma_copy(src=output_sbuf[:, i], dst=X_out[batch_idx, oc_idx * PARTITION_SIZE: (oc_idx + 1) * PARTITION_SIZE, oh, ow])

    return X_out