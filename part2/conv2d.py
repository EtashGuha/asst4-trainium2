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



    for batch_idx in nl.affine_range(batch_size):
        output_tile = nl.zeros((out_channels, out_height * out_width), dtype=X.dtype, buffer=nl.sbuf)
        for fh in range(filter_height):
            for fw in range(filter_width):
                # Build input matrix for this filter position
                # Shape: (in_channels, out_height * out_width)
                x_tile = nl.ndarray((in_channels, out_height * out_width), dtype=X.dtype, buffer=nl.sbuf)
                for i in range(out_height * out_width):
                    oh = i // out_width
                    ow = i % out_width
                    nisa.dma_copy(src=X[batch_idx, :, oh + fh, ow + fw], dst=x_tile[:, i])
                w_tile = nl.ndarray((out_channels, in_channels), dtype=W.dtype, buffer=nl.sbuf)
                nisa.dma_copy(src=W[:, :, fh, fw], dst=w_tile)
                # Transpose weights: nc_matmul computes stationary.T @ moving
                w_transposed = nl.copy(nisa.nc_transpose(w_tile), dtype=W.dtype)
                out = nl.copy(nisa.nc_matmul(w_transposed, x_tile), dtype=X.dtype)
                output_tile[:, :] = nisa.tensor_tensor(output_tile, out, nl.add)
        # Add bias to all channels at once
        bias_tile = nl.ndarray((out_channels, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=bias, dst=bias_tile[:, 0])
        # Broadcast bias across all output positions
        for i in range(out_height * out_width):
            output_tile[:, i] = nisa.tensor_tensor(output_tile[:, i], bias_tile[:, 0], nl.add)
        output_tile = output_tile.reshape((out_channels, out_height, out_width))
        nisa.dma_copy(src=output_tile, dst=X_out[batch_idx, :, :, :])

    return X_out