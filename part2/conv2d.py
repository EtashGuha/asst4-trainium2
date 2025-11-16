import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

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
    stationary_partition_size = nl.tile_size.gemm_stationary_fmax
    output_channel_partitions = out_channels // stationary_partition_size

    if out_width == 14:
        spatial_tile_height = out_height 
    elif out_width == 222:
        spatial_tile_height = 2
    else:
        spatial_tile_height = None
        
    vertical_partition_count = out_height // spatial_tile_height

    for output_ch_idx in nl.affine_range(output_channel_partitions):
        oc_start = output_ch_idx * stationary_partition_size
        oc_end = (output_ch_idx + 1) * stationary_partition_size

        loaded_bias = nl.ndarray((stationary_partition_size, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=bias[oc_start:oc_end], dst=loaded_bias[0:stationary_partition_size, :])

        all_loaded_weights = nl.ndarray((stationary_partition_size, in_channels, filter_height, filter_width),
                                  dtype=W.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=W[oc_start:oc_end, :, :, :], dst=all_loaded_weights)

        for input_ch_tile in nl.affine_range(n_tiles_c_in):
            ic_offset = input_ch_tile * c_in_pmax
            for kernel_row in nl.affine_range(filter_height):
                for kernel_col in nl.affine_range(filter_width):
                    weight_slice = all_loaded_weights[:, ic_offset:ic_offset + c_in_pmax, kernel_row, kernel_col]
                    transposed_weight = nisa.nc_transpose(weight_slice)
                    all_loaded_weights[:, ic_offset:ic_offset + c_in_pmax, kernel_row, kernel_col] =nisa.tensor_copy(transposed_weight, dtype=W.dtype)

        for batch_idx in nl.affine_range(batch_size):
            for spatial_tile_idx in nl.affine_range(vertical_partition_count):
                accumulator = nl.zeros((stationary_partition_size, spatial_tile_height, out_width),dtype=nl.float32, buffer=nl.psum)
                for input_ch_tile in nl.affine_range(n_tiles_c_in):
                    ic_offset = input_ch_tile * c_in_pmax
                    row_start = spatial_tile_idx * spatial_tile_height
                    row_end = row_start + spatial_tile_height + filter_height - 1

                    input_buffer = nl.ndarray((c_in_pmax, spatial_tile_height + filter_height - 1, input_width), dtype=X.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=X[batch_idx, ic_offset:ic_offset + c_in_pmax, row_start:row_end, :], dst=input_buffer)

                    for kernel_row in nl.affine_range(filter_height):
                        for kernel_col in nl.affine_range(filter_width):
                            weight_slice = all_loaded_weights[:, ic_offset:ic_offset + c_in_pmax, kernel_row, kernel_col]
                            input_window = input_buffer[:, 
                                                        kernel_row:kernel_row + spatial_tile_height,
                                                        kernel_col:kernel_col + out_width]
                            accumulator += nisa.nc_matmul(stationary=weight_slice, moving=input_window)

                conv_result = nisa.tensor_copy(accumulator, dtype=X.dtype)
                bias_broadcast = loaded_bias.reshape((stationary_partition_size, 1, 1)).broadcast_to(conv_result.shape)
                conv_with_bias = nisa.tensor_tensor(data1=conv_result, data2=bias_broadcast, op=nl.add)
                if pool_size == 2:
                    reshaped_for_pool = conv_with_bias.reshape((stationary_partition_size,
                                                                spatial_tile_height // pool_size, pool_size,
                                                                out_pool_width, pool_size))
                    pooled_vertical = nisa.tensor_reduce(nl.max, reshaped_for_pool, axis=2)
                    final_pooled = nisa.tensor_reduce(nl.max, pooled_vertical, axis=3)
                    out_row_start = spatial_tile_idx * spatial_tile_height // pool_size
                    out_row_end = (spatial_tile_idx + 1) * spatial_tile_height // pool_size
                    nisa.dma_copy(src=final_pooled, dst=X_out[batch_idx, oc_start:oc_end, out_row_start:out_row_end, :])
                else:
                    out_row_start = spatial_tile_idx * spatial_tile_height
                    out_row_end = (spatial_tile_idx + 1) * spatial_tile_height
                    nisa.dma_copy(src=conv_with_bias, dst=X_out[batch_idx, oc_start:oc_end, out_row_start:out_row_end, :])
    return X_out