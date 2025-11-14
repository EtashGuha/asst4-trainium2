import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

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
    out_channels, in_channels_w, filter_height, filter_width = W.shape
    bias_channels = bias.shape[0]

    assert in_channels == in_channels_w and out_channels == bias_channels
    assert pool_size in (1, 2)

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    assert out_height % pool_size == 0 and out_width % pool_size == 0

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    assert in_channels % nl.tile_size.pmax == 0
    assert out_channels % nl.tile_size.pmax == 0

    PARTITION = nl.tile_size.pmax  # 128
    MOVING_TILE = nl.tile_size.gemm_moving_fmax  # 512

    # Choose tile sizes that keep tensor-engine tiles <= MOVING_TILE and pool-aligned.
    tile_h_base = min(out_height, 8)
    if tile_h_base % pool_size != 0:
        tile_h_base = (tile_h_base // pool_size) * pool_size
        if tile_h_base == 0:
            tile_h_base = pool_size
    tile_w_base = min(out_width, max(1, MOVING_TILE // tile_h_base))
    if tile_w_base % pool_size != 0:
        tile_w_base = (tile_w_base // pool_size) * pool_size
        if tile_w_base == 0:
            tile_w_base = pool_size

    full_h_tiles = out_height // tile_h_base
    h_remainder = out_height % tile_h_base
    full_w_tiles = out_width // tile_w_base
    w_remainder = out_width % tile_w_base

    num_oc_tiles = out_channels // PARTITION
    num_ic_tiles = in_channels // PARTITION

    X_out = nl.ndarray(
        (batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    def compute_tile(
        batch_idx,
        oc_tile,
        oc_start,
        oh_start,
        ow_start,
        tile_h,
        tile_w,
        bias_vec,
    ):
        tile_spatial = tile_h * tile_w
        psum_tile = nl.zeros((PARTITION, tile_spatial), dtype=nl.float32, buffer=nl.psum)

        for ic_tile in nl.affine_range(num_ic_tiles):
            ic_start = ic_tile * PARTITION

            x_cols = nl.ndarray((PARTITION, tile_spatial), dtype=X.dtype, buffer=nl.sbuf)
            patch_tile = nl.ndarray(
                (PARTITION, tile_h, tile_w), dtype=X.dtype, buffer=nl.sbuf
            )

            weight_block = nl.ndarray(
                (PARTITION, PARTITION, filter_height, filter_width),
                dtype=W.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=weight_block,
                src=W[
                    oc_start : oc_start + PARTITION,
                    ic_start : ic_start + PARTITION,
                    :,
                    :,
                ],
            )

            # Load a larger patch that covers all filter positions (HBM -> SBUF once)
            large_patch_h = tile_h + filter_height - 1
            large_patch_w = tile_w + filter_width - 1
            large_patch = nl.ndarray(
                (PARTITION, large_patch_h, large_patch_w), dtype=X.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=large_patch,
                src=X[
                    batch_idx,
                    ic_start : ic_start + PARTITION,
                    oh_start : oh_start + large_patch_h,
                    ow_start : ow_start + large_patch_w,
                ],
            )

            for fh in range(filter_height):
                for fw in range(filter_width):
                    # Extract the specific patch for this filter position (SBUF -> SBUF)
                    patch_tile[:, :, :] = nisa.tensor_copy(
                        large_patch[:, fh : fh + tile_h, fw : fw + tile_w],
                        engine=nisa.vector_engine
                    )

                    for rel_h in nl.affine_range(tile_h):
                        start = rel_h * tile_w
                        x_cols[:, start : start + tile_w] = nisa.tensor_copy(
                            patch_tile[:, rel_h, :],
                            engine=nisa.vector_engine
                        )

                    weight_slice = weight_block[:, :, fh, fw]
                    weight_transposed = nisa.nc_transpose(weight_slice)
                    weight_stationary = nisa.tensor_copy(
                        weight_transposed, engine=nisa.vector_engine
                    )

                    psum_tile += nisa.nc_matmul(weight_stationary[...], x_cols[...])

        conv_tile = nl.copy(psum_tile, dtype=X.dtype)
        for col in nl.affine_range(tile_spatial):
            conv_tile[:, col : col + 1] = nisa.tensor_tensor(
                conv_tile[:, col : col + 1], bias_vec, nl.add
            )

        if pool_size == 2:
            pooled_h = tile_h // pool_size
            pooled_w = tile_w // pool_size
            conv_hw = nl.ndarray(
                (PARTITION, tile_h, tile_w), dtype=X.dtype, buffer=nl.sbuf
            )
            for rel_h in nl.affine_range(tile_h):
                start = rel_h * tile_w
                conv_hw[:, rel_h, :] = nisa.tensor_copy(
                    conv_tile[:, start : start + tile_w],
                    engine=nisa.vector_engine
                )
            pooled_hw = nl.ndarray(
                (PARTITION, pooled_h, pooled_w), dtype=X.dtype, buffer=nl.sbuf
            )
            for ph in nl.affine_range(pooled_h):
                for pw in nl.affine_range(pooled_w):
                    h_base = ph * pool_size
                    w_base = pw * pool_size
                    patch = conv_hw[
                        :, h_base : h_base + pool_size, w_base : w_base + pool_size
                    ]
                    patch_max = nisa.tensor_reduce(
                        nl.max, patch, axis=(1, 2), keepdims=True
                    )
                    pooled_hw[:, ph : ph + 1, pw : pw + 1] = nisa.tensor_copy(
                        patch_max,
                        engine=nisa.vector_engine
                    )

            pooled_tile = nl.ndarray(
                (PARTITION, pooled_h * pooled_w), dtype=X.dtype, buffer=nl.sbuf
            )
            for rel_h in nl.affine_range(pooled_h):
                start = rel_h * pooled_w
                pooled_tile[:, start : start + pooled_w] = nisa.tensor_copy(
                    pooled_hw[:, rel_h, :],
                    engine=nisa.vector_engine
                )

            store_tile = pooled_tile
            store_h = pooled_h
            store_w = pooled_w
            base_h = oh_start // pool_size
            base_w = ow_start // pool_size
        else:
            store_tile = conv_tile
            store_h = tile_h
            store_w = tile_w
            base_h = oh_start
            base_w = ow_start

        oc_start = oc_tile * PARTITION
        # Reshape store_tile to 2D format and do a single DMA copy
        store_tile_2d = nl.ndarray((PARTITION, store_h, store_w), dtype=X.dtype, buffer=nl.sbuf)
        for rel_h in nl.affine_range(store_h):
            store_tile_2d[:, rel_h, :] = nisa.tensor_copy(
                store_tile[:, rel_h * store_w : (rel_h + 1) * store_w],
                engine=nisa.vector_engine
            )

        # Single DMA copy for the entire tile
        nisa.dma_copy(
            dst=X_out[
                batch_idx,
                oc_start : oc_start + PARTITION,
                base_h : base_h + store_h,
                base_w : base_w + store_w,
            ],
            src=store_tile_2d,
        )

    for batch_idx in nl.affine_range(batch_size):
        for oc_tile in nl.affine_range(num_oc_tiles):
            oc_start = oc_tile * PARTITION

            bias_vec = nl.ndarray((PARTITION, 1), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=bias_vec[:, 0:1], src=bias[oc_start : oc_start + PARTITION]
            )

            for oh_tile in nl.affine_range(full_h_tiles):
                oh_start = oh_tile * tile_h_base
                for ow_tile in nl.affine_range(full_w_tiles):
                    compute_tile(
                        batch_idx,
                        oc_tile,
                        oc_start,
                        oh_start,
                        ow_tile * tile_w_base,
                        tile_h_base,
                        tile_w_base,
                        bias_vec,
                    )
                if w_remainder > 0:
                    compute_tile(
                        batch_idx,
                        oc_tile,
                        oc_start,
                        oh_start,
                        full_w_tiles * tile_w_base,
                        tile_h_base,
                        w_remainder,
                        bias_vec,
                    )

            if h_remainder > 0:
                oh_start = full_h_tiles * tile_h_base
                for ow_tile in nl.affine_range(full_w_tiles):
                    compute_tile(
                        batch_idx,
                        oc_tile,
                        oc_start,
                        oh_start,
                        ow_tile * tile_w_base,
                        h_remainder,
                        tile_w_base,
                        bias_vec,
                    )
                if w_remainder > 0:
                    compute_tile(
                        batch_idx,
                        oc_tile,
                        oc_start,
                        oh_start,
                        full_w_tiles * tile_w_base,
                        h_remainder,
                        w_remainder,
                        bias_vec,
                    )

    return X_out
