# Conv2D Performance Analysis

## Overview
Analysis of performance bottlenecks in `conv2d.py` for larger images.

## Critical Performance Bottlenecks

### 1. Extremely Wasteful Memory Access (Lines 197-206)
**BIGGEST ISSUE** for larger images:

```python
# Loads ENTIRE plane (128, out_height, out_width)
x_plane = nl.ndarray((PARTITION_SIZE, out_height, out_width), dtype=X.dtype, buffer=nl.sbuf)
nisa.dma_copy(src=X[...], dst=x_plane)

# Flattens it to (128, out_height * out_width)
x_plane_flat = x_plane.reshape((PARTITION_SIZE, out_height * out_width))

# But only uses 512 spatial positions!
for i in nl.affine_range(SPATIAL_PARTITION_SIZE):  # SPATIAL_PARTITION_SIZE = 512
    if i < spatial_size:
        x_tile[:, i] = x_plane_flat[:, spatial_start + i]
```

**Impact**:
- For a 224x224 image: loading 128 × 224 × 224 = ~6.4M elements but only using 512 per iteration
- Memory bandwidth wasted: **~98.5% of loaded data is unused**
- Gets exponentially worse with larger images

**Location**: `conv2d.py:197-206`

---

### 2. Element-wise Copy Loop (Lines 204-206)
Copying data one column at a time in a Python-level loop:

```python
for i in nl.affine_range(SPATIAL_PARTITION_SIZE):
    if i < spatial_size:
        x_tile[:, i] = x_plane_flat[:, spatial_start + i]
```

**Problem**: This should be a single vectorized operation or direct slice copy, not 512 individual assignments.

**Location**: `conv2d.py:204-206`

---

### 3. Redundant Data Loading
The entire `x_plane` is reloaded from HBM for every `(ic_idx, fh, fw)` combination:

- Loop structure: `for ic_idx` → `for fh` → `for fw` → **load x_plane**
- Same spatial region loaded multiple times for different input channels and filter positions
- No reuse of loaded data across spatial partitions

**Location**: `conv2d.py:193-198`

---

### 4. Inefficient Writeback Pattern (Lines 221-226)
Writing back one spatial position at a time:

```python
for i in nl.affine_range(SPATIAL_PARTITION_SIZE):
    if i < spatial_size:
        # Individual DMA copy per position
        nisa.dma_copy(src=output_tile[:, i],
                     dst=X_out[batch_idx, oc_idx * PARTITION_SIZE: (oc_idx + 1) * PARTITION_SIZE, oh, ow])
```

**Problem**:
- Each DMA operation has overhead
- For 512 positions: 512 separate DMAs instead of batching
- Should reshape and write in larger chunks

**Location**: `conv2d.py:221-226`

---

### 5. Repeated Transpose Operations (Line 211)

```python
w_transposed = nl.copy(nisa.nc_transpose(w_tile), dtype=W.dtype)
```

**Problem**:
- Happens inside the innermost filter loop (`fh`, `fw`)
- Same weight tile transposed multiple times for each spatial partition
- Transpose could be done once per (ic_idx, oc_idx) pair

**Location**: `conv2d.py:211`

---

### 6. Unused Optimized Helper
The `nki_matmul_tiled_` function (lines 36-93) is defined but never used. The manual matmul approach may not be as optimized as this helper.

**Location**: `conv2d.py:36-93` (defined), never called

---

## Impact on Larger Images

### Example: 224×224 image with 3×3 filters

- `out_height × out_width = 222 × 222 = 49,284` spatial positions
- Number of spatial partitions: `⌈49,284 / 512⌉ = 97`
- **For each partition**: load entire 222×222 plane (49,284 elements) but use only 512
- **Total wasted loads**: 97 partitions × (49,284 - 512) = ~4.7M wasted elements loaded per filter position
- **Multiplied by**: `in_channels/128 × filter_height × filter_width` iterations

The memory bandwidth waste grows quadratically with image size!

---

## Key Recommendations

### High Priority
1. **Only load the spatial slice you need**
   - Instead of loading entire `(128, out_height, out_width)` plane
   - Load only `(128, spatial_size)` elements for current spatial partition
   - Compute which rows/cols in 2D space correspond to linear spatial indices

2. **Eliminate the element-wise copy loop**
   - Use direct slicing or vectorized operations
   - Avoid 512-iteration Python loop

3. **Batch writeback operations**
   - Reshape output_tile to match output layout
   - Write in fewer, larger DMA operations
   - Consider writing full rows or tiles at once

### Medium Priority
4. **Pre-transpose weights or reorganize computation**
   - Avoid repeated transposes in inner loop
   - Either transpose once outside loops or change matmul order

5. **Reuse loaded data across iterations**
   - Cache loaded input data when processing different filter positions
   - Consider reordering loops to maximize data reuse

6. **Consider im2col-style approach**
   - Reshape input into column matrix once
   - Perform single large matmul instead of many small ones
   - Better leverages matmul hardware

---

## Root Cause

The implementation was likely optimized for small images where loading full spatial planes is acceptable. The approach **does not scale** to larger spatial dimensions due to the massive memory bandwidth waste from loading entire planes but using only small slices.

The key insight: **spatial tiling should drive memory access patterns**, not just computation partitioning.
