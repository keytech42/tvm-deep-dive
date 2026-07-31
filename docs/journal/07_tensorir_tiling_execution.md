# TensorIR Tiling & Scheduling: Execution and Tracing

!!! info "Deep Dive"
    This journal documents the execution of TVM's `tvm.tir.Schedule` API to tile and reorder loops. For a theoretical breakdown of why tiling is necessary and how it affects hardware utilization, see [[06_tensorir_scheduling]].

## The Objective

In [Journal 06](06_operator_fusion_execution.md), we saw how `FuseTIR` groups three operations (Conv2d, Add, ReLU) into a single C++ function (`PrimFunc`). However, the loops for these operations were still strictly sequential. 

Because an entire 224x224 image (and its intermediate states) cannot physically fit into a GPU's tiny L1 Cache (Shared Memory), we must chop the computation into smaller, manageable chunks (Tiles). Our objective is to trace how the `tvm.tir.Schedule` API achieves this.

## The Experiment (`05_tensorir_tiling_trace.py`)

We wrote a TVM script that takes the fused `Conv2d + Add + ReLU` module and applies TensorIR scheduling primitives to tile the spatial dimensions.

### 1. Loop Splitting (Tiling)

First, we extracted the loops for the convolution block (`yy` for height, `xx` for width). The original size is 224x224. We split these loops into tiles of 16x16 using `sch.split()`.

```python
# 224 / 16 = 14 outer iterations.
y_outer, y_inner = sch.split(yy, factors=[None, 16])
x_outer, x_inner = sch.split(xx, factors=[None, 16])
```

### 2. Loop Reordering

We then reordered the loops to group the outer dimensions together, creating a distinct "Tile Level" in the AST.

```python
sch.reorder(y_outer, x_outer, y_inner, x_inner)
```

### 3. Compute At (Physical Interleaving)

This is the most crucial step. We used `sch.reverse_compute_at` to pull the `Add` and `ReLU` computations *inside* the outer spatial loops of the `Conv2d` block.

```python
sch.reverse_compute_at(block_add, x_outer)
sch.reverse_compute_at(block_relu, x_outer)
```

## The AST Transformation (Visualized)

By observing the AST before and after, we can see exactly how the compiler physically interleaves the loops to keep data resident in the L1 Cache.

<div class="diff-code" style="display: none;">
--- a/Before_Tiling
+++ b/After_Tiling
@@ -1,9 +1,11 @@
- # 1. Convolution Loop (Computes Entire 224x224)
- for nn, ff, yy, xx, rc, ry, rx in T.grid(1, 16, 224, 224, 3, 3, 3):
-     # writes to conv2d_nchw_intermediate
- 
- # 2. Add Loop (Computes Entire 224x224)
- for ax0, ax1, ax2, ax3 in T.grid(1, 16, 224, 224):
-     # writes to T_add_intermediate
- 
- # 3. ReLU Loop (Computes Entire 224x224)
- for i0, i1, i2, i3 in T.grid(1, 16, 224, 224):
-     # writes to compute_intermediate
+ # Outer Tile Loop: Iterates over 14x14 Grid of Tiles
+ for nn, ff, yy_0, xx_0 in T.grid(1, 16, 14, 14):
+     
+     # 1. Convolution (Computes ONLY one 16x16 tile)
+     for yy_1, xx_1, rc, ry, rx in T.grid(16, 16, 3, 3, 3):
+         # writes to conv2d_nchw_intermediate
+         
+     # 2. Add (Computes ONLY one 16x16 tile)
+     for ax0, ax1 in T.grid(16, 16):
+         # writes to T_add_intermediate
+         
+     # 3. ReLU (Computes ONLY one 16x16 tile)
+     for ax0, ax1 in T.grid(16, 16):
+         # writes to compute_intermediate
</div>

!!! success "Result: Perfect Cache Utilization"
    By nesting the operations inside the `14x14` outer loop, the GPU will now load a single `16x16` tile of data, run the Convolution, Add the bias, and apply ReLU—all while the intermediate data remains safely inside the ultra-fast L1 Cache. Only the final result of that tile is written back to VRAM.
