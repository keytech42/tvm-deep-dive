# Operator Fusion: Execution and Tracing

!!! info "Deep Dive"
    This journal documents the execution of TVM's `FuseOps` and `FuseTIR` passes. For a deep dive into the C++ Pattern Matching algorithm that powers this fusion, see [[04_operator_fusion_architecture]].

## The Objective

In [Journal 04](04_frontend_code_tracing.md), we discovered that the frontend intentionally atomizes PyTorch's `Conv2d` into a separate `Conv` and `Add`. Our objective here is to prove that the C++ backend can detect this atomized pattern and re-fuse it into a single, hardware-optimized block.

## The Experiment (`04_operator_fusion_trace.py`)

We wrote a minimal Relax script constructing a graph with `conv2d -> add -> relu`, mimicking a standard Convolutional block.

We applied three transformation passes:

1. `LegalizeOps()`: Lowers high-level Relax operators into raw TIR `call_tir` bindings.
2. `FuseOps()`: Analyzes the graph and groups compatible bindings into a single `Primitive` function.
3. `FuseTIR()`: Takes the grouped functions and merges their underlying loops into a single monolithic block of TensorIR.

### 3 Core AST Transformations (Visualized)

To understand how TVM achieves this, we will trace the Abstract Syntax Tree (AST) mutations across the three critical compiler passes. Note that complex type hints and grid coordinates have been abstracted to highlight the pure architectural changes.

#### Phase 1: Lowering (`Original` -> `LegalizeOps`)

The compiler cannot optimize abstract mathematical symbols (`R.nn.conv2d`). It must "legalize" them into concrete TensorIR loop functions (`call_tir`) so the backend has actual C++ instructions to manipulate.

!!! note "Terminology: Legalize (Lowering)"
    In compiler engineering, to **Legalize** means converting an abstract, unexecutable concept (like a high-level `Conv2d` math symbol) into a concrete, valid, and executable representation that conforms to the strict rules of the target architecture or the lower-level Intermediate Representation (TIR).

<div class="diff-code" style="display: none;">
--- a/Original_Graph
+++ b/After_LegalizeOps
@@ -1,7 +1,7 @@
 @R.function
 def main(data, weight, bias):
+    cls = Module
     with R.dataflow():
-        conv = R.nn.conv2d(data, weight, padding=(1, 1))
-        out = R.add(conv, bias)
-        relu = R.nn.relu(out)
+        conv = R.call_tir(cls.conv2d, (data, weight))
+        out = R.call_tir(cls.add, (conv, bias))
+        relu = R.call_tir(cls.relu, (out,))
         R.output(relu)
</div>

#### Phase 2: Logical Grouping (`LegalizeOps` -> `FuseOps`)

`FuseOps` is a Graph-level pass. It does **not** touch the `for` loops. It runs a pattern matching algorithm (via Post-Dominator Trees and `OpPatternKind`), identifies that `Conv->Add->ReLU` can be safely fused, and wraps them in a new function marked with `Primitive: True`. This serves as a hard boundary telling the next pass to forcefully fuse these loops.

!!! info "Graph-Level vs. TIR-Level Passes"
    TVM compiler optimizations are strictly divided into two architectural levels:
    
    1. **Graph-Level (e.g., `FuseOps`)**: Looks at the macro topology (the forest). It analyzes the dataflow connections between operations and makes structural decisions (like drawing a logical bounding box for fusion) without knowing or touching the internal loop mechanics.
    2. **TIR-Level (e.g., `FuseTIR`)**: Looks at the micro execution (the trees). It directly manipulates the AST of the actual `for` loops, memory buffer allocations, and execution bindings within a specific node boundary.

<div class="diff-code" style="display: none;">
--- a/After_LegalizeOps
+++ b/After_FuseOps
@@ -1,6 +1,4 @@
 @R.function
 def main(data, weight, bias):
     cls = Module
     with R.dataflow():
-        conv = R.call_tir(cls.conv2d, (data, weight))
-        out = R.call_tir(cls.add, (conv, bias))
-        relu = R.call_tir(cls.relu, (out,))
-        R.output(relu)
+        gv = R.call_tir(cls.fused_conv2d_add_relu, (data, weight, bias))
+        R.output(gv)
@@ -10,0 +8,10 @@
+ @R.function(private=True)
+ def fused_conv2d_add_relu(data, weight, bias):
+     R.func_attr({"Primitive": True})
+     cls = Module
+     with R.dataflow():
+         conv = R.call_tir(cls.conv2d, (data, weight))
+         out = R.call_tir(cls.add, (conv, bias))
+         gv = R.call_tir(cls.relu, (out,))
+         R.output(gv)
</div>

#### Phase 3: Physical Fusion (`FuseOps` -> `FuseTIR`)

Here lies the true physical merge. The three independent `call_tir` blocks within the `Primitive` function are ripped apart and their `for` loops are bundled into a single C++ function (`@T.prim_func`). Notice that the loops are **not** immediately nested into each other; they remain as three independent sequential loops. However, the critical change is that intermediate buffers now use `T.sblock_alloc_buffer(...)` (mapped to L1 Cache/Registers) rather than allocating new tensors in global memory.

!!! success "Why is this sequential bundling called a 'Fusion'?"
    Before fusion, the 3 loops existed in completely separate functions (kernels). Because multiple kernels run under **temporal isolation** (one must finish completely before the next starts), moving data between them forces the data to be flushed to the extremely slow **VRAM (Global Memory)**. 
    By pulling these 3 independent loops into the *same* physical function context, `FuseTIR` breaks this temporal barrier. It creates an environment where they can pass data through `sblock_alloc_buffer` (L1 Cache / Shared Memory), completely eliminating the VRAM I/O bottleneck. 
    *(Note: The actual mathematical loop interleaving/nesting to maximize cache hit rates happens in the subsequent **TensorIR Tiling/Scheduling** phase).*

<div class="diff-code" style="display: none;">
--- a/After_FuseOps
+++ b/After_FuseTIR
@@ -1,9 +1,11 @@
- @R.function(private=True)
- def fused_conv2d_add_relu(data, weight, bias):
-     R.func_attr({"Primitive": True})
-     with R.dataflow():
-         conv = R.call_tir(cls.conv2d, (data, weight))
-         out = R.call_tir(cls.add, (conv, bias))
-         gv = R.call_tir(cls.relu, (out,))
-         R.output(gv)
+ @T.prim_func(private=True, s_tir=True)
+ def fused_conv2d_add_relu(data: T.Buffer, weight: T.Buffer, bias: T.Buffer, compute: T.Buffer):
+     T.func_attr({"tirx.noalias": True})
+     
+     conv2d_intermediate = T.sblock_alloc_buffer(...)
+     T_add_intermediate = T.sblock_alloc_buffer(...)
+     
+     for nn, ff, yy, xx, rc, ry, rx in T.grid(...):
+         # 1. Convolution Loop -> writes to conv2d_intermediate
+     for ax0, ax1, ax2, ax3 in T.grid(...):
+         # 2. Add Loop -> writes to T_add_intermediate
+     for i0, i1, i2, i3 in T.grid(...):
+         # 3. ReLU Loop -> writes to compute (VRAM)
</div>

## Architecture Deep Dive

!!! abstract "Architecture Clarifications & Key Insights"
    Before proceeding to TensorIR Tiling (Step 4B), we documented key insights regarding the philosophy of FFI zero-copy, the precise role of compiler passes like `LegalizeOps`, and the memory hierarchy mechanics that make physical fusion effective. See [[05_architecture_clarifications]] for a detailed breakdown of these architectural nuances.

