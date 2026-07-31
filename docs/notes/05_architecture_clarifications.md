# Architecture Clarifications (Step 3 & 4A)

When dissecting the TVM compiler pipeline—specifically the transition from Python Graph execution to C++ TensorIR loops (Steps 3 & 4A)—there are several architectural nuances that require precise understanding. This document clarifies key mechanisms regarding the FFI boundary, compiler passes, and memory fusion.

## 1. The FFI Boundary: Serialization vs. Zero-Copy

!!! info "Key Insight: Shared Memory & Type Erasure"
    **"TVM avoids serialization across the Python/C++ boundary by passing memory pointers directly via `PackedFunc`."**
    
    It is natural to assume that crossing the language boundary from Python to C++ requires marshalling or serialization of data (like JSON or Protobuf). However, TVM strictly avoids this overhead by leveraging a shared memory space.

**Deep Dive:**

- **`PackedFunc` as a Type Eraser**: The core of this mechanism is the `PackedFunc`. Rather than writing heavy bindings for every specific C++ function signature, `PackedFunc` acts as a type eraser. It packs dynamically-typed arguments from Python (Tensors, integers, strings) into a single, uniform C-union array called `TVMValue`.
- **Zero-Copy**: The C++ core receives this `TVMValue` array and directly accesses the underlying memory pointers (e.g., `DLTensor`). Because both Python and C++ are accessing the same "shared cutting board", no data is ever copied across the boundary.

## 2. Compiler Passes and the Definition of `call_tir`

!!! info "Key Insight: The Nature of LegalizeOps and call_tir"
    **"`LegalizeOps` does not atomize nodes but translates them to loops, and `call_tir` is an intrinsic boundary, not a standard function."**
    
    When viewing the AST, one might assume `LegalizeOps` splits complex PyTorch nodes into smaller operations, and that `call_tir` is just a standard function call. In reality, the roles are distinct and fundamentally different.

**Deep Dive:**

- **Atomization vs. Legalization**:<br>The atomization of high-level PyTorch nodes (e.g., splitting a fused Conv+Bias into `Conv` and `Add`) actually occurs earlier, during the **Frontend Parsing (Step 2)**. The `LegalizeOps` pass does not split nodes; it translates abstract mathematical symbols into concrete C++ loops.
- **Implementation of a Compiler Pass**:<br>In the C++ backend, a pass like `LegalizeOps` is an **Object-Oriented C++ Class** utilizing the **Visitor Design Pattern**. It inherits from an [[glossary#ast-mutator|AST Mutator]] class (like `tvm::relax::ExprMutator`). When the compiler traverses the AST and hits an `R.nn.conv2d` node, an overridden `VisitExpr_` method is triggered, replacing the node with a `call_tir` node.
- **The Nature of `call_tir`**:<br>`call_tir` is **not a normal function**. It is a **[[glossary#intrinsic-node|Built-in Operator (Intrinsic Node)]]**. It serves as a hard boundary instructing the compiler: *"Halt graph-level execution here, drop down to the specific C++ loop (PrimFunc), and write the execution result directly into this pre-allocated buffer."*

## 3. Physical Fusion and Memory Hierarchy

!!! warning "Common Misconception"
    **False Belief: "Fusion works by building data up from the registers, and physical fusion literally interleaves(nests) loops immediately."**

    It is easy to imagine that physical fusion literally interleaves loops immediately, or that data is built "bottom-up" from registers. However, the actual mechanism relies on fetching data from VRAM and sequentially bundling loops to control I/O.

**Deep Dive:**

- **Memory Directionality**: Hardware does not build data upwards. The source of truth is always the massive, slow **VRAM (Global Memory)**. Data is pulled (loaded/fetched) *down* the hierarchy: VRAM → L2 Cache → L1 Cache (Shared Memory) → Registers.
- **The True Benefit of Sequential Fusion**: `FuseTIR` initially places the three independent loops (`Conv`, `Add`, `ReLU`) **sequentially** within the same physical function boundary (`PrimFunc`); it does not immediately interleave them. However, by existing in the same physical function, these loops can pass intermediate data to each other via `T.sblock_alloc_buffer` (which maps to L1 Cache or Registers) rather than flushing data all the way back up to VRAM. This elimination of VRAM I/O is the true power of fusion.
- **Capacity Limits and Tiling**: Even with sequential fusion, L1 cache has severe capacity limits (e.g., 64KB). A full 224x224 image cannot fit inside it. This physical limitation mandates **Loop Tiling & Scheduling** (Step 4B). Using TensorIR scheduling primitives (like `split` and `compute_at`), the compiler slices the data into small blocks (e.g., 16x16 tiles) and physically nests (interleaves) the loops. This ensures the hardware runs the fully fused sequence exclusively within that tile before moving on to the next, perfectly optimizing cache hits.

--8<-- "docs/snippets/glossary_abbr.md"
