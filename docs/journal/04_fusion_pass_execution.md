# 04. Operator Fusion: Execution and Tracing

!!! info "Deep Dive"
    This journal documents the execution of TVM's `FuseOps` and `FuseTIR` passes. For a deep dive into the C++ Pattern Matching algorithm that powers this fusion, see [[04_operator_fusion_architecture]].

## The Objective

In [Journal 02](02_frontend_code_tracing.md), we discovered that the frontend intentionally atomizes PyTorch's `Conv2d` into a separate `Conv` and `Add`. Our objective here is to prove that the C++ backend can detect this atomized pattern and re-fuse it into a single, hardware-optimized block.

## The Experiment (`04_operator_fusion_trace.py`)

We wrote a minimal Relax script constructing a graph with `conv2d -> add -> relu`, mimicking a standard Convolutional block.

We applied three transformation passes:
1. `LegalizeOps()`: Lowers high-level Relax operators into raw TIR `call_tir` bindings.
2. `FuseOps()`: Analyzes the graph and groups compatible bindings into a single `Primitive` function.
3. `FuseTIR()`: Takes the grouped functions and merges their underlying loops into a single monolithic block of TensorIR.

### Results

Before Fusion, the AST contained three separate function calls:
```python
conv = R.call_tir(cls.conv2d, (data, weight))
out = R.call_tir(cls.add, (conv, bias))
gv = R.call_tir(cls.relu, (out,))
```

**Phase 1: Graph-Level Fusion (`FuseOps`)**
The `FuseOps` pass detected the pattern and collapsed all three calls into a single fused function wrapper:
```python
gv = R.call_tir(cls.fused_conv2d_add_relu, (data, weight, bias))
```

**Phase 2: TensorIR Instruction Generation (`FuseTIR`)**
When digging into the generated `fused_conv2d_add_relu` function, we observed the true power of compiler fusion. Instead of three separate loop structures writing to main memory, TVM generated a single monolithic `prim_func`.

*Key Observation:*
The intermediate results (`conv2d_nchw_intermediate` and `T_add_intermediate`) were allocated inside the shared loop context.
```python
# The Convolution Loop
conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = ... + pad_temp[...] * weight[...]

# The Add Loop (FMA realization)
T_add_intermediate[...] = conv2d_nchw_intermediate[...] + bias[...]

# The ReLU Loop
compute_intermediate[...] = T.max(T_add_intermediate[...], T.float32(0.0))
```
These values are kept in intermediate buffers (which compile down to L1 cache or registers), completely bypassing the massive overhead of flushing memory back to VRAM between operations. This confirms our hypothesis: the frontend atomizes operations precisely so the backend has the granular freedom to reassemble them perfectly for the target hardware.
