# Journal 03: The Big Picture (Relax Compilation Pipeline)

## Objective
Analyze the end-to-end pipeline of TVM: The transformation of a high-level Deep Learning model (PyTorch) into a hardware-specific C++ Shared Library (`.so`).

## Architectural Shift: Relay to Relax
Initial roadmaps included the `tvmc` CLI tool, which utilized TVM's legacy **Relay** frontend. Inspection of the `0.26.dev0` build confirms the complete deprecation and removal of Relay and `tvmc`.

The current repository exclusively utilizes **TVM Unity (Relax)**. This architecture is designed specifically to resolve the limitations of static compilation, providing native support for dynamic shapes and modern Large Language Model (LLM) workloads. We interact directly with the `tvm.relax` API.

## Pipeline Execution

The compilation process is executed in two sequential stages:

### 1. Generating a Dummy Model (`scripts/01_generate_model.py`)
A minimal PyTorch model containing `Conv2d`, `ReLU`, `Flatten`, and `Linear` operations was constructed and exported to `dummy_model.onnx`. A minimal model ensures rapid compilation while preserving the exact architectural pipeline utilized by large-scale models.

### 2. Compiling with Relax (`scripts/02_compile_model.py`)
Compilation is performed via the Python API:
1. Load the ONNX model.
2. Ingest the graph into TVM's internal representation via `relax.frontend.onnx.from_onnx`.
3. Compile the Intermediate Representation (IR) into an Executable using `relax.build`.
4. Export the compiled C++ binary to `compiled_model.so`.

## Inspecting the TVMScript (Internal IR)

The intermediate artifact `dummy_model_tvmscript.txt` reveals the parsed Abstract Syntax Tree (AST). TVM translates the ONNX graph into **TVMScript**, a round-trippable, Python-like IR.

```python
@I.ir_module
class Module:
    @R.function
    def main(input: R.Tensor(("batch_size", 3, 224, 224), dtype="float32")) -> R.Tensor(("batch_size", 10), dtype="float32"): # (1)!
        batch_size = T.int64() # (2)!
        R.func_attr({"num_input": 1})
        with R.dataflow(): # (3)!
            lv: R.Tensor((batch_size, 16, 224, 224), dtype="float32") = R.nn.conv2d(input, metadata["relax.expr.Constant"][0], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype=None)
            lv1: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][1], R.shape([1, 16, 1, 1]))
            lv2: R.Tensor((batch_size, 16, 224, 224), dtype="float32") = R.add(lv, lv1) # (4)!
            lv3: R.Tensor((batch_size, 16, 224, 224), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((batch_size, 802816), dtype="float32") = R.reshape(lv3, R.shape([batch_size, 802816]))
            lv5: R.Tensor((802816, 10), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][2], axes=[1, 0])
            lv6: R.Tensor((batch_size, 10), dtype="float32") = R.matmul(lv4, lv5, out_dtype=None)
            gv: R.Tensor((batch_size, 10), dtype="float32") = R.add(lv6, metadata["relax.expr.Constant"][3])
            R.output(gv)
        return gv
```

1.  **Dynamic Shape Mapping:** PyTorch defined a static input dimension `1`, but TVM maps this dimension to a string identifier `"batch_size"`.
2.  **Symbolic Variable Binding:** `T.int64()` constructs a symbolic variable. This confirms TVM does not hardcode the dimension, allowing for dynamic batch sizes at runtime.
3.  **Dataflow Block:** Operations within `R.dataflow()` are guaranteed to lack side effects. This context allows the compiler to safely perform aggressive optimizations (e.g., operator fusion, instruction reordering).
4.  **Operator Decomposition:** PyTorch's `Conv2d` implicitly handles bias addition. TVM IR decomposes this into discrete mathematical operations: a convolution (`lv`) followed by an explicit matrix addition (`lv2`).

!!! info "Deep Dive: TVM Design Philosophy"
    Why are dynamic shapes critical for modern LLMs? Why did legacy compilers like TensorRT enforce static compilation? What is the exact binary composition of the `.so` artifact? For a rigorous technical analysis of these concepts, refer to the [Compiler Design Philosophy](../notes/01_compiler_design_philosophy.md) note.

## Conclusion
The end-to-end pipeline is verified. A PyTorch model is ingested, translated into the **Relax IR** (TVMScript), and lowered into a native `compiled_model.so` binary targeting the LLVM backend. 

**Next Objective:** Trace the FFI (Foreign Function Interface) boundary via `PackedFunc` to observe the zero-serialization transition from Python to C++.
