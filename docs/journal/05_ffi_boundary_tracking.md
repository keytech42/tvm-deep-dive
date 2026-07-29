# Crossing the Boundary: FFI and PackedFunc

!!! info "Deep Dive"
    This journal entry documents the experimental tracing of the Python-to-C++ boundary in TVM. For a theoretical explanation of *why* this boundary is designed this way (Serialization vs. Zero-copy), see [[03_ffi_and_packedfunc]].

## The Objective

In [Journal 04](04_frontend_code_tracing.md), we observed how the Python frontend parses an ONNX model and constructs an AST using `relax.op.nn.conv2d`. However, TVM's core compiler engine is written in C++. The objective of this trace is to explicitly observe the **Foreign Function Interface (FFI)** mechanism that bridges Python and C++ without the overhead of serialization.

## The Experiment (`03_ffi_boundary_test.py`)

To isolate the FFI boundary, we wrote a minimal script to bypass the Python wrapper and directly invoke the C++ function pointer.

```python title="03_ffi_boundary_test.py"
import tvm
from tvm import relax

print("--- 1. Getting the C++ PackedFunc ---")
# (1)!
conv2d_cpp = tvm.get_global_func("relax.op.nn.conv2d")
print("Python Type:", type(conv2d_cpp))
print("Object:", conv2d_cpp)

print("\n--- 2. Examining the Python Wrapper ---")
# (2)!
print("Python Type:", type(relax.op.nn.conv2d))
print("Object:", relax.op.nn.conv2d)

print("\n--- 3. Proving Zero-Copy Execution ---")
# (3)!
data = relax.Var("data", relax.TensorType([1, 3, 224, 224], "float32"))
weight = relax.Var("weight", relax.TensorType([16, 3, 3, 3], "float32"))

# (4)!
conv2d_node = conv2d_cpp(data, weight, (1, 1), (0, 0, 0, 0), (1, 1), 1, "NCHW", "OIHW", None, None)
print("Result Node Type:", type(conv2d_node))
print("Result Node:\n", conv2d_node)
```

1. We query the global C++ registry for the raw function pointer.
2. We examine the standard Python API used in the frontend parser.
3. We create dummy TVM AST variable nodes.
4. We execute the C++ function pointer directly, passing in the Python-created variables.

## Execution Results

```text
--- 1. Getting the C++ PackedFunc ---
Python Type: <class 'tvm_ffi.core.Function'>
Object: ffi.Function

--- 2. Examining the Python Wrapper ---
Python Type: <class 'function'>
Object: <function conv2d at 0x110705430>

--- 3. Proving Zero-Copy Execution ---
Result Node Type: <class 'tvm.ir.expr.Call'>
Result Node:
 R.nn.conv2d(data, weight, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype=None)
```

## Architectural Deductions

1. **The PackedFunc (`tvm_ffi.core.Function`)**: The C++ function `conv2d` registered via `TVM_REGISTER_GLOBAL` (or `refl::GlobalDef().def`) surfaces in Python not as a standard function, but as a wrapped C++ pointer (`ffi.Function`). The standard `relax.op.nn.conv2d` is merely a syntactic sugar wrapper around this pointer.
2. **Zero-Copy Node Generation**: When `conv2d_cpp(data, weight, ...)` is called, the Python `Var` objects are not serialized. Instead, their memory handles are passed directly to C++. 
3. **C++ AST Projection**: The returned object is of type `<class 'tvm.ir.expr.Call'>`. This confirms that the return value is a C++ AST memory structure projected back into the Python runtime.
4. **Separation of Creation and Binding**: The output shows an isolated `Call` node (`R.nn.conv2d(...)`). The FFI call strictly *creates* the node. Appending this node to the computational graph (AST) is a separate operation handled by the `BlockBuilder` (`bb.emit()`).
