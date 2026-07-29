# Compiler Design Philosophy

## The Necessity of Dummy Inputs for ONNX Export

PyTorch utilizes a **define-by-run** (eager execution) paradigm, meaning the computation graph is dynamically constructed during runtime as data flows through operations. Consequently, PyTorch lacks an explicit, static representation of the graph architecture prior to execution.

To export a model to a static format like ONNX, PyTorch employs a technique known as **tracing**. 
Tracing requires passing a **dummy input** (a tensor containing arbitrary values but structurally identical to real data) through the model. The tracer monitors and records the sequence of operations applied to this tensor, serializing the resulting execution path into the ONNX graph. If data-dependent control flow (e.g., `if` statements) exists, tracing strictly captures the branch executed by the specific dummy input.

## The `.onnx.data` File Mechanism

During ONNX export, a deep learning model is bifurcated into two conceptual components:

1.  **Topology:** The structural layout defining how mathematical nodes are interconnected.
2.  **Parameters (Weights):** The trained numerical data tensors residing within the nodes.

By default, ONNX attempts to embed both components within a single `.onnx` file. However, Protocol Buffers (the underlying serialization format for ONNX) enforce a strict 2GB file size limit. If the model weights exceed this threshold, or if the external data flag is explicitly enabled during export, the topology is serialized into the primary `.onnx` file, while the heavy parameter tensors are externalized to a binary `.onnx.data` file. The compiler consumes both simultaneously.

## TVMScript: Abstract Syntax Tree vs. Pseudo-code

A common misconception is that `tvm_script.txt` functions merely as read-only pseudo-code for human inspection. This is fundamentally incorrect.

**TVMScript is a fully parseable, round-trippable Intermediate Representation (IR).**
It serves as the concrete Abstract Syntax Tree (AST) that the TVM compiler internalizes, mutates, and optimizes. A developer can author TVMScript manually, completely bypassing the ONNX frontend, and feed it directly into the TVM compilation pipeline via Python decorators (e.g., `@I.ir_module`, `@R.function`) to generate the identical `.so` binary. It is the core language of the TVM backend.

## Static vs. Dynamic Compilation

### The Static Paradigm (Legacy Compilers & Vanilla TensorRT)
Historically, compilers demanded strict static shapes (e.g., dimensions locked to `1x3x224x224`). 
**Advantages:**
- **Deterministic Memory Allocation:** The precise memory footprint is known at compile-time, allowing the GPU to allocate fixed buffers without dynamic runtime `malloc`/`free` overhead.
- **Aggressive Optimization:** The compiler can execute perfect loop unrolling, select exact hardware-specific instruction tiles, and maximize cache locality.

**Disadvantages:**
- If the batch size or input dimensions vary at runtime (e.g., feeding two images instead of one), the compiled binary crashes. Recompilation is strictly required.

### The Dynamic Paradigm (TVM Relax & TensorRT-LLM)
Modern workloads, particularly Large Language Models (LLMs), fundamentally conflict with static compilation. LLMs process dynamic sequence lengths depending on user prompts and utilize autoregressive decoding where the Key-Value (KV) cache grows continuously.

Architectures like TVM Relax and NVIDIA's TensorRT-LLM resolve this by introducing **Symbolic Shapes**.
In TVMScript, binding `batch_size = T.int64()` forces the compiler to treat the dimension as an algebraic variable rather than a constant. The resulting machine code retains the logic necessary to allocate memory and calculate strides dynamically at runtime. While incurring minor computational overhead, it ensures robust execution across variable sequence lengths—a strict prerequisite for modern GenAI infrastructure.

## Anatomy of the Compiled `.so` Artifact

The output of the TVM pipeline is a Shared Object (`.so`) file.

Fundamentally, an `.so` file is an Executable and Linkable Format (ELF) binary containing low-level **machine code instructions** (Assembly lowered to binary form, specific to the target architecture such as ARM64, x86_64, or PTX/CUBIN). 

However, TVM's `.so` artifacts are not exclusively comprised of executable logic. The binary is structured into segments, which embed:

1.  **Instructions (`.text` segment):** The actual hardware-executable operations.
2.  **Metadata (`.rodata` segment):** Serialized JSON strings delineating the graph structure or function signatures.
3.  **Weights (The Dummy vs. Production Dichotomy):** The storage of model parameters varies drastically depending on the compilation strategy:
    - *Minimal/Dummy Execution (Embedded):* For small models compiled without specific serialization flags, TVM treats weights as constants. These constants are physically compiled directly into the read-only data section (`.rodata`) of the `.so` binary.
    - *Production Architecture (Separated):* For modern Large Language Models (LLMs) scaling to tens of gigabytes, embedding weights within an ELF binary is fundamentally impossible due to OS binary loading constraints. In production, the executable logic (`.so`) and the weight dictionaries (typically `.params` files) are strictly separated. The C++ runtime engine dynamically loads the `.params` file into memory during inference.

The logic execution is strictly handled by the machine code instructions, rendering the binary opaque without the use of a disassembler.

## The Execution Gap and the FFI Boundary

Once the `.so` binary is generated, a critical architectural challenge emerges: execution. 

Python and C++ operate within fundamentally different memory management paradigms. Attempting to pass a standard Python NumPy array into a compiled C++ binary natively would require massive serialization overhead, negating any performance gains achieved by the compiler.

To execute the `.so` seamlessly from Python without this overhead (Zero-copy), TVM relies on a highly optimized Foreign Function Interface (FFI). The core technology facilitating this boundary crossing is the `PackedFunc`—a unified C++ abstraction that allows Python to invoke C++ functions natively. Dissecting this mechanism is the focus of the upcoming FFI Boundary Tracking phase.
