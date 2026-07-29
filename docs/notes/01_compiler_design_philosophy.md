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
3.  **Weights (Optional):** If exported via specific packing routines, the model's trained parameter arrays are physically bundled into data segments within the binary.

The logic execution is strictly handled by the machine code instructions, rendering the binary opaque without the use of a disassembler.
