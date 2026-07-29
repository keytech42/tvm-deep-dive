# Serialization vs. Zero-Copy: The FFI Boundary

This note explains the rationale behind TVM's `PackedFunc` architecture and the concept of "Zero-copy" execution across language boundaries.

## The Bottleneck: Serialization in Inter-Language Communication

In typical software systems, when two different environments (e.g., a Python process and a C++ engine) need to share information or communicate, they rely on a technique called **Serialization**.

Serialization converts a complex, memory-resident object (like a Python dictionary or class) into a linear format (like a JSON string, byte array, or Protobuf) that can be transmitted across the boundary. The receiving end then **Deserializes** this string back into a memory-resident object native to its own language.

### Why Serialization is an Anti-Pattern for Compilers

While serialization is standard for microservices or network communication, it introduces a massive CPU overhead when used within the tight loop of an AI compiler:

1. **Volume**: Deep learning graphs (e.g., LLaMA, ResNet) contain tens to hundreds of thousands of interconnected nodes.
2. **Overhead**: If the Python frontend had to serialize every single `Conv2d`, `Add`, and `ReLU` node into a JSON string, and the C++ backend had to parse that string back into a C++ AST object, the compiler would spend the vast majority of its CPU cycles just doing string manipulation, severely bottlenecking the actual compilation and optimization processes.

## The Solution: PackedFunc and Zero-Copy

To eliminate this overhead, TVM utilizes an extreme optimization for its Foreign Function Interface (FFI): **Zero-copy memory sharing via `PackedFunc`**.

### The Mechanics of Zero-Copy

If typical communication relies on serialization, the ultimate optimization is to bypass communication entirely. "Zero-copy" literally means that data is never copied or duplicated across the language boundary. There is no transmission of information in the traditional serialized sense.

1. **Unified Memory (`tvm::Object`)**: Both the Python runtime and the C++ engine in TVM are designed to operate on a shared, unified memory structure known as `tvm::Object`.
2. **Passing Pointers, Not Data**: When a Python script (the frontend) invokes a C++ function (the compiler engine), it does not send the contents of the variables. Instead, it sends the 64-bit memory address (a pointer/handle) pointing to where the `tvm::Object` resides in the C++ heap.
3. **Absolute Sharing**: Because both Python and C++ are physically looking at the exact same address space in RAM, no data is transmitted. Python simply says to C++, "Look at memory address `0x110705430`." 

This is the essence of TVM's FFI boundary. By treating the Python frontend strictly as a lightweight control plane that juggles C++ memory pointers, TVM achieves native C++ compilation speeds while retaining the user-friendly interface of Python.

---
**Related Execution Log**: [[05_ffi_boundary_tracking]]
