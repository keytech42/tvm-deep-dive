# Glossary of TVM Terminology

This glossary serves as the Single Source of Truth (SSOT) for technical terms used throughout this deep dive. 

## AST Mutator
An **Object-Oriented C++ Class** utilizing the **Visitor Design Pattern**. In TVM, a Mutator traverses the Abstract Syntax Tree (AST) and replaces specific nodes with new ones (e.g., translating a high-level `nn.conv2d` node into a `call_tir` node pointing to a C++ loop).

## FFI (Foreign Function Interface)
A boundary mechanism that allows code written in one language (e.g., Python) to invoke routines in another (e.g., C++). In TVM, the FFI is critical for exposing low-level C++ compiler passes and data structures to the Python frontend without incurring serialization overhead.

## Intrinsic Node
A built-in primitive node within the compiler that holds special, hard-coded semantic meaning rather than acting as a standard function call. For example, `call_tir` is an intrinsic that instructs the compiler to halt graph-level execution and drop down into physical TensorIR execution.

## L1 Cache (Shared Memory)
The extremely fast, physically close memory region on a hardware device (such as a GPU). It has very low latency but severely limited capacity (e.g., 64KB), mandating the use of Loop Tiling to ensure data chunks can fit entirely within it during execution.

## Loop Tiling
The physical scheduling technique of dividing large loops (e.g., a 224x224 spatial loop) into smaller, fixed-size chunks (e.g., 16x16 tiles) to ensure intermediate data perfectly fits within hardware capacity limits (L1 Cache / Shared Memory).

## Operator Fusion
The process of mathematically grouping independent graph-level operations (e.g., `Conv2d` and `Add`) into a single logical execution boundary to eliminate intermediate VRAM I/O overhead.

## PackedFunc
TVM's proprietary, type-erased FFI mechanism. It packs diverse Python arguments into a standard C-union array (`TVMValue`), allowing seamless C++ function invocation from Python with virtually zero copy or serialization costs.

## PrimFunc
A "Primitive Function" representing a physical, nested loop boundary in TVM's C++ core (TensorIR). While `relax.Function` represents graph-level logic, a `PrimFunc` dictates the exact spatial execution instructions for the hardware.

## VRAM (Global Memory)
The massive, relatively slow global memory space on a hardware device. Fetching data from VRAM is extremely expensive; therefore, optimization techniques like Fusion and Tiling aim to minimize VRAM I/O round-trips.
