# Glossary of TVM Terminology

This glossary serves as the Single Source of Truth (SSOT) for technical terms used throughout this deep dive. 

## AST Mutator
An **Object-Oriented C++ Class** utilizing the **Visitor Design Pattern**. In TVM, a Mutator traverses the Abstract Syntax Tree (AST) and replaces specific nodes with new ones (e.g., translating a high-level `nn.conv2d` node into a `call_tir` node pointing to a C++ loop).

## Intrinsic Node
A built-in primitive node within the compiler that holds special, hard-coded semantic meaning rather than acting as a standard function call. For example, `call_tir` is an intrinsic that instructs the compiler to halt graph-level execution and drop down into physical TensorIR execution.

## Operator Fusion
The process of mathematically grouping independent graph-level operations (e.g., `Conv2d` and `Add`) into a single logical execution boundary to eliminate intermediate VRAM I/O overhead.

## Loop Tiling
The physical scheduling technique of dividing large loops (e.g., a 224x224 spatial loop) into smaller, fixed-size chunks (e.g., 16x16 tiles) to ensure intermediate data perfectly fits within hardware capacity limits (L1 Cache / Shared Memory).
