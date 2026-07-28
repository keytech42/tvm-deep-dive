# 🚀 TVM Reverse-Engineering Portfolio

Welcome to my deep-dive journey into **Apache TVM**.

## 🎯 Ultimate Goal

The purpose of this project is not just to use TVM, but to **dismantle and understand it from the inside out**. As a junior engineer aiming for elite expertise, I am taking a "Top-Down Reverse Engineering" approach. The goal is to prove that I can navigate complex, massive C++ AI compiler codebases, understand the bridging between Python and C++, and pinpoint low-level hardware optimizations.

## 🗺️ The 3-Step Reverse Engineering Roadmap

### Step 0: The Foundation (Completed)
- Clone TVM and build it from source (handling C++ CMake and Python FFI module complexities).
- *See [Journal 01](journal/01_environment_and_clone.md) and [Journal 02](journal/02_build_and_compile.md).*

### Step 1: The Big Picture via TVMC (Next)
- Use the `tvmc` CLI tool to compile a simple PyTorch/ONNX model.
- **Objective:** Visually confirm the entire pipeline—"A Python model goes in, a `.so` library comes out."

### Step 2: Python Frontend Tracing
- Dive into `python/tvm/relax` or `relay`.
- **Objective:** Trace how familiar Python operations (e.g., `Conv2d`) are ingested into TVM's internal graph nodes.

### Step 3: Crossing the Boundary (FFI & PackedFunc)
- Explore how TVM transitions from Python to C++ with zero serialization overhead.
- **Objective:** Master the `PackedFunc` mechanism to understand how heavy lifting is instantly delegated to C++.

### Step 4: C++ Core & TensorIR Optimization
- Zoom in on `src/tir/transforms`.
- **Objective:** Dissect a single core optimization technique (like **Tiling** or **Operator Fusion**) as if examining it under a microscope.

---
*This documentation is served via MkDocs and acts as a living journal of my learning process.*
