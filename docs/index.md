# 🚀 TVM Reverse-Engineering Portfolio

Welcome to my deep-dive journey into **Apache TVM**.

## 🎯 Ultimate Goal

The purpose of this project is not just to use TVM, but to **dismantle and understand it from the inside out**. As a junior engineer aiming for elite expertise, I am taking a "Top-Down Reverse Engineering" approach. The goal is to prove that I can navigate complex, massive C++ AI compiler codebases, understand the bridging between Python and C++, and pinpoint low-level hardware optimizations.

## 🗺️ The 3-Step Reverse Engineering Roadmap

### Step 0: The Foundation (Completed)
- Clone TVM and build it from source (handling C++ CMake and Python FFI module complexities).
- *Environment requirements are frozen in `requirements.txt`. Make sure to run `pip install -r requirements.txt`.*
- *See [Journal 01](journal/01_environment_and_clone.md) and [Journal 02](journal/02_build_and_compile.md).*

### Step 1: The Big Picture via Relax (Completed)
- Used the modern `tvm.relax` Python API to compile a simple PyTorch/ONNX model into a native `.so` library.
- *See [Journal 03](journal/03_relax_compilation_pipeline.md).*

### Step 2: Python Frontend Tracing (Completed)
- Dive into `python/tvm/relax` or `relay`.
- **Objective:** Trace how familiar Python operations (e.g., `Conv2d`) are ingested into TVM's internal graph nodes.
- *See [Journal 04: Python Frontend Code Tracing](journal/02_frontend_code_tracing.md).*
- *See [Note 02: Frontend Parsing and AST](notes/02_frontend_parsing_and_ast.md).*

### Step 3: Crossing the Boundary (FFI & PackedFunc)
- Explore how TVM transitions from Python to C++ with zero serialization overhead.
- **Objective:** Master the `PackedFunc` mechanism to understand how heavy lifting is instantly delegated to C++.

### Step 4: C++ Core & TensorIR Optimization
- Zoom in on `src/tir/transforms`.
- **Objective:** Dissect a single core optimization technique (like **Tiling** or **Operator Fusion**) as if examining it under a microscope.

---
*This documentation is served via MkDocs and acts as a living journal of my learning process.*
