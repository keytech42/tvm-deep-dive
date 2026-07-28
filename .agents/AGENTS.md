# TVM System Engineering Portfolio Context

This project is a deep-dive reverse engineering and learning portfolio for Apache TVM.

## Ultimate Goal

To master TVM and build an elite junior engineer portfolio by taking a "top-down" reverse engineering approach: starting from the CLI, tracing through the Python frontend, crossing the FFI boundary, and finally dissecting C++ TensorIR optimizations (Tiling, Fusion).

## AI Agent Instructions (CRITICAL)

When you (the AI) start a new session in this workspace, ALWAYS follow these rules and refer to this roadmap:

### 1. Understand the Roadmap (The 3-Step Strategy + FFI)
- **[x] Step 0: The Foundation**: Clone TVM and build it from source (Completed: C++ Core & Python FFI bindings are built and working).
- **[x] Step 1: The Big Picture (Relax)**: Compiling an ONNX model to `.so` via TVM Relax API.
- **[ ] Step 2: Python Frontend Tracing**: Dive into `python/tvm/relax` or `relay`. Goal: Trace how familiar Python operations (e.g., `Conv2d`) are ingested into TVM's internal graph nodes.
- **[ ] Step 3: FFI Boundary Tracking (PackedFunc)**: Goal: Master the `PackedFunc` mechanism to understand how TVM transitions from Python to C++ with zero serialization overhead.
- **[ ] Step 4: C++ Core (TensorIR)**: Zoom in on `src/tir/transforms`. Goal: Dissect a single core optimization technique (like Tiling or Operator Fusion).

### 2. Environment Setup
Before running any TVM Python scripts or tests, you MUST activate the environment, install dependencies, and set paths:
```bash
source .venv/bin/activate
pip install -r requirements.txt
export TVM_HOME="$(pwd)/tvm"
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### 3. Documentation
Every time you complete a significant step, document the journey in the MkDocs `docs/journal/` folder. The primary audience is human readers (interviewers, recruiters, or learners). Write detailed, professional markdown explanations and ensure `mkdocs.yml` is updated.
