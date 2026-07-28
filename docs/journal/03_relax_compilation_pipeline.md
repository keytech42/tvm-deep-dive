# Journal 03: The Big Picture (Relax Compilation Pipeline)

## 🎯 Objective

Understand the end-to-end pipeline of TVM: How does a Python Deep Learning model (like PyTorch) get converted into a hardware-specific C++ Shared Library (`.so`)?

## 🚨 The Plot Twist: Goodbye Relay, Hello Relax

Initially, our roadmap planned to use the legacy `tvmc` CLI tool (which relied on TVM's older **Relay** frontend). However, upon inspecting our `0.26.dev0` bleeding-edge build, we made a massive discovery: **TVMC and Relay have been completely removed from this fork!**

This repository is purely dedicated to **TVM Unity (Relax)**, the next-generation architecture built specifically for dynamic shapes and Large Language Models (LLMs). This is actually a huge advantage, as we get to bypass legacy code and interact directly with the modern `tvm.relax` API.

## 🛠️ The Pipeline Execution

We executed the pipeline in two steps:

### 1. Generating a Dummy Model (`scripts/01_generate_model.py`)
We created a minimal PyTorch model containing a `Conv2d`, `ReLU`, `Flatten`, and `Linear` layer. We exported this to `dummy_model.onnx`. We used a small dummy model to ensure instant compilation while preserving the exact same architectural pipeline used for massive models like LLaMA.

### 2. Compiling with Relax (`scripts/02_compile_model.py`)
Instead of a black-box CLI tool, we used a simple Python script to compile the model:
1. Load the ONNX model.
2. Ingest it into TVM's internal representation using `relax.frontend.onnx.from_onnx`.
3. Compile it to an Executable using `relax.build`.
4. Export the C++ library to `compiled_model.so`.

## 🔬 Inspecting the TVMScript (The Internal IR)

The most valuable artifact from this process is `dummy_model_tvmscript.txt`. Before compiling to C++, TVM translates the ONNX graph into its own Python-like AST called **TVMScript**.

```python
@I.ir_module
class Module:
    @R.function
    def main(input: R.Tensor(("batch_size", 3, 224, 224), dtype="float32")) -> R.Tensor(("batch_size", 10), dtype="float32"):
        batch_size = T.int64()
        with R.dataflow():
            # Convolution Layer (notice the layout and strides)
            lv: R.Tensor(...) = R.nn.conv2d(input, metadata["relax.expr.Constant"][0], strides=[1, 1], padding=[1, 1, 1, 1], ...)
            
            # Bias Add and ReLU
            lv2: R.Tensor(...) = R.add(lv, lv1)
            lv3: R.Tensor(...) = R.nn.relu(lv2)
            
            # Matmul (Linear Layer)
            lv6: R.Tensor(...) = R.matmul(lv4, lv5, out_dtype=None)
            gv: R.Tensor(...) = R.add(lv6, metadata["relax.expr.Constant"][3])
            R.output(gv)
        return gv
```

### Key Takeaways from the TVMScript:
1. **Dynamic Shapes are First-Class**: Notice `"batch_size"`. Unlike older frameworks that freeze dimensions, Relax natively tracks symbolic dimensions (`T.int64()`) throughout the entire graph.
2. **Dataflow Block**: The `with R.dataflow():` context means these operations have no side effects. The compiler can aggressively reorder, fuse, or parallelize anything inside this block safely.
3. **Hardware Agnostic to Hardware Specific**: This high-level IR (Relax) will eventually be lowered into TensorIR (TIR) loops and then to LLVM C++ code inside the `compiled_model.so`.

## ✅ Conclusion

We successfully witnessed the "Big Picture". A PyTorch model went in, was translated into TVM's internal **Relax IR**, and was ultimately compiled out as a `compiled_model.so` native library. 

**Next Step:** Now that we've seen the very top (Python API) and the very bottom (`.so`), it's time to explore the bridge that connects them: the **FFI Boundary Tracking (PackedFunc)**.
