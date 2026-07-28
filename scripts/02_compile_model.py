import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

def main():
    print("1. Loading ONNX model...")
    onnx_model = onnx.load("dummy_model.onnx")
    
    print("2. Ingesting ONNX into TVM Relax IRModule...")
    # Convert ONNX to TVM's internal representation (IRModule)
    tvm_mod = from_onnx(onnx_model)
    
    # Save the human-readable TVMScript for inspection
    script_path = "dummy_model_tvmscript.txt"
    with open(script_path, "w") as f:
        f.write(tvm_mod.script())
    print(f"   -> Saved internal TVMScript representation to {script_path}")
    
    print("3. Compiling the IRModule for LLVM (CPU)...")
    target = tvm.target.Target("llvm")
    
    # In newer TVM versions, we use relax.build to generate the Executable
    ex = relax.build(tvm_mod, target=target)
    
    print("4. Exporting to C++ Shared Library (.so)...")
    out_path = "compiled_model.so"
    ex.export_library(out_path)
    print(f"   -> Successfully compiled and saved to {out_path}")

if __name__ == "__main__":
    main()
