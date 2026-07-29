import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

@I.ir_module
class ConvAddModule:
    @R.function
    def main(data: R.Tensor((1, 3, 224, 224), dtype="float32"),
             weight: R.Tensor((16, 3, 3, 3), dtype="float32"),
             bias: R.Tensor((16, 1, 1), dtype="float32")):
        with R.dataflow():
            conv = R.nn.conv2d(data, weight, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
            out = R.add(conv, bias)
            relu = R.nn.relu(out)
            R.output(relu)
        return relu

print("--- 1. Original Module ---")
ConvAddModule.show()

print("\n--- 2. After LegalizeOps ---")
mod_legalized = relax.transform.LegalizeOps()(ConvAddModule)
mod_legalized.show()

print("\n--- 3. After FuseOps (Graph-level Grouping) ---")
mod_fused = relax.transform.AnnotateTIROpPattern()(mod_legalized)
mod_fused = relax.transform.FuseOps()(mod_fused)
mod_fused.show()

print("\n--- 4. After FuseTIR (TensorIR Generation) ---")
mod_tir = relax.transform.FuseTIR()(mod_fused)
mod_tir.show()
