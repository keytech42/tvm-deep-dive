import tvm
from tvm import relax

print("--- 1. Getting the C++ PackedFunc ---")
# Get the global function registered in C++
conv2d_cpp = tvm.get_global_func("relax.op.nn.conv2d")
print("Python Type:", type(conv2d_cpp))
print("Object:", conv2d_cpp)

print("\n--- 2. Examining the Python Wrapper ---")
print("Python Type:", type(relax.op.nn.conv2d))
print("Object:", relax.op.nn.conv2d)

print("\n--- 3. Proving Zero-Copy Execution ---")
# Create some dummy TVM variables
data = relax.Var("data", relax.TensorType([1, 3, 224, 224], "float32"))
weight = relax.Var("weight", relax.TensorType([16, 3, 3, 3], "float32"))

# Call the C++ PackedFunc directly
conv2d_node = conv2d_cpp(data, weight, (1, 1), (0, 0, 0, 0), (1, 1), 1, "NCHW", "OIHW", None, None)
print("Result Node Type:", type(conv2d_node))
print("Result Node:\n", conv2d_node)
