*[AST Mutator]: An Object-Oriented C++ Class utilizing the Visitor Design Pattern that traverses the AST and mutates nodes.
*[Intrinsic Node]: A built-in primitive node within the compiler that holds special, hard-coded semantic meaning.
*[call_tir]: An intrinsic node that halts graph-level execution and drops down to C++ TensorIR execution.
*[Operator Fusion]: The process of grouping independent graph-level operations into a single logical execution boundary.
*[Loop Tiling]: The scheduling technique of dividing large spatial loops into smaller, fixed-size chunks (e.g., 16x16 tiles) to fit L1 cache.
*[PackedFunc]: TVM's type-erased FFI mechanism that packs Python arguments into a C-union array, avoiding serialization.
