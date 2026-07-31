*[AST Mutator]: An Object-Oriented C++ Class utilizing the Visitor Design Pattern that traverses the AST and mutates nodes.
*[FFI (Foreign Function Interface)]: A boundary mechanism that allows Python to invoke low-level C++ compiler routines without serialization overhead.
*[Intrinsic Node]: A built-in primitive node within the compiler that holds special, hard-coded semantic meaning.
*[L1 Cache (Shared Memory)]: The extremely fast, physically close memory region on a hardware device with severely limited capacity.
*[Loop Tiling]: The scheduling technique of dividing large spatial loops into smaller, fixed-size chunks (e.g., 16x16 tiles) to fit L1 cache.
*[Operator Fusion]: The process of grouping independent graph-level operations into a single logical execution boundary.
*[PackedFunc]: TVM's type-erased FFI mechanism that packs Python arguments into a C-union array, avoiding serialization.
*[PrimFunc]: A Primitive Function representing a physical, nested loop boundary in TVM's C++ core (TensorIR).
*[VRAM (Global Memory)]: The massive, relatively slow global memory space on a hardware device. Fetching data from VRAM is extremely expensive.
