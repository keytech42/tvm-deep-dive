# Environment Setup & TVM Source Clone

This entry logs the exact environment state and steps taken to clone and configure the Apache TVM project from source.

## System Diagnostics

Before building TVM, it is crucial to understand the host architecture, especially on macOS where paths differ between Intel and Apple Silicon (ARM64).

- **OS**: macOS 26.5.2
- **Architecture**: `arm64` (Apple Silicon)
- **CMake Version**: 4.4.0
- **Python Version**: 3.14.6
- **LLVM Path**: `/opt/homebrew/opt/llvm`

Since this is an ARM64 Mac, Homebrew installs LLVM to `/opt/homebrew/` instead of `/usr/local/`. This path must be explicitly provided to TVM's CMake configuration.

## Fetching the Source (Reproducibility)

TVM was cloned recursively to fetch all third-party submodules (like `cutlass`, `libflash_attn`, etc.).

```bash
git clone --recursive https://github.com/apache/tvm.git tvm
```

> [!NOTE]
> **Version Pinning**
> To ensure reproducibility of this study and the code referenced in these notes, the exact commit cloned was:
> `a104a7b0a299103d1e910debcbe63aeafcea045f`
> 
> You can view the exact state of the repository at this commit here:
> [apache/tvm @ a104a7b](https://github.com/apache/tvm/tree/a104a7b0a299103d1e910debcbe63aeafcea045f)

*(As discussed, the `tvm` folder is added to `.gitignore` to keep this documentation repository lightweight. Re-cloning and checking out this specific commit will perfectly restore the build environment.)*

## CMake Configuration (`config.cmake`)

TVM uses an explicit configuration file (`build/config.cmake`) rather than passing all arguments via CLI. 

Following the TVM System Core Guide, we initialized the build directory:
```bash
mkdir -p build
cp cmake/config.cmake build/config.cmake
```

### LLVM Engine Setup

Because we are on an M-series Mac, we cannot simply use `set(USE_LLVM ON)`. We must point CMake exactly to the Homebrew LLVM binary:

```cmake
# Changed in build/config.cmake
set(USE_LLVM "/opt/homebrew/opt/llvm/bin/llvm-config")
```

This guarantees that the Target Codegen phase properly binds to the LLVM version we installed, avoiding clashes with Apple's default `clang`.

---
*Next step: Generating Makefiles and Executing Parallel Compilation.*
