# Build & Compilation Troubleshooting

Building complex C++ projects like TVM from source rarely goes perfectly on the first try. In this phase, we encounter a classic dependency issue, debug it, and successfully compile the source.

## The Homebrew `prefix` Trap

We attempted to run the initial CMake configuration:
```bash
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

However, CMake immediately failed with the following error:
```text
CMake Error at cmake/utils/FindLLVM.cmake:87 (message):
  Fatal error executing: /opt/homebrew/opt/llvm/bin/llvm-config --libfiles
```

### Investigation
During the initial environment diagnostic, we ran `brew --prefix llvm` which returned `/opt/homebrew/opt/llvm` with an exit code of `0`. We assumed LLVM was installed. \
However, `brew info llvm` revealed the truth: **LLVM was not actually installed!**

!!! warning
    **Homebrew Gotcha**<br>
    The `brew --prefix <formula>` command will return the *expected* path for a formula regardless of whether it is currently installed on the system. Always verify installation with `brew info` or by checking if the binary actually exists.

### Resolution
We resolved this by installing LLVM properly via Homebrew:
```bash
brew install llvm
```

## Compilation

With LLVM correctly installed and `config.cmake` pointing to `/opt/homebrew/opt/llvm/bin/llvm-config`, CMake configuration succeeds.

We then triggered the parallel build utilizing all available CPU cores:
```bash
make -j$(nproc)
```

### The `autoconf` White-Space Trap (libbacktrace)

At 94% completion, the compilation abruptly failed with a cryptic `make` error. Digging into the logs revealed the true culprit inside one of the submodules:
```text
checking whether build environment is sane... configure: error: unsafe srcdir value: '~/Developer/LEARN/26.07.27 -- TVM/tvm/3rdparty/tvm-ffi/cmake/Utils/../../3rdparty/libbacktrace'
```

**What happened?**<br>
The TVM repository was cloned inside a workspace folder named `26.07.27 -- TVM`. The `libbacktrace` submodule uses `autoconf` (the `configure` script), which **strictly prohibits spaces** in absolute file paths. Because our workspace directory name contains spaces, `configure` crashed, halting the entire TVM build.

### Resolution
Moving the entire workspace to a path without spaces is one solution, but sometimes we don't have control over upstream folder names. Since `libbacktrace` is only used to provide nice C++ stack traces upon segfaults, we can safely disable it in `config.cmake` to bypass the `autoconf` step entirely.

```cmake
# Disabled in build/config.cmake to avoid space-in-path issues
set(TVM_FFI_USE_LIBBACKTRACE OFF)
```

After updating the configuration, we re-ran CMake and resumed the compilation:
```bash
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j$(nproc)
```

*(Build resumed and successfully reached 100% - `Built target tvm_compiler`)*

## Python Bindings and a Second White-Space Trap

After completing the C++ build, the next step in the TVM system blueprint is to test the Python FFI boundary. TVM now extensively uses a separate `apache-tvm-ffi` Python package which must be installed alongside the main TVM python package. 

When attempting to install this using pip (`pip install -e tvm/3rdparty/tvm-ffi`), we were hit by the exact same `autoconf` white-space error again! This is because the Python package triggers its own internal CMake build for `tvm-ffi`, which ignored our top-level `config.cmake` override.

### Resolution
To completely eradicate the issue, we hardcoded the default option to `OFF` directly inside the `tvm-ffi` module:
Modified `tvm/3rdparty/tvm-ffi/CMakeLists.txt`:
```cmake
- option(TVM_FFI_USE_LIBBACKTRACE "Enable libbacktrace" ON)
+ option(TVM_FFI_USE_LIBBACKTRACE "Enable libbacktrace" OFF)
```

With this fixed, we installed the necessary Python dependencies (`numpy`, `scipy`, `decorator`, `attrs`) and the TVM packages, and verified the installation:
```bash
export TVM_HOME="$(pwd)/tvm"
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
python3 -c "import tvm; print('TVM Compiled Successfully! TVM Version:', tvm.__version__)"
```

**Success!** The terminal returned `TVM Compiled Successfully! TVM Version: 0.26.dev0`.
