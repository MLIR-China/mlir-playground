# Building WasmCompiler

Source and build scripts for generating the clang & wasm-ld js modules for the WasmCompiler component.

## Build Instructions

### Docker Setup

0. (Make sure you're in the `cpp/` directory).
1. Run `build-docker.sh` to create a docker image tagged "clang-wasm".

### Generate JS Modules

0. (Make sure you're in the `cpp/` directory).
1. Run `build-compiler-in-docker.sh` to use the "clang-wasm" image to generate the clang & wasm-ld js modules. The script will handle generating the result files into the correct places in the project root.
