#!/bin/bash

set -e

# Run inside docker to get these three emscripten packages:
# 1. clang: compiler executable
# 2. wasm-ld: linker executable
# 3. libs: sysroot file package

source /app/docker/common.sh
mkdir /app/build/bin -p
cd /app/build/bin

# Step 1: Copy clang and wasm-ld build results
cp $LLVM_WASM_BUILD/bin/clang.js-13 ./clang_raw.mjs
cp $LLVM_WASM_BUILD/bin/clang.wasm ./clang.wasm
cp $LLVM_WASM_BUILD/bin/lld.js ./wasm-ld_raw.mjs
cp $LLVM_WASM_BUILD/bin/lld.wasm ./lld.wasm

# Step 2: Create sysroot packages inside build directory
source /app/mlir/package-libs.sh

# Step 3: Insert sysroot package loaders into corresponding js modules
sed -e '/Module = Module || {};/r onlyincludes.js' clang_raw.mjs > clang.mjs
sed -e '/Module = Module || {};/r onlylibs.js' wasm-ld_raw.mjs > wasm-ld.mjs
