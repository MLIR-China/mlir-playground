#!/bin/bash

set -e

# Run inside docker to get these three emscripten packages:
# 1. clang: compiler executable
# 2. wasm-ld: linker executable
# 3. libs: sysroot file package

source /app/docker/common.sh
mkdir /app/build/bin -p
cd /app/build/bin

# Step 0: Create constant metadata file and insert llvm version
CONSTANTS_FILENAME="constants.js"
> $CONSTANTS_FILENAME

pushd $LLVM_SRC
LLVM_GIT_TAG=$(git describe --exact-match)
popd

printf "export const LLVM_VERSION = \"$LLVM_GIT_TAG\";\n\n" >> $CONSTANTS_FILENAME

# Step 1: Copy clang and wasm-ld build results
cp $LLVM_WASM_BUILD/bin/clang.js-13 ./clang_raw.mjs
cp $LLVM_WASM_BUILD/bin/clang.wasm ./clang.wasm
cp $LLVM_WASM_BUILD/bin/lld.js ./wasm-ld_raw.mjs
cp $LLVM_WASM_BUILD/bin/lld.wasm ./lld.wasm

# Step 2: Create sysroot packages inside build directory
/app/datagen/package-libs.sh $CONSTANTS_FILENAME

# Step 3: Insert sysroot package loaders into corresponding js modules
sed -e '/Module = Module || {};/r onlyincludes.js' clang_raw.mjs > clang.mjs
sed -e '/Module = Module || {};/r onlylibs.js' wasm-ld_raw.mjs > wasm-ld.mjs

# Step 4: Remove hardcoded wasm file location expectations
sed -i -E 's/\{wasmBinaryFile=new URL[^\}]+\}/{throw "must implement locateFile method on Module."}/' clang.mjs
sed -i -E 's/\{wasmBinaryFile=new URL[^\}]+\}/{throw "must implement locateFile method on Module."}/' wasm-ld.mjs

# Step 5: Remove typeof window check to work around erroneous JS optimization for web workers
sed -i -E "s/typeof window === 'object'/false/" clang.mjs
sed -i -E "s/typeof window === 'object'/false/" wasm-ld.mjs

# Step 6: Export all toy chapter builds as examples
mkdir /app/build/bin/toy -p
cd /app/build/bin/toy

for chapter_idx in {1..7}
do
    cp "${LLVM_WASM_BUILD}/bin/toyc-ch${chapter_idx}.wasm" .
    cp "${LLVM_WASM_BUILD}/bin/toyc-ch${chapter_idx}.js" .
    sed -i -E 's/\{wasmBinaryFile=new URL[^\}]+\}/{throw "must implement locateFile method on Module."}/' "toyc-ch${chapter_idx}.js"
done
