#!/bin/bash

# Run build-compiler.sh in docker and copy the files to the correct location in the project.

set -e

# Command line argument to override docker image. Default is `clang-wasm`.
image=${1:-clang-wasm}

# Run script inside docker.
docker run --rm -w /app/ -v $(pwd):/app/ $image bash -c "./build-compiler.sh"

# Copy results to where they're needed.
cp build/bin/clang.mjs ../components/WasmCompiler/wasm/clang.mjs
cp build/bin/clang.wasm ../public/clang.wasm

cp build/bin/wasm-ld.mjs ../components/WasmCompiler/wasm/wasm-ld.mjs
cp build/bin/lld.wasm ../public/lld.wasm

cp build/bin/onlyincludes.data ../public/onlyincludes.data
cp build/bin/onlylibs.data ../public/onlylibs.data
