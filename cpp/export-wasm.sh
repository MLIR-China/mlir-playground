#!/bin/bash

# Run build-compiler.sh in docker and copy the files to the correct location in the project.

set -e

# Command line argument to override docker image. Default is `clang-wasm`.
image=${1:-ghcr.io/mlir-china/clang-wasm:latest}

# Run script inside docker.
docker run --rm -w /app/ -v $(pwd):/app/ $image bash -c "./wasmgen/package-all.sh"

# Copy results to where they're needed.
BUILD_DIR=wasmgen/build/bin

mkdir -p ../public/wasm

cp $BUILD_DIR/clang.mjs ../components/WasmCompiler/wasm/clang.mjs
cp $BUILD_DIR/clang.wasm ../public/wasm/clang.wasm

cp $BUILD_DIR/wasm-ld.mjs ../components/WasmCompiler/wasm/wasm-ld.mjs
cp $BUILD_DIR/lld.wasm ../public/wasm/lld.wasm

cp $BUILD_DIR/mlir-tblgen.mjs ../components/MlirTblgen/wasm/mlir-tblgen.mjs
cp $BUILD_DIR/mlir-tblgen.wasm ../public/wasm/mlir-tblgen.wasm

cp $BUILD_DIR/onlyincludes.data ../public/wasm/onlyincludes.data
cp $BUILD_DIR/onlylibs.data ../public/wasm/onlylibs.data

cp $BUILD_DIR/constants.js ../components/WasmCompiler/wasm/constants.js

cp $BUILD_DIR/toy/*.js ../components/Toy/wasm/
cp $BUILD_DIR/toy/*.wasm ../public/wasm/

cp $BUILD_DIR/template.js ../components/WasmCompiler/wasm/template.js
