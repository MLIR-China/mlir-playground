#!/bin/bash

set -e

source ./common.sh

# basic configurations
NATIVE_GENERATORS="llvm-tblgen clang-tblgen mlir-tblgen mlir-linalg-ods-yaml-gen"

mkdir -p $LLVM_ROOT && cd $LLVM_ROOT

# clone llvm-project
git clone https://github.com/llvm/llvm-project.git --branch llvmorg-16.0.0 --depth 1

# build native versions of generators
cmake -G Ninja \
    -S $LLVM_SRC/llvm/ \
    -B $LLVM_NATIVE_BUILD/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=WebAssembly \
    -DLLVM_ENABLE_PROJECTS="clang;lld;mlir"

cmake --build $LLVM_NATIVE_BUILD/ -- $NATIVE_GENERATORS clang lld
