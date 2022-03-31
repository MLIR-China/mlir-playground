#!/bin/bash

# basic configurations
LLVM_ROOT=/opt/llvm
NATIVE_GENERATORS="llvm-tblgen clang-tblgen mlir-tblgen mlir-linalg-ods-gen mlir-linalg-ods-yaml-gen"

# derived configurations
LLVM_SRC="$LLVM_ROOT/llvm-project"
LLVM_NATIVE_BUILD="$LLVM_ROOT/llvm-native-build"

mkdir -p $LLVM_ROOT && cd $LLVM_ROOT

# clone llvm-project
git clone https://github.com/llvm/llvm-project.git --branch llvmorg-13.0.1 --depth 1

# build native versions of generators
cmake -G Ninja \
    -S $LLVM_SRC/llvm/ \
    -B $LLVM_NATIVE_BUILD/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=WebAssembly \
    -DLLVM_ENABLE_PROJECTS="clang;mlir"
cmake --build $LLVM_NATIVE_BUILD/ -- $NATIVE_GENERATORS
