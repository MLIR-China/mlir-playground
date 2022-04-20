#!/bin/bash

set -e

if [[ -z $LLVM_ROOT ]]
then
    echo "Error: common.sh requires LLVM_ROOT to be set"
    exit 1
fi

# derived configurations
export LLVM_SRC="$LLVM_ROOT/llvm-project"
export LLVM_NATIVE_BUILD="$LLVM_ROOT/llvm-native-build"
export LLVM_WASM_BUILD="$LLVM_ROOT/llvm-wasm-build"
