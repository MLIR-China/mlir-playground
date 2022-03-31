#!/bin/bash

APP_ROOT=/app/
LLVM_ROOT=/opt/llvm

# find directory
LLVM_SRC="$LLVM_ROOT/llvm-project"
LLVM_NATIVE_BUILD="$LLVM_ROOT/llvm-native-build"
LLVM_WASM_BUILD="$APP_ROOT/build"

# apply patches
pushd $LLVM_SRC
git apply $APP_ROOT/patches/mlir.patch
git apply $APP_ROOT/patches/toyc7.patch
git apply $APP_ROOT/patches/toyc7-cmake.patch
popd

# build toy
CXXFLAGS="-Dwait4=__syscall_wait4 -stdlib=libc++" \
LDFLAGS="-s ENVIRONMENT='web' -s EXPORT_ES6=1 -s MODULARIZE=1 -s LLD_REPORT_UNDEFINED=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=_main,_free,_malloc -s EXPORTED_RUNTIME_METHODS=FS,PROXYFS,allocateUTF8 -lproxyfs.js" \
emcmake cmake -G Ninja \
    -S $LLVM_SRC/llvm/ \
    -B $LLVM_WASM_BUILD/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=WebAssembly \
    -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
    -DLLVM_ENABLE_DUMP=OFF \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_EXPENSIVE_CHECKS=OFF \
    -DLLVM_ENABLE_BACKTRACES=OFF \
    -DLLVM_ENABLE_THREADS=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TABLEGEN=$LLVM_NATIVE_BUILD/bin/llvm-tblgen \
    -DCLANG_TABLEGEN=$LLVM_NATIVE_BUILD/bin/clang-tblgen \
    -DMLIR_TABLEGEN=$LLVM_NATIVE_BUILD/bin/mlir-tblgen \
    -DMLIR_LINALG_ODS_GEN=$LLVM_NATIVE_BUILD/bin/mlir-linalg-ods-gen \
    -DMLIR_LINALG_ODS_YAML_GEN=$LLVM_NATIVE_BUILD/bin/mlir-linalg-ods-yaml-gen \

cmake --build $LLVM_WASM_BUILD/ -- toyc-ch7
