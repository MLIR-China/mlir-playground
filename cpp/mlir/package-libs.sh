#!/bin/bash

# Use emscripten's file_packager tool to package all the necessary includes and libraries from emscripten's sysroot.
# Each invocation creates two files in the current working directory: *.js, *.data.
# Uses lz4 compression, which means the emscripten module that loads this package also needs to have been built with lz4 support (use -s LZ4=1)

/emsdk/upstream/emscripten/tools/file_packager.py onlyincludes.data --js-output=onlyincludes.js --preload /emsdk/upstream/emscripten/cache/sysroot/include@include /emsdk/upstream/lib/clang/15.0.0/include@clang-15 --lz4 --no-node --from-emcc

# Selectively package libs based on what's needed.
# TODO: Use a lazy file system to fetch individual libraries on demand.
LIB_DIR=/emsdk/upstream/emscripten/cache/sysroot/lib
mkdir -p tmp/wasm32-emscripten
cp $LIB_DIR/libLLVM*.a tmp/
cp $LIB_DIR/libMLIR*.a tmp/
cp $LIB_DIR/wasm32-emscripten/*.a tmp/wasm32-emscripten
rm tmp/wasm32-emscripten/lib*-asan*.a
rm tmp/wasm32-emscripten/lib*-except*.a
rm tmp/wasm32-emscripten/lib*-mt*.a

/emsdk/upstream/emscripten/tools/file_packager.py onlylibs.data --js-output=onlylibs.js --preload tmp@lib --lz4 --no-node --from-emcc

rm -r tmp

# Need to then inject this into clang.mjs or wasm-ld.mjs before the line `var moduleOverrides = Object.assign({}, Module);`
