#!/bin/bash

# Use emscripten's file_packager tool to package all the necessary includes and libraries from emscripten's sysroot.
# Each invocation creates two files in the current working directory: *.js, *.data.
# Uses lz4 compression, which means the emscripten module that loads this package also needs to have been built with lz4 support (use -s LZ4=1)

/emsdk/upstream/emscripten/tools/file_packager.py onlyincludes.data --js-output=onlyincludes.js --preload /emsdk/upstream/emscripten/cache/sysroot/@include /emsdk/upstream/lib/clang/15.0.0/include@clang-15 --lz4 --no-node --from-emcc
/emsdk/upstream/emscripten/tools/file_packager.py onlylibs.data --js-output=onlylibs.js --preload /emsdk/upstream/emscripten/cache/sysroot/@lib --lz4 --no-node --from-emcc

# Need to then inject this into clang.mjs or wasm-ld.mjs before the line `var moduleOverrides = Object.assign({}, Module);`