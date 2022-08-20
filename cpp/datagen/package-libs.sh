#!/bin/bash

# Use emscripten's file_packager tool to package all the necessary includes and libraries from emscripten's sysroot.
# Each invocation creates two files in the current working directory: *.js, *.data.
# Uses lz4 compression, which means the emscripten module that loads this package also needs to have been built with lz4 support (use -s LZ4=1)

/emsdk/upstream/emscripten/tools/file_packager.py onlyincludes.data --js-output=onlyincludes.js --preload /emsdk/upstream/emscripten/cache/sysroot/include@include /emsdk/upstream/lib/clang/15.0.0/include@clang-15 --lz4 --no-node --from-emcc

# Selectively package libs based on what's needed.
# TODO: Use a lazy file system to fetch individual libraries on demand.
LIB_DIR=/emsdk/upstream/emscripten/cache/sysroot/lib

SYSTEM_LIB_NAMES=(
    "GL"
    "al"
    "html5"
    "stubs-debug"
    "noexit"
    "c-debug"
    "dlmalloc"
    "compiler_rt"
    "c++-noexcept"
    "c++abi-noexcept"
    "sockets"
)

mkdir -p tmp/wasm32-emscripten
cp $LIB_DIR/libLLVM*.a tmp/
cp $LIB_DIR/libMLIR*.a tmp/

LLVM_LIB_NAMES=($(find tmp/ -maxdepth 1 -type f -printf "%f\n"))

for libname in "${SYSTEM_LIB_NAMES[@]}"
do
    fullname="lib${libname}.a"
    cp $LIB_DIR/wasm32-emscripten/$fullname tmp/wasm32-emscripten/$fullname
    echo $LIB_DIR/wasm32-emscripten/$fullname
done

/emsdk/upstream/emscripten/tools/file_packager.py onlylibs.data --js-output=onlylibs.js --preload tmp@lib --lz4 --no-node --from-emcc

# Append to metadata file
CONSTANTS_FILENAME=$1

printf "export const SYSTEM_LIB_NAMES = [\n" >> $CONSTANTS_FILENAME
printf "  \"%s\",\n" "${SYSTEM_LIB_NAMES[@]}" >> $CONSTANTS_FILENAME
printf "];\n\n" >> $CONSTANTS_FILENAME

printf "export const LLVM_LIB_FILES = [\n" >> $CONSTANTS_FILENAME
printf "  \"/lib/%s\",\n" "${LLVM_LIB_NAMES[@]}" >> $CONSTANTS_FILENAME
printf "];\n\n" >> $CONSTANTS_FILENAME

# Cleanup
rm -r tmp
