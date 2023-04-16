#!/bin/bash

set -e

# Run this script inside a clang-wasm docker container to package and fetch
# the necessary emscripten-generated wasm files needed by the web app.

BUILD_DIR=/app/wasmgen/build/bin

source /app/docker/common.sh
mkdir $BUILD_DIR -p
cd $BUILD_DIR

###############################################################################
#  1. Create constant metadata file and insert llvm version.
###############################################################################
CONSTANTS_FILENAME="constants.js"
> $CONSTANTS_FILENAME

pushd $LLVM_SRC
LLVM_GIT_TAG=$(git describe --exact-match)
popd

printf "export const LLVM_VERSION=\"$LLVM_GIT_TAG\";\n\n" >> $CONSTANTS_FILENAME

###############################################################################
#  2. Create sysroot and llvm/mlir library packages.
###############################################################################
/app/wasmgen/package-libs.sh $CONSTANTS_FILENAME

###############################################################################
#  3. Fetch and patch the emscripten-compiled clang & lld executables.
#     Also fetch the pre-compiled toy executables.
###############################################################################
# Export a generated wasm module from the LLVM build directory into the local
# build directory.
# Arguments:
#   $1 - Source JS file name
#   $2 - Source WASM file name
#   $3 - Dest JS file name
#   $4 - Dest WASM file name
#   $5 - Packaged data file loader path (if empty, don't package any files)
function export_generated_wasm() {
    local src_js="$LLVM_WASM_BUILD/bin/$1"
    local src_wasm="$LLVM_WASM_BUILD/bin/$2"
    local dst_js=$3
    local dst_wasm=$4
    local fs_js=$5

    # Step 1: Copy clang and wasm-ld build results
    local dst_js_raw="$dst_js-raw"
    cp $src_js $dst_js_raw
    cp $src_wasm $dst_wasm

    # Step 2: Insert sysroot package loaders into corresponding js modules
    if [ -z $fs_js ]; then
        mv $dst_js_raw $dst_js
    else
        local insertion_after_pattern='var Module = config || {};'
        if ! grep -q "$insertion_after_pattern" $dst_js_raw; then
            echo "Failed to find replacement pattern '$insertion_before_pattern' in JS"
            exit 1
        fi
        sed -e "/$insertion_after_pattern/r $fs_js" $dst_js_raw > $dst_js
        rm $dst_js_raw
    fi

    # Step 3: Remove hardcoded wasm file location expectations
    sed -i -E 's/\{wasmBinaryFile=new URL[^\}]+\}/{throw "must implement locateFile method on Module.";}/' $dst_js

    # Step 4: Remove typeof window check to work around erroneous JS optimization for web workers
    sed -i -E "s/typeof window === 'object'/false/" $dst_js
}

# Export clang & lld
export_generated_wasm clang.js clang.wasm clang.mjs clang.wasm onlyincludes.js
export_generated_wasm lld.js lld.wasm wasm-ld.mjs lld.wasm onlylibs.js

# Export mlir-tblgen
export_generated_wasm mlir-tblgen.js mlir-tblgen.wasm mlir-tblgen.mjs mlir-tblgen.wasm onlyincludes.js

# Export all toy chapter builds as examples
mkdir $BUILD_DIR/toy -p
pushd $BUILD_DIR/toy

for chapter_idx in {1..5}
do
    js_name="toyc-ch${chapter_idx}.js"
    wasm_name="toyc-ch${chapter_idx}.wasm"
    export_generated_wasm $js_name $wasm_name $js_name $wasm_name
done
popd

###############################################################################
#  4. Generate emscripten JS glue template for instantiating wasm-compiled
#     modules in the web-app. (Not a foolproof solution, but in most cases
#     will be good enough).
###############################################################################
# Generate js driver template
pushd /app/wasmgen/template
./compile-template.sh
popd

cp /app/wasmgen/template/build/MlirOptTemplate.js template.js
# Remove hardcoded wasm file location expectations
sed -i -E 's/wasmBinaryFile = new URL.+;/throw "must implement locateFile method on Module.";/' template.js

###############################################################################
#  5. Calculate checksum for entire build/bin dir and append to constants file
###############################################################################
LLVM_PACKAGE_CHECKSUM=$(find $BUILD_DIR -type f -exec sha256sum {} + | sort | sha256sum | cut -f 1 -d ' ')
printf "export const LLVM_PACKAGE_CHECKSUM = \"$LLVM_PACKAGE_CHECKSUM\";\n\n" >> $CONSTANTS_FILENAME
