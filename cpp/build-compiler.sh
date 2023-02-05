#!/bin/bash

set -e

# Run inside docker to get these three emscripten packages:
# 1. clang: compiler executable
# 2. wasm-ld: linker executable
# 3. libs: sysroot file package

source /app/docker/common.sh
mkdir /app/build/bin -p
cd /app/build/bin

# Step 0: Create constant metadata file and insert llvm version
CONSTANTS_FILENAME="constants.js"
> $CONSTANTS_FILENAME

pushd $LLVM_SRC
LLVM_GIT_TAG=$(git describe --exact-match)
popd

printf "export const LLVM_VERSION=\"$LLVM_GIT_TAG\";\n\n" >> $CONSTANTS_FILENAME

# Create sysroot packages inside build directory
/app/datagen/package-libs.sh $CONSTANTS_FILENAME

# Export a generated wasm module from the LLVM build directory into the local build directory.
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
        local insertion_after_pattern='function(Module = {})  {'
        if ! grep -q "$insertion_after_pattern" $dst_js_raw; then
            echo "Failed to find replacement pattern '$insertion_before_pattern' in JS"
            exit 1
        fi
        sed -e "/$insertion_after_pattern/r $fs_js" $dst_js_raw > $dst_js
        rm $dst_js_raw
    fi

    # Step 3: Remove hardcoded wasm file location expectations
    sed -i -E 's/\{wasmBinaryFile=new URL[^\}]+\}/{throw "must implement locateFile method on Module."}/' $dst_js

    # Step 4: Remove typeof window check to work around erroneous JS optimization for web workers
    sed -i -E "s/typeof window === 'object'/false/" $dst_js
}

# Export clang & lld
export_generated_wasm clang.js-13 clang.wasm clang.mjs clang.wasm onlyincludes.js
export_generated_wasm lld.js lld.wasm wasm-ld.mjs lld.wasm onlylibs.js

# Export mlir-tblgen
export_generated_wasm mlir-tblgen.js mlir-tblgen.wasm mlir-tblgen.mjs mlir-tblgen.wasm onlyincludes.js

# Export all toy chapter builds as examples
mkdir /app/build/bin/toy -p
cd /app/build/bin/toy

for chapter_idx in {1..7}
do
    js_name="toyc-ch${chapter_idx}.js"
    wasm_name="toyc-ch${chapter_idx}.wasm"
    export_generated_wasm $js_name $wasm_name $js_name $wasm_name
done
