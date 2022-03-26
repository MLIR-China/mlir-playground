#!/bin/bash

set -e

docker run --rm -ti -w /app/ -v $(pwd):/app/ clang-wasm bash -c "./build.sh"

# copy results
cp build/bin/toyc-ch7.js ../pages/
cp build/bin/toyc-ch7.wasm ../pages/
cp build/bin/toyc-ch7.data ../public/
