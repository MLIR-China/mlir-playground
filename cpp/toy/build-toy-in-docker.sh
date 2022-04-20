#!/bin/bash

set -e

docker run --rm -ti -w /app/ -v $(pwd):/app/ clang-wasm bash -c "./build-toy.sh"

# copy results
cp build/bin/toyc-ch7.js ../../components/Toy/
cp build/bin/toyc-ch7.wasm ../../components/Toy/
cp build/bin/toyc-ch7.data ../../public/
