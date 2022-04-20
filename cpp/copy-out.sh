#!/bin/bash

set -e

mkdir -p build

IMAGE_ID=$(docker create clang-wasm)
docker cp $IMAGE_ID:/opt/llvm/llvm-wasm-build/bin/ build
docker rm -v $IMAGE_ID
