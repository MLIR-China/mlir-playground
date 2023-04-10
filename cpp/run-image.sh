#!/bin/bash

# Command line argument to override docker image. Default is `clang-wasm`.
image=${1:-clang-wasm:latest}

docker run --rm -ti -w /app/ -v $(pwd):/app/ $image bash
