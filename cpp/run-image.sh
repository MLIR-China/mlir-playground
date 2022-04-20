#!/bin/bash
docker run --rm -ti -w /app/ -v $(pwd):/app/ clang-wasm bash
