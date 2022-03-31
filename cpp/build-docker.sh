#!/bin/bash

pushd docker
docker build -t clang-wasm .
popd
