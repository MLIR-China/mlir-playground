#!/bin/bash

pushd docker
docker build -t clang-wasm . --network host
popd
