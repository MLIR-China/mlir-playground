#!/bin/bash

# Script for building the WASM & the website locally

cd ../cpp
./build-docker.sh
./build-compiler-in-docker.sh
cd ..
npm install
npm run build
