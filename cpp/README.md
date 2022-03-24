# Cpp sources for building Toy in WASM

Currently using a very hacky way to modify Toy source code inside llvm-project repo.
Future iterations will no longer need this hack once we're compiling our own example.

## Build Instructions

1. Run `build-in-docker.sh` to create `build/bin/toyc-ch7.{js|data|wasm)`.
2. Combine these three files with `static/index.html` and serve with an http server.
3. Click the "Start" button to initialize the page. Click "Compile" to send data to compile the left side, and display result on the right side.
