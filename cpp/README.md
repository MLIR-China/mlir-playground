# POC for building Toy in WASM

The current project uses source code patches to directly modify the Toy example inside a llvm-project repo.
Future iterations will no longer need this workaround once we're compiling our own code, not Toy.

## Build Instructions

### Docker Setup

0. (Make sure you're in the `cpp` directory).
1. Run `build-docker.sh` to create a docker image tagged "clang-wasm".

### Compile Toy to WASM

0. (Make sure you're in the `cpp` directory).
1. Run `build-in-docker.sh` to create `build/bin/toyc-ch7.{js|data|wasm)`.
2. Combine these three files with `shell/index.html` and serve with an http server.
3. Click the "Start" button to initialize the page. The first text box will display the source code.
Click the "Compile" button to compile the source code. The second text box will display the compiled results.
