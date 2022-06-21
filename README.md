# MLIR Playground

Play with [MLIR](https://mlir.llvm.org/) directly in your browser.

- **Get Started Immediately**: No need to install any dependencies or setup a build system.
- **Private & Secure**: Everything is run locally in your browser. No code or input is transmitted back to the server.

Try it out now at our production site: [MLIR Playground](https://playground.mlir-china.org/).

## Features

### Custom mlir-opt

Create & run your own mlir-opt C++ program.

Using the built-in C++ editor, you can implement and register your own pass (or any logic, for that matter). All MLIR & LLVM libraries in the [LLVM codebase](https://github.com/llvm/llvm-project) are available to use directly.

To compile & run the program, hit the `Run` button above the editor. This will compile your C++ program using clang & lld (compiled to web assembly and run in your browser), and execute the program with the user-provided MLIR input file. You can also edit the command line arguments before hitting `Run` to customize the run behavior.

The output MLIR editor (read-only) will display the output dumped to `output.mlir`. The Logging window will display any output to stdout/stderr during the entire compile & run phase of the C++ program.

### Prebuilt Toy Binaries

Run any Toy tutorial chapter with your own input.

Using the program mode selector, select a Toy chapter. Under this mode, the code editor is disabled. The program that will run when you hit `Run` is the final binary for the corresponding Toy Chapter. The input MLIR editor can be edited under this mode to play with different user input.

## Contributing

[![Build and Deploy](https://github.com/MLIR-China/mlir-playground/actions/workflows/build-and-deploy.yml/badge.svg?branch=main)](https://github.com/MLIR-China/mlir-playground/actions/workflows/build-and-deploy.yml)

### Running Locally

1. Clone this repo.
2. Enter the `cpp` directory.
3. Run `build-docker.sh` to build the necessary docker image to build the web assembly libraries and binaries. Alternatively, pull from our DockerHub repo (specified in the GitHub workflow `build-and-deploy`).
4. Run `build-compiler-in-docker.sh` to build the web assembly libraries and binaries. This will run a series of steps using the `clang-wasm` docker image, and export the built files to the locations expected by the web app.
5. Exit to the project root directory.
6. Run `npm run dev` to start the development server locally.
