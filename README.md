# MLIR Playground

Play with [MLIR](https://mlir.llvm.org/) directly in your browser.

- **Get Started Immediately**: No need to install any dependencies or setup a build system.
- **Private & Safe**: Everything runs locally in the sandboxed environment of your browser.

Try it out now at our production site: [MLIR Playground](https://playground.mlir-china.org/).

![Screenshot](https://user-images.githubusercontent.com/3676913/187361568-5ab648a3-22e2-4e88-aa7c-c17375ff3134.png)

## Features

### Custom mlir-opt

Create & run your own mlir-opt C++ program.

Using the built-in C++ editor, you can implement any feature you want. For example, you can write a pass and run it on some mlir input. All MLIR & LLVM libraries in the [LLVM codebase](https://github.com/llvm/llvm-project) are available to use immediately.

To compile & run your program, hit the `Run` button above the editor. This will compile your C++ program using clang & lld (compiled to web assembly and run in your browser), and execute the program with the user-provided MLIR input file. You can also edit the command line arguments before hitting `Run` to customize the run behavior.

The output MLIR editor (read-only) will display the output dumped to `output.mlir`. The Logging window will display any output to stdout/stderr during the entire compile & run phase of the C++ program.

### Prebuilt Toy Binaries

Run any Toy tutorial chapter with your own input.

Using the program mode selector, select a Toy chapter. Under this mode, the code editor is disabled. The program that will run when you hit `Run` is the final binary for the corresponding Toy Chapter. The input MLIR editor can be edited under this mode to play with different user input.

## The road ahead

MLIR is an exciting technology that modularized different compiler building blocks, however the developer workflow remains pretty much the same, i.e. local C++/CMake projects on a latop or a workstation. It's still quite inconvenient for people to land productive discussions on some technical details, altough `.mlir` files are very helpful.

MLIR-Playground is a community effort that inspired by the prevailing adoption of using web technology to boost collaboration effeciency and lower entrance barriers, such as Figma to UI design, OpenAI Playground to large AI models and countless sandboxs by many great web frameworks.

As a starting point, MLIR-Plagyround kicked off as a simple Wasm app to verify modern web browsers' capability. For the long run, MLIR-Plagyround will explore different opportunities to enable more productive technical discussions under the MLIR ecosystem, i.e. more collaboration features and lower barrier to try out using MLIR. Proposals and suggestions are welcome, let's build something together!

## Contributing

[![Build and Deploy](https://github.com/MLIR-China/mlir-playground/actions/workflows/build-and-deploy.yml/badge.svg?branch=main)](https://github.com/MLIR-China/mlir-playground/actions/workflows/build-and-deploy.yml)

### Running Locally

1. Clone this repo.
2. Enter the `cpp` directory.
3. Run `build-docker.sh` to build the necessary docker image for building the web assembly libraries and binaries. Alternatively, pull from our DockerHub repo (specified in the GitHub workflow `build-and-deploy`).
4. Run `build-compiler-in-docker.sh` to build the web assembly libraries and binaries. This will run a series of steps using the `clang-wasm` docker image, and export the built files to the locations expected by the web app.
5. Exit to the project root directory.
6. Run `npm run dev` to start the development server locally.
