# MLIR Playground

Play with [MLIR](https://mlir.llvm.org/) directly in your browser.

- **Get Started Immediately**: No need to install any dependencies or setup a build system.
- **Private & Safe**: Everything runs locally in the sandboxed environment of your browser.

Try it out now at our production site: [MLIR Playground](https://playground.mlir-china.org/).

![Screenshot](https://user-images.githubusercontent.com/3676913/190872831-ec97d68f-31df-4058-974c-90bfc0b2d1bc.png)

## Features

### Custom mlir-opt

Create & run your own mlir-opt C++ program.

Using the built-in C++ editor, you have the freedom to implement anything that is possible with the framework. For example, you can write a pass and run it on some mlir input. All MLIR & LLVM libraries in the [LLVM codebase](https://github.com/llvm/llvm-project) are available to use immediately.

To compile & run your program, hit the `Run` button below the editor. This will compile your C++ program using clang & lld (compiled to web assembly and run in your browser), and execute the program with the user-provided MLIR input file. You can also edit the command line arguments before hitting `Run` to customize the run behavior.

The output MLIR editor will display the output. The Logging window will display any output to stdout/stderr during the entire compile & run phase of the C++ program.

### Table-Driven Declarative Rewrite Rule (DRR)

Declare your rewrite rules using DRR syntax and avoid all the C++ boilerplate.

Under the "TableGen DRR" preset, you can define your own rewrite rules declaratively ([docs](https://mlir.llvm.org/docs/DeclarativeRewrites/)). This greatly reduces the amount of code needed to implement a transformation.

For heavy users, two other tabs show more information that can be tweaked:

- The "Generated" tab shows the generated C++ code for your rewrite patterns. This allows you to better understand how things work under the hood, and make sure the pattern you write matches what you expect the code to do.
- The "Driver" tab shows the mlir-opt-like driver that is actually running the pattern. It is very similar to the preset code for "C++ Pattern". This allows you to further tweak how the rewrite pattern is actually run on the IR.

### Prebuilt Toy Binaries

Run any Toy tutorial chapter with your own input.

Using the program mode selector, select a Toy chapter. Under this mode, the code editor is disabled. The program that will run when you hit `Run` is the final binary for the corresponding Toy Chapter. The input MLIR editor can be edited under this mode to play with different user input.

## The Road Ahead

MLIR is an exciting technology that modularized different compiler building blocks, however the developer workflow remains pretty much the same, i.e. local C++/CMake projects on a laptop or a workstation. It's still quite inconvenient for people to land productive discussions on some technical details, even though `.mlir` files are already very helpful.

MLIR-Playground is a community effort that inspired by the trend of leveraging web technology to boost collaboration efficiency and lower entrance barriers, such as Figma to UI design, OpenAI Playground to large AI models and countless sandboxs by many great web frameworks.

To test out this idea on both the technical side and the “business” side, especially whether modern browsers are capable of handling MLIR workload, MLIR-Playground is kicked off as a pure Wasm app. Such approach also enables low deployment/distribution cost and therefore be able to reach more people. In the long run, MLIR-Playground will explore different opportunities to enable productive technical compiler debates, such as building more collaboration features and lowering barriers to trying out MLIR features.

PRs and suggestions are all warmly welcomed. At the end of the day, it's all about building something together, especially when it can improve our daily life in some practical ways.

## Contributing

[![Build and Deploy](https://github.com/MLIR-China/mlir-playground/actions/workflows/build-and-deploy.yml/badge.svg?branch=main)](https://github.com/MLIR-China/mlir-playground/actions/workflows/build-and-deploy.yml)

### Running Locally

1. Clone this repo.
2. Enter the `cpp` directory.
3. Run `build-docker.sh` to build the necessary docker image for building the web assembly libraries and binaries. Alternatively, pull from our DockerHub repo (specified in the GitHub workflow `build-and-deploy`).
4. Run `build-compiler-in-docker.sh` to build the web assembly libraries and binaries. This will run a series of steps using the `clang-wasm` docker image, and export the built files to the locations expected by the web app.
5. Exit to the project root directory.
6. Run `npm run dev` to start the development server locally.

### Self-Host Instruction

To selfhost MLIR-Playground locally, you need to have Docker installed beforehand.
To install Docker, please refer to the [official docs](https://docs.docker.com/get-started/#download-and-install-docker).
Once docker is installed properly, all you need is running the following command in terminal. This process could take up to several minutes depending on your internet connection.

```sh
docker run -d -p 3000:3000 mlirchina/mlir-playground
```

Once the command finishes, you can access MLIR-Playground by opening http://localhost:3000 in your browser. Have fun!
