# MLIR Playground

Play with [MLIR](https://mlir.llvm.org/) directly in your browser.

- **Get Started Immediately**: No need to install any dependencies or setup a build system.
- **Private & Safe**: Everything runs locally in the sandboxed environment of your browser.

Try it out now at our production site: [MLIR Playground](https://playground.mlir-china.org/).

![Screenshot](https://user-images.githubusercontent.com/3676913/234480160-80876753-82d2-4459-8861-a73df143ded5.png)

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

### Local Development

1. Clone this repo.
2. Enter the `cpp` directory.
3. (Optional) Run `build-image.sh` to build the necessary docker image for building the web assembly libraries and binaries (32G RAM is required to build the docker image locally). Pre-built images are available at [ghcr.io/mlir-china/clang-wasm](https://ghcr.io/mlir-china/clang-wasm) to use directly.
4. Run `export-wasm.sh` to build the web assembly libraries and binaries. This will run a series of steps using the `clang-wasm` docker image, and export the built files to the locations expected by the web app.
5. Exit to the project root directory.
6. Run `npm run dev` to start the development server locally.

### Static Deployment

We provide a standalone static release package that contains all you need to host your own MLIR-Playground instance.

Visit our [Releases page](https://github.com/MLIR-China/mlir-playground/releases) to view and download a static release (`mlir-playground-static.tar.gz`). Each release is self-sufficient, and is ready to be served without any special dependencies.

For example, to download and start a simple HTTP server serving the site, run:

```sh
# Download the latest release.
wget https://github.com/MLIR-China/mlir-playground/releases/latest/download/mlir-playground-static.tar.gz
# Untar the package.
tar xzf mlir-playground-static.tar.gz
# Serve the website with python.
python3 -m http.server
```

The entire experience does not require an internet connection (The only exception is when generating a share link, which requires contacting our api servers. However, local export / import will always be available offline).
