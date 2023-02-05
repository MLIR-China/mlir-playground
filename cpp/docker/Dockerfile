FROM emscripten/emsdk:3.1.31

# Install required packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates build-essential git cmake ninja-build clang lld python3

# Temporary working dir for building the image
WORKDIR /tmp

# The setup script clones the necessary repo and builds any native targets
ADD common.sh setup.sh ./
ENV LLVM_ROOT /opt/llvm
RUN ./setup.sh

# The install script builds the wasm targets
ADD patches ./patches/
ADD install.sh ./
RUN ./install.sh
