FROM emscripten/emsdk:3.1.7

# Install required packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates build-essential git cmake ninja-build clang lld python3

# Import the setup script and run
ADD install.sh /tmp/
RUN /tmp/install.sh
