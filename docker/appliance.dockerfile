FROM ubuntu:20.04

RUN echo "Force refresh"

ENV TZ=US/Pacific
ENV LANG=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Install basic packages
RUN apt-get clean
RUN apt-get update && apt-get install -y \
    locales \
    sudo \
    wget \
    curl \
    jq \
    git \
    nano \
    make \
    ffmpeg \
    build-essential \
    ninja-build \
    vim \

# Install node.js
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs

# Setup Git (Nano doesn't work well with VSCode remote due to keybinding issues)
RUN git config --global core.editor "vim"

# Setup a non-root user
ARG USER=playground
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USER \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER  -d /home/$USER/\
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

USER $USER_UID:$USER_GID
WORKDIR /home/$USER

ADD . .

ENTRYPOINT ["/bin/bash", "docker/appliance.sh"]


