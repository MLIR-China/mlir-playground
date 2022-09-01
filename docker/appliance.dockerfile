FROM ubuntu:22.04

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
    libicu-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    apt-transport-https \
    build-essential \
    ninja-build \
    rsync \
    vim \
    bash-completion

# Install node.js
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs

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


