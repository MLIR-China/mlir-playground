# Command line argument to override docker image. Default is `clang-wasm`.
image=${1:-clang-wasm:latest}
workdir=/opt/tmp/mlir-playground

docker container create -it --privileged \
  --name mlir-playground \
  --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
  --device=/dev/infiniband/ --ipc=host \
  -v $(readlink -f `pwd`)/..:${workdir} \
  --workdir ${workdir} \
  --cpus=64 \
  ${image}
