#!/bin/bash

set -o xtrace

drun='sudo docker run -it --rm --network=host --user root --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

# DEVICES="--gpus all"
DEVICES="--device=/dev/kfd --device=/dev/dri"

MEMORY="--ipc=host --shm-size 16G"

VOLUMES="-v $HOME/:/workspace -v /data:/data"

# WORK_DIR='/root/$(basename $(pwd))'
WORK_DIR="/workspace/"

# IMAGE_NAME=rocm/pytorch-nightly:latest
# IMAGE_NAME=rocm/pytorch:latest
# IMAGE_NAME=nvcr.io/nvidia/pytorch
# IMAGE_NAME=rocm/jax

# build docker
DOCKERFILE_PATH=jax-triton.Dockerfile
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
IMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f -1 -d '.')
CONTAINER_NAME=${IMAGE_NAME}_container_batra
bash scripts/docker_build.sh $DOCKERFILE_PATH

# start new container
#docker stop $CONTAINER_NAME
#docker rm $CONTAINER_NAME
CONTAINER_ID=$($drun -d -w $WORK_DIR --name $CONTAINER_NAME $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
# docker cp . $CONTAINER_ID:$WORK_DIR
# docker exec $CONTAINER_ID bash -c "bash scripts/amd/run.sh"
#docker attach $CONTAINER_ID
#docker stop $CONTAINER_ID
#docker rm $CONTAINER_ID
