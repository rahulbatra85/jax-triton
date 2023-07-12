# print every command
# set -o xtrace

# set path
DOCKERFILE_PATH="${1:-./jax-triton.Dockerfile}"

# get tag
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
DOCKERIMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f -1 -d '.')

# build docker
docker build --build-arg CACHEBUST=$(date +%s) -f $DOCKERFILE_PATH -t $DOCKERIMAGE_NAME .

echo $DOCKERIMAGE_NAME