set -x -u -e -o pipefail
if [ -z "$1" ]; then
    echo "Please specify 'cuda' or 'cpu'."
    exit 1
fi

PLATFORM=$(uname -m)

if [ "$1" = "cuda" ]; then
    if [ "$PLATFORM"!="x86_64" ]; then
        echo "Only x86_64 platform is supported when using GPU."
        exit 1
    fi
    cp Dockerfiles/docker-cuda.dockerfile ./Dockerfile
fi

if [ "$1" = "cpu" ]; then
    cp Dockerfiles/docker-nocuda.dockerfile ./Dockerfile
fi

docker build --platform "linux/$PLATFORM" -t comprisk .
rm -rf Dockerfile