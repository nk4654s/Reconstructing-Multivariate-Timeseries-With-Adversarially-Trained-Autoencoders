#set -x -u -u -e -o pipefail

# configure mounts
CONF_FOLDER="./configs/"
LOG_FOLDER="./logs/"
MODEL_FOLDER="./models/"
DATA_FOLDER="/Users/jonas/Dataset"

# configure ports
TB_PORT=6006
OP_PORT=8080

# configure specs
N_CPU=10
SHARED_MEM=8G

# auto-configure platform
PLATFORM=$(uname -m)

# check the given command, if none then fallback to bash
CMD="conda run -n comprisk $@"
ARG="-d -it"
NAME="comprisk"
if [ $# -eq 0 ]; then
    ARG="-it"
    CMD="/bin/bash"
else
    NAME="$1"
fi

# build the docker command
docker run --shm-size="$SHARED_MEM" --cpus="$N_CPU" \
    --name $NAME \
    -p 8080:8080 \
    -p 6006:6006 \
    -v "$MODEL_FOLDER":/app/models/ \
    -v "$CONF_FOLDER":/app/configs/ \
    -v "$LOG_FOLDER":/app/logs/ \
    -v "$DATA_FOLDER":/app/dataset/ \
    $ARG comprisk:latest /bin/bash

docker exec $NAME /bin/bash -c "$CMD"
docker logs -f --tail 10 $NAME

docker rm $(docker ps -aq)
