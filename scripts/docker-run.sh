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
SYS_ARGS="--shm-size=$SHARED_MEM --cpus=$N_CPU"
if [ -z "$1" ]; then
    echo "Set first argument to 'cpu' or 'gpu'"
    exit 1
else
    if [ "$1" = "gpu" ]; then
        SYS_ARGS="--shm-size=$SHARED_MEM --cpus=$N_CPU --gpus all"
    else
        if [ "$1" != "cpu" ]; then
            echo "Only 'cpu' or 'gpu' are allowed as first argument"
            exit 1
        fi
    fi
fi

# auto-configure platform
PLATFORM=$(uname -m)

# check the given command, if none then fallback to bash
CMD="${@:2}"
ARG="-d -it"
NAME="comprisk"
if [ -z "$2" ]; then
    ARG="-it"
    CMD="/bin/bash"
else
    NAME="$2"
fi

# build the docker command
docker run $SYS_ARGS \
    --name $NAME \
    -v "$MODEL_FOLDER":/app/models/ \
    -v "$CONF_FOLDER":/app/configs/ \
    -v "$LOG_FOLDER":/app/logs/ \
    -v "$DATA_FOLDER":/app/dataset/ \
    $ARG comprisk:latest /bin/bash -c "$CMD"

if [ "$NAME" = "comprisk-tuner" ]; then
    # start tensorboard
    docker run --shm-size=256M --cpus=1 \
        --name optuna \
        -p 8080:8080 \
        -v "$MODEL_FOLDER":/app/models/ \
        -it -d comprisk:latest optuna-dashboard sqlite:///logs/optuna.sqlite3
else
    # start tensorboard
    docker run --shm-size=256M --cpus=1 \
        --name tensorboard \
        -p 6006:6006 \
        -v "$MODEL_FOLDER":/app/models/ \
        -it -d comprisk:latest tensorboard --logdir=models --host 0.0.0.0
fi
