#set -x -u -u -e -o pipefail

# configure mounts
CONF_FOLDER="./configs/"
LOG_FOLDER="./logs/"
MODEL_FOLDER="./models/"
DATA_FOLDER="/home/jonas/NEAX/Dataset"

# configure ports
TB_PORT=6006
OP_PORT=8080

# configure specs
N_CPU=10
SHARED_MEM=8G

# auto-configure platform
PLATFORM=$(uname -m)

# check the given command, if none then fallback to bash
CMD="$@"
ARG="-d -it"
NAME="comprisk"
if [ $# -eq 0 ]; then
    ARG="-it"
    CMD="/bin/bash"
else
    NAME="$1"
fi

# build the docker command
docker run --shm-size="$SHARED_MEM" --cpus="$N_CPU" --gpus all \
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
        -it -d comprisk:latest /bin/bash -c "conda run -n comprisk optuna-dashboard sqlite:///logs/optuna.sqlite3 "
else
    # start tensorboard
    docker run --shm-size=256M --cpus=1 \
        --name tensorboard \
        -p 6006:6006 \
        -v "$MODEL_FOLDER":/app/models/ \
        -it -d comprisk:latest tensorboard --logdir=models --host 0.0.0.0
fi
