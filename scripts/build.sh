USE_USER="$USER"
if [ -z "$2" ]; then
    echo "Using current user"
else
    USE_USER=$2
fi

DEVICE="cpu"
if [ -z "$1" ]; then
    echo "Building for CPU"
else
    if [ "$1" = "cpu" ] || [ "$1" = "cuda" ]; then
        DEVICE="$1"
    else
        echo "Unknown device: $1"
        exit 1
    fi
fi

printf "Build 'Dataflux' wheel... "
DATAFLUX_DIR="../Dataflux"
if [ -d "$DATAFLUX_DIR" ]; then
    pip wheel "$DATAFLUX_DIR" > /dev/null 2>&1
    printf "Done.\n"
else
    if [ -e "Dataflux-1.0-py3-none-any.whl" ]; then
        printf "Found existing wheel. \n"
    else
        printf "ERROR: Can not build '$DATAFLUX_DIR'. Make sure to have the code next to this repository.\n"
        exit 1
    fi
fi

printf "Update conda environment... "
sh scripts/setup.sh $USE_USER $DEVICE > /dev/null 2>&1
printf "Done.\n"

printf "Generate documentation... "
pdoc --html . -o docs --force > /dev/null 2>&1
printf "Done.\n"

printf "Build 'Comprisk' pip package... "
conda run -n comprisk pip install --force-reinstall Dataflux-1.0-py3-none-any.whl
conda run -n comprisk pip wheel . > /dev/null 2>&1
conda run -n comprisk pip install -e . > /dev/null 2>&1
printf "Done.\n"
