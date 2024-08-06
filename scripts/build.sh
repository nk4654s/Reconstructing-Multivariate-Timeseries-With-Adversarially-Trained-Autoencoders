DATAFLUX_DIR="../Dataflux"
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

printf "Build 'Dataflux' pip package... "
if [ -d "$DATAFLUX_DIR" ]; then
    pip install -e ../Dataflux > /dev/null 2>&1
    printf "Done.\n"
else
    printf "WARNING: Can not build '$DATAFLUX_DIR'. Make sure to have the code next to this repository.\n"
fi

printf "Update conda environment... "
sh scripts/setup.sh $USER $DEVICE > /dev/null 2>&1
printf "Done.\n"

printf "Generate documentation... "
pdoc --html . -o docs --force > /dev/null 2>&1
printf "Done.\n"

printf "Build 'Comprisk' pip package... "
conda run -n comprisk pip install -e . > /dev/null 2>&1
printf "Done.\n"
