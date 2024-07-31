DATAFLUX_DIR="../Dataflux"

printf "Build 'Dataflux' pip package... "
if [ -d "$DATAFLUX_DIR" ]; then
    pip install -e ../Dataflux > /dev/null 2>&1
    printf "Done.\n"
else
    printf "WARNING: Can not build '$DATAFLUX_DIR'. Make sure to have the code next to this repository.\n"
fi

printf "Update conda environment... "
conda env update -f requrements.yml > /dev/null 2>&1
printf "Done.\n"

printf "Generate documentation... "
pdoc --html . -o docs --force > /dev/null 2>&1
printf "Done.\n"

printf "Build 'Comprisk' pip package... "
pip install -e . > /dev/null 2>&1
printf "Done.\n"
