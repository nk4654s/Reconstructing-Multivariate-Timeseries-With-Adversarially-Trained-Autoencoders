DATAFLUX_DIR="../Dataflux"

if [ -d "$DIRECTORY" ]; then
    echo "Build 'Dataflux' pip package."
    pip install -e ../Dataflux > $PIP_BUILD
else
    echo "Can not build 'Dataflux'. Make sure to have the code next to this repository."
fi

echo "Update conda environment"
conda env update -f requrements.yml

echo "Generate documentation."
pdoc --html . -o docs

echo "Build 'Comprisk' pip package."
pip install -e . > $PIP_BUILD
