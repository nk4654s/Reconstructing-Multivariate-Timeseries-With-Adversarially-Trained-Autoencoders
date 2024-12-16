DATAFLUX_DIR="packages/Dataflux"
COMPRISK_DIR="packages/Comprisk"

pdoc --html $COMPRISK_DIR -o docs/ --force
pdoc --html $DATAFLUX_DIR -o docs/ --force