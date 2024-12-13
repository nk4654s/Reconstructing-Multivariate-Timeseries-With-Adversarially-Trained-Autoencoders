export PATH="/Users/jonas/.miniconda/bin:$PATH"

DEVICE="cpu"

DATAFLUX_DIR="packages/Dataflux"
COMPRISK_DIR="packages/Comprisk"

BUILD_LOG="logfile-build"

#########################################################################
#                               get device                              #
#########################################################################
if [ -z "$1" ]; then
    echo "Building for CPU"
else
    if [ "$1" = "cpu" ] || [ "$1" = "cuda" ]; then
        DEVICE="$1"
        echo "Building for '$DEVICE'"
    else
        echo "Unknown device: $1"
        exit 1
    fi
fi

#########################################################################
#                        update conda environments                      #
#########################################################################
printf "Update conda environment... "
if [ "$DEVICE" == "cuda" ]; then
    conda env create -f $COMPRISK_DIR/requirements.yml >> $BUILD_LOG 2>&1
    conda run -n comprisk conda install pytorch-cuda=12.4 -c nvidida -c pytorch >> $BUILD_LOG 2>&1
else
    conda env create -f $COMPRISK_DIR/requirements.yml >> $BUILD_LOG 2>&1
fi

conda run -n comprisk pip install -U 'jsonargparse[signatures]>=4.27.7' >> $BUILD_LOG 2>&1
conda env create -f $DATAFLUX_DIR/environment.yml >> $BUILD_LOG 2>&1
printf "Done.\n"

#########################################################################
#                         generate documentation                        #
#########################################################################
printf "Generate documentation... "
pdoc --html $COMPRISK_DIR -o docs/ --force >> $BUILD_LOG 2>&1
pdoc --html $DATAFLUX_DIR -o docs/ --force >> $BUILD_LOG 2>&1
printf "Done.\n"

#########################################################################
#                         install the packages                          #
#########################################################################
printf "Build pip packages... "
conda run -n dataflux pip install -e $DATAFLUX_DIR >> $BUILD_LOG 2>&1

conda run -n comprisk pip install -e $COMPRISK_DIR >> $BUILD_LOG 2>&1
conda run -n comprisk pip install -e $COMPRISK_DIR >> $BUILD_LOG 2>&1
printf "Done.\n"