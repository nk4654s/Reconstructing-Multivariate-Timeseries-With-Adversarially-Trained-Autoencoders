DATAFLUX_DIR="packages/Dataflux"
COMPRISK_DIR="packages/Comprisk"

#########################################################################
#                                get user                               #
#########################################################################
if [ -z "$1" ]; then
    echo "Arg 1: Please specify a user."
    exit 1
fi

#########################################################################
#                               get device                              #
#########################################################################
if [ -z "$2" ]; then
    echo "Arg: 2 Please specify 'cpu' or 'cuda'"
    exit 1
fi

#########################################################################
#                              get os/home                              #
#########################################################################
HOME="/home"
if [ "$(uname)" = "Darwin" ]; then
    HOME="/Users"
fi

PLATFORM=$(uname -m)
echo "User ($1) on $PLATFORM with home=$HOME/$1/"

#########################################################################
#                            install miniconda                          #
#########################################################################
if [ -d "$HOME/$1/.miniconda" ]; then
    echo "Conda already installed"
else
    echo "Downloading miniconda"
    # download platform specific miniconda installer
    if [ "$PLATFORM" = "x86_64" ]; then
        curl "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output miniconda.sh
    else
        if [ "$2" = "cuda" ]; then
            echo "CUDA on non x86 architecture not supported."
            exit 1
        fi
        curl "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" --output miniconda.sh
    fi

    echo "Installing miniconda"
    bash miniconda.sh -b -p "$HOME/$1/.miniconda"
    rm -f miniconda.sh
fi

#########################################################################
#                           create environment                          #
#########################################################################
echo "Creating environment"

if [ "$2" = "cuda" ]; then
    conda env create -f requirements.yml
    conda run -n cas conda install pytorch-cuda=12.4 -c nvidida -c pytorch
else
    conda env create -f requirements.yml
fi

#########################################################################
#                        install missing packages                       #
#########################################################################
conda run -n cas pip install -U 'jsonargparse[signatures]>=4.27.7'
conda run -n cas pip install -e $DATAFLUX_DIR
conda run -n cas pip install -e $COMPRISK_DIR
