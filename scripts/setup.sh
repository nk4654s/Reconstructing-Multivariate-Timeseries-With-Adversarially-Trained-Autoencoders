if [ -z "$1" ]; then
    echo "Arg 1: Please specify a user."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Arg: 2 Please specify 'cpu' or 'cuda'"
    exit 1
fi

PLATFORM=$(uname -m)
echo "User ($1) on $PLATFORM"

# download platform specific miniconda installer
if [ "$PLATFORM" = "x86_64" ]; then
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh
else
    if [ "$2" = "cuda" ]; then
        echo "CUDA on non x86 architecture not supported."
        exit 1
    fi
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" -O miniconda.sh
fi

# install miniconda
if [ -d "/home/$1/miniconda3" ]; then
    echo "Conda already installed"
else
    echo "Installing miniconda"
    mkdir -p "/home/$1/.conda"
    bash miniconda.sh -b -p "/home/$1/miniconda3"
    rm -f miniconda.sh
    source "/home/$1/.bashrc"
    conda init
fi

# create environmnet
echo "Creating environment"

if [ "$2" == "cuda" ]; then
    conda env create -f requirements.yml
    conda run -n comprisk conda install pytorch-cuda=12.4 -c nvidida -c pytorch
else
    conda env create -f requirements.yml
fi

# install packages conda missed
conda run -n comprisk pip install -U 'jsonargparse[signatures]>=4.27.7'
