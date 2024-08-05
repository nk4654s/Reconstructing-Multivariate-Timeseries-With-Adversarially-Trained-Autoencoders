if [ -z "$1" ]; then
    echo "Please specify a user."
    exit 1
fi

PLATFORM=$(uname -m)
echo "User ($1) on $PLATFORM"

# download platform specific miniconda installer
if [ "$PLATFORM" = "x86_64" ]; then
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh
else
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" -O miniconda.sh
fi

# install miniconda
echo "Installing miniconda"
mkdir -p "/home/$1/.conda"
bash miniconda.sh -b -p "/home/$1/miniconda3"
rm -f miniconda.sh
source "/home/$1/.bashrc"

# create environmnet
echo "Creating environment"
conda init
conda env create -f requirements.yml

# install packages conda missed
conda run -n comprisk pip install -U 'jsonargparse[signatures]>=4.27.7'
