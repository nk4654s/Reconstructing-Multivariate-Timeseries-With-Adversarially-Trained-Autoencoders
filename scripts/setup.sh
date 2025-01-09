#########################################################################
#                              get os/home                              #
#########################################################################
HOME="/home/$USER"
if [ "$(uname)" = "Darwin" ]; then
    HOME="/Users/$USER"
fi

PLATFORM=$(uname -m)
echo "User ($1) on $PLATFORM with home=$HOME/"

#########################################################################
#                               install uv                              #
#########################################################################
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync
