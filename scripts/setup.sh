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
source "$HOME/.local/bin/env"

uv sync

#########################################################################
#                               setup env                               #
#########################################################################

if [ "$SHELL" = /bin/zsh ]; then
  rc=".zshrc"
elif [ "$SHELL" = /bin/bash ]; then
  rc=".bashrc"
else
  echo "Shell $SHELL is not supported, add environment var to your shell's init file."
  echo "> export CAS_DEBUG_MODE=1"
fi

echo "export CAS_DEBUG_MODE=1" >> "$HOME/$rc"
echo "export CAS_PROJECT_ROOT=$(pwd)" >> "$HOME/$rc"
