printf "Cleaning up..."
rm -rf "requirements.txt"
rm -rf "Comprisk.egg-info"
rm -rf ".VSCodeCounter"
rm -rf "logfile"

if [ "$(pwd)" = "/app" ]; then
    rm -rf "Dockerfile"
    rm -rf "Dockerfiles/"
    rm -rf "logs/*"
    rm -rf "configs/*"
    rm -rf "models/*"
    rm -rf "docs"
    rm -rf "playground.ipynb"
fi

printf "Done!\n"