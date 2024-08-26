#!/bin/sh
if [ -z "$1" ]; then
    echo "Please specify a dataset name."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please specify a start date (YYYY-MM-DD)."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please specify an end date (YYYY-MM-DD)."
    exit 1
fi

printf "$1: Downloading dataset... "
flux-pre --folder "$1" --machine-keep-state on --interval "$2" "$3" --overwrite > /dev/null 2>&1
printf "Done\n"