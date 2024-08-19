#!/bin/sh
DIRECTORY="dataset-2023-04-01-orange"

FROM_DATE="2023-04-01"
TO_DATE="2023-08-01"

printf "$DIRECTORY_RAW: Rebuilding package... "
pip install -e . > /dev/null 2>&1
printf "Done\n"

printf "$DIRECTORY_RAW: Downloading dataset... "
flux-pre --folder "$DIRECTORY" --machine-keep-state on --interval "$FROM_DATE" "$TO_DATE" --overwrite > /dev/null 2>&1
printf "Done\n"