#!/usr/bin/python -u
DIRECTORY="/app/models"

# Loop through files in the target directory
for MODEL_FOLDER in "$DIRECTORY"/*; do
  if [ -d "$MODEL_FOLDER" ]; then
      VERSIONS=$(ls "$MODEL_FOLDER" | sort -r -V | grep "version_*")
      LATEST=$(echo "${VERSIONS}" | head -1)

      CONFIG="$MODEL_FOLDER/$LATEST/config.yaml"
      UPLOAD="$MODEL_FOLDER/$LATEST/upload.yaml"

      echo "uv run comprisk-generator -c $CONFIG --dataset $UPLOAD --data.stride 1"
  fi
done