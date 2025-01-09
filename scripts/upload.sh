DIRECTORY="/app/models"

# Loop through files in the target directory
for MODEL_FOLDER in "$DIRECTORY"/*; do
  if [ -d "$MODEL_FOLDER" ]; then
      CONFIG="$MODEL_FOLDER/config.yaml"
      UPLOAD="$MODEL_FOLDER/upload.yaml"

      uv run flux-pipeline -c "$UPLOAD" --folder upload --latest
      uv run comprisk-generator -c "$CONFIG" --dataset upload
      rm -rf upload
  fi
done