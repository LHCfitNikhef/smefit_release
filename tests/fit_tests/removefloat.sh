#!/usr/bin/env bash
# Remove all 'float64: True' lines from YAML files in current folder

# Loop over YAML and YML files in the current directory
for f in ./test_results/*/*.yaml *.yml; do
  # Skip if no files match
  [ -e "$f" ] || continue
  echo "Cleaning $f ..."
  sed -i '/float64:[[:space:]]*True/d' "$f"
done
