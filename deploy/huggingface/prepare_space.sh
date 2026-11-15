#!/usr/bin/env bash
# Assemble the Hugging Face Space folder in build/hf_space (needs models/ with the 4 model files).
# Run from the project root: bash deploy/huggingface/prepare_space.sh
set -euo pipefail
for f in items.parquet user_side.npz ranker.txt manifest.json; do
  [ -f "models/$f" ] || { echo "models/$f is missing"; exit 1; }
done
rm -rf build/hf_space
mkdir -p build/hf_space/src build/hf_space/models
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md build/hf_space/
cp -R app build/hf_space/app
cp src/__init__.py src/serving.py src/load_qdrant.py build/hf_space/src/
cp models/items.parquet models/user_side.npz models/ranker.txt models/manifest.json build/hf_space/models/
find build/hf_space -name __pycache__ -type d -prune -exec rm -rf {} +
echo "Ready: build/hf_space"
du -sh build/hf_space
