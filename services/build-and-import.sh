#!/usr/bin/env bash
# Build adapter and UI container images and import into k3s on DarkBase.
# Usage: ./build-and-import.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building mimi-llm-adapter ==="
docker build -t mimi-llm-adapter:latest ./llm-adapter

echo "=== Building mimi-image-adapter ==="
docker build -t mimi-image-adapter:latest ./image-adapter

echo "=== Building mimi-ai-ui ==="
docker build -t mimi-ai-ui:latest ./ui

echo "=== Importing images into k3s ==="
docker save mimi-llm-adapter:latest | sudo k3s ctr images import -
docker save mimi-image-adapter:latest | sudo k3s ctr images import -
docker save mimi-ai-ui:latest | sudo k3s ctr images import -

echo "=== Done! ==="
echo "Imported images:"
sudo k3s ctr images ls | grep mimi
