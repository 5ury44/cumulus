#!/bin/bash
set -e

echo "🏗️  Building Cumulus Job Container..."

# Build the container image
docker build -t cumulus-job:latest .

echo "✅ Container built successfully!"
echo ""
docker images cumulus-job:latest
