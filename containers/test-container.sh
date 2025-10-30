#!/bin/bash
set -e

echo "🧪 Testing Cumulus Job Container"

# Build the container first
echo "🏗️  Building container..."
cd "$(dirname "$0")"
./build.sh

echo ""
echo "📦 Packaging test job..."

# Create test job package
cd test-job
zip -q ../test-job.zip main.py requirements.txt

cd ..

# Create temporary job directory
TEST_DIR="/tmp/cumulus-test-job-$$"
mkdir -p "$TEST_DIR"

# Copy job package
cp test-job.zip "$TEST_DIR/code.zip"

echo "🚀 Running test job in container..."

# Run the container
docker run --rm \
    -v "$TEST_DIR:/job" \
    -e JOB_ID="test-job-123" \
    -e CHRONOS_PARTITION_ID="test-partition-456" \
    -e GPU_DEVICE="0" \
    cumulus-job:latest

echo ""
echo "🧹 Cleaning up..."
rm -rf "$TEST_DIR"
rm -f test-job.zip

echo "✅ Container test completed!"