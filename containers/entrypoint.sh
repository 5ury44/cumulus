#!/bin/bash
set -e

echo "🚀 Cumulus Job Container Starting"
echo "Job ID: ${JOB_ID:-unknown}"
echo "Chronos Partition: ${CHRONOS_PARTITION_ID:-none}"
echo "GPU Device: ${GPU_DEVICE:-none}"

# Check if we're in a job directory
if [ ! -d "/job" ]; then
    echo "❌ Error: /job directory not found"
    exit 1
fi

cd /job

# Check if code package exists
if [ ! -f "code.zip" ]; then
    echo "❌ Error: code.zip not found in /job directory"
    echo "Contents of /job:"
    ls -la /job/
    exit 1
fi

echo "📦 Extracting code package..."
unzip -q code.zip

# Check if main.py exists after extraction
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found after extracting code.zip"
    echo "Contents after extraction:"
    ls -la /job/
    exit 1
fi

# Install requirements if they exist
if [ -f "requirements.txt" ]; then
    echo "📋 Installing requirements..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "📋 No requirements.txt found, skipping package installation"
fi

# Show GPU information if available
if command -v clinfo &> /dev/null; then
    echo "🔍 GPU Information:"
    clinfo -l || echo "No OpenCL devices found"
fi

# Show environment
echo "🌍 Environment:"
echo "  Python: $(python --version)"
echo "  Working Directory: $(pwd)"
echo "  User: $(whoami)"

# Execute the job
echo "▶️  Executing job..."
python /app/runner.py

echo "✅ Job execution completed"