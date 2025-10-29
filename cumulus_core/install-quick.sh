#!/bin/bash

# Quick Chronos GPU Installation Script
# This is a simplified version for experienced users

set -e

echo "🚀 Quick Chronos GPU Installation"
echo "================================="

# Check if we're in the right directory
if [[ ! -f "CMakeLists.txt" ]] || [[ ! -f "src/partitioner.cpp" ]]; then
    echo "❌ Please run from Chronos root directory"
    exit 1
fi

# Install dependencies (Ubuntu/Debian)
echo "📦 Installing dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake ocl-icd-opencl-dev opencl-headers python3 python3-pip python3-venv git pciutils clinfo

# Apply GPU platform fix
echo "🔧 Applying GPU platform fix..."
python3 -c "
import re
with open('src/partitioner.cpp', 'r') as f:
    content = f.read()
pattern = r'(platform = platforms\[0\];)'
replacement = '''platform = platforms[0];
    // Prefer GPU platform over CPU platform
    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err == CL_SUCCESS && numDevices > 0) {
            platform = platforms[i];
            break;
        }
    }'''
new_content = re.sub(pattern, replacement, content)
with open('src/partitioner.cpp', 'w') as f:
    f.write(new_content)
print('✅ GPU platform fix applied')
"

# Build and install
echo "🔨 Building Chronos..."
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
make -j$(nproc)
sudo make install

# Set up environment
echo "⚙️ Setting up environment..."
sudo mkdir -p /tmp/chronos_locks
sudo chmod 777 /tmp/chronos_locks
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
echo 'export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc

# Install Python bindings
echo "🐍 Installing Python bindings..."
cd ..
python3 -m pip install -e . --user

# Test installation
echo "🧪 Testing installation..."
chronos_cli stats

echo "✅ Chronos GPU installation complete!"
echo ""
echo "Usage examples:"
echo "  chronos_cli stats"
echo "  chronos_cli create 0 0.5 3600"
echo "  python3 -c 'from chronos import Partitioner; p = Partitioner(); p.show_stats()'"
