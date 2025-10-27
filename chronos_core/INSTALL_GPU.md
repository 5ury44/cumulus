# Chronos GPU Installation Guide

This guide provides installation instructions for Chronos with GPU support, including the necessary fixes for proper GPU detection.

## 🚀 Quick Installation

For a fast installation on Ubuntu/Debian systems:

```bash
# Clone the repository
git clone <your-repo-url>
cd chronos

# Run the quick install script
./install-quick.sh
```

## 📋 Detailed Installation

For a comprehensive installation with full error checking and cross-platform support:

```bash
# Clone the repository
git clone <your-repo-url>
cd chronos

# Run the detailed install script
./install-chronos-gpu.sh
```

## 🔧 What the Installation Does

### 1. System Dependencies

- **Build tools**: gcc, g++, cmake
- **OpenCL development**: ocl-icd-opencl-dev, opencl-headers
- **Python**: python3, python3-pip, python3-venv
- **Utilities**: git, pciutils, clinfo

### 2. GPU Platform Fix

The installation automatically applies a critical fix to `src/partitioner.cpp` that ensures Chronos detects GPU devices instead of just CPU devices:

```cpp
// Original code (detects CPU first)
platform = platforms[0];

// Fixed code (prefers GPU platforms)
platform = platforms[0];
// Prefer GPU platform over CPU platform
for (cl_uint i = 0; i < numPlatforms; i++) {
    cl_uint numDevices;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err == CL_SUCCESS && numDevices > 0) {
        platform = platforms[i];
        break;
    }
}
```

### 3. Build and Install

- Configures CMake with Release build
- Enables tests and examples
- Builds with optimal parallel jobs
- Installs system-wide

### 4. Environment Setup

- Creates lock directory: `/tmp/chronos_locks`
- Sets up library paths in shell configuration
- Installs Python bindings

## 🧪 Verification

After installation, verify everything works:

```bash
# Test CLI
chronos_cli stats

# Test Python bindings
python3 -c "from chronos import Partitioner; p = Partitioner(); p.show_stats()"

# Run tests
cd build && ctest --output-on-failure
```

## 📊 Expected Output

You should see output similar to:

```
Found 1 OpenCL device(s)
Device 0: NVIDIA GeForce RTX 3090
  Type: GPU
  Vendor: NVIDIA Corporation
  OpenCL version: OpenCL 3.0 CUDA
  Total memory: 24148 MB
```

## 🎯 Usage Examples

### CLI Usage

```bash
# Show GPU statistics
chronos_cli stats

# Create a partition (50% of GPU 0 for 1 hour)
chronos_cli create 0 0.5 3600

# List active partitions
chronos_cli list

# Check available memory
chronos_cli available 0
```

### Python Usage

```python
from chronos import Partitioner

# Create partitioner
p = Partitioner()

# Show stats
p.show_stats()

# Create a partition with context manager
with p.create(device=0, memory=0.5, duration=3600) as partition:
    print(f"Partition: {partition.partition_id}")
    print(f"Time remaining: {partition.time_remaining} seconds")
    # Your GPU code here
    # Automatic cleanup when done

# Check available memory
available = p.get_available(device=0)
print(f"Available: {available:.1f}%")
```

## 🔍 Troubleshooting

### GPU Not Detected

If Chronos only detects CPU devices:

1. **Check OpenCL platforms**:

   ```bash
   clinfo -l
   ```

2. **Verify NVIDIA drivers**:

   ```bash
   nvidia-smi
   ```

3. **Check device files**:
   ```bash
   ls -la /dev/nvidia*
   ```

### Library Not Found

If you get "library not found" errors:

```bash
# Add to your shell configuration
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# For macOS
export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH}"
```

### Python Import Errors

If Python can't import Chronos:

```bash
# For virtual environment users
source ~/.chronos-venv/bin/activate

# For system-wide installation
python3 -m pip install -e . --user
```

## 🏗️ Manual Installation

If the scripts don't work, you can install manually:

```bash
# 1. Install dependencies
sudo apt-get install build-essential cmake ocl-icd-opencl-dev opencl-headers python3 python3-pip git

# 2. Apply GPU fix (see above code snippet)

# 3. Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
make -j$(nproc)

# 4. Install
sudo make install

# 5. Set up environment
sudo mkdir -p /tmp/chronos_locks
sudo chmod 777 /tmp/chronos_locks
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# 6. Install Python bindings
cd ..
python3 -m pip install -e . --user
```

## 📁 File Structure

After installation:

```
/usr/local/bin/chronos_cli          # CLI executable
/usr/local/lib/libchronos.so        # Shared library
/usr/local/lib/libchronos.a         # Static library
/usr/local/include/chronos/         # Header files
/tmp/chronos_locks/                 # Lock directory
~/.chronos-venv/                    # Python virtual environment (if used)
```

## 🎉 Success!

Once installed, Chronos provides:

- ✅ **Fair GPU time-sharing** with automatic expiration
- ✅ **Multi-user support** with resource isolation
- ✅ **Memory enforcement** to prevent conflicts
- ✅ **Python and CLI APIs** for easy integration
- ✅ **Cross-platform support** (Linux, macOS, Windows)

Perfect for research labs, development environments, and distributed systems that need reliable GPU resource management!
