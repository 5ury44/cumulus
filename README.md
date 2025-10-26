# Cumulus - Cumulus-Based Distributed Execution SDK

A distributed execution system that sends code to remote GPU servers with Cumulus GPU partitioning, similar to Modal but without Docker.

## 🏗️ Architecture

```
[Local Client] → [Code Packaging] → [Remote Server] → [Cumulus Partition] → [Execution] → [Results]
```

### Components

1. **Client SDK** (`sdk/`) - Local Python SDK for wrapping and sending code
2. **Server Worker** (`worker/`) - Remote execution worker with Cumulus integration
3. **Code Transfer** - ZIP-based code packaging and transfer
4. **Cumulus Integration** - GPU partitioning and resource management

## 🚀 Quick Start

### 1. Setup Remote Server

```bash
# On your GPU server
cd /path/to/chronos
./install-chronos-gpu.sh

# Start the worker
python3 cumulus/worker/server.py --host 0.0.0.0 --port 8080
```

### 2. Use Local SDK

```python
from cumulus.sdk import CumulusClient

# Create client
client = CumulusClient(server_url="http://your-server:8080")

# Define your function
def train_model():
    import torch
    model = torch.nn.Linear(10, 1)
    # Your training code here
    return model.state_dict()

# Run on remote GPU with Cumulus partition
result = client.run(
    func=train_model,
    gpu_memory=0.5,  # 50% of GPU memory
    duration=3600,   # 1 hour
    requirements=["torch", "numpy"]
)

print(f"Training result: {result}")
```

## 📁 Project Structure

```
cumulus/
├── README.md                 # This file
├── sdk/                      # Client SDK
│   ├── __init__.py
│   ├── client.py            # Main client class
│   ├── code_packager.py     # Code packaging utilities
│   └── decorators.py        # Function decorators
├── worker/                   # Server-side worker
│   ├── __init__.py
│   ├── server.py            # FastAPI server
│   ├── executor.py          # Code execution engine
│   ├── chronos_manager.py   # Cumulus integration
│   └── requirements.txt     # Server dependencies
├── examples/                 # Usage examples
│   ├── basic_usage.py
│   ├── gpu_training.py
│   └── distributed_training.py
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_installation.py  # Local SDK installation test
    ├── test_remote.py        # Remote server connection test
    ├── test_ssh.py           # Server-side execution test
    ├── test_simple_pytorch.py # PyTorch CUDA functionality test
    └── test_gpu_benchmark.py # Comprehensive GPU performance test
```

## 🔧 Features

- **No Docker Required** - Direct Python execution on remote server
- **Cumulus Integration** - Automatic GPU partitioning and resource management
- **Code Packaging** - Automatic dependency detection and packaging
- **Result Serialization** - Automatic serialization of return values
- **Error Handling** - Comprehensive error reporting and logging
- **Resource Management** - Automatic cleanup and resource monitoring

## 📊 Benefits over Docker-based Solutions

1. **Faster Startup** - No container build/start time
2. **Lower Resource Overhead** - Direct Python execution
3. **Better GPU Access** - Direct Cumulus integration
4. **Simpler Debugging** - Direct access to server logs
5. **Dynamic Dependencies** - Install packages on-demand

## 🧪 Testing

### Local Testing

```bash
# Test the SDK installation
python tests/test_installation.py

# Test with remote server (requires running worker)
python tests/test_remote.py
```

### Server Testing

```bash
# Test on the server directly
python tests/test_ssh.py

# Test PyTorch CUDA functionality
python tests/test_simple_pytorch.py

# Run comprehensive GPU benchmarks
python tests/test_gpu_benchmark.py
```

## 🎯 Use Cases

- **Machine Learning Training** - Send training scripts to GPU servers
- **Data Processing** - Run compute-intensive tasks remotely
- **Research Experiments** - Execute experiments on shared GPU resources
- **Development** - Test code on remote hardware without local setup
