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
# On your GPU server (clean install)
python3 -m venv cumulus-env && source cumulus-env/bin/activate
pip install -e .

# Option A: build vendored Chronos and let cumulus discover it
#   (see cumulus/chronos_vendor/README.md)

# Start the worker
cumulus-cli serve --host 0.0.0.0 --port 8080
```

### 2. Use Local SDK

#### **Basic Usage**

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

#### **Framework-Agnostic Checkpointing**

```python
from cumulus.sdk import CumulusClient
from runtime import save_checkpoint, load_checkpoint

def train_with_checkpointing():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Create model (works with any framework)
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = optim.Adam(model.parameters())

    # Training loop...
    for epoch in range(5):
        # ... training code ...

        # Save checkpoint (framework-agnostic)
        checkpoint_info = save_checkpoint(
            model, optimizer,
            epoch=epoch, step=100,
            framework="pytorch"  # or "tensorflow", "sklearn", "xgboost", "lightgbm"
        )

        # Resume from checkpoint
        load_result = load_checkpoint(
            model, optimizer,
            checkpoint_path=checkpoint_info['local_path'],
            framework="pytorch"
        )

    return model.state_dict()

# Run with checkpointing
client = CumulusClient("http://your-server:8080")
result = client.run(
    func=train_with_checkpointing,
    gpu_memory=0.8,
    duration=3600,
    requirements=["torch", "boto3"]
)
```

## 💾 Distributed Checkpointing System

Cumulus provides a unified checkpointing system that works across all major ML frameworks:

### **Supported Frameworks**

- **PyTorch** (`framework="pytorch"`)
- **TensorFlow/Keras** (`framework="tensorflow"`)
- **scikit-learn** (`framework="sklearn"`)
- **XGBoost** (`framework="xgboost"`)
- **LightGBM** (`framework="lightgbm"`)

### **Two-Tier Storage Architecture**

- **L1 Cache (Local)**: Fast local disk storage for recent checkpoints
- **L2 Cache (S3)**: Cloud storage for cross-machine access and persistence

### **Unified API**

```python
from runtime import save_checkpoint, load_checkpoint

# Save checkpoint (works with any framework)
checkpoint_info = save_checkpoint(
    model, optimizer,
    epoch=epoch, step=step,
    framework="pytorch"  # Auto-detects if omitted
)

# Load checkpoint
load_result = load_checkpoint(
    model, optimizer,
    checkpoint_path=checkpoint_info['local_path'],
    framework="pytorch"
)
```

### **S3 Configuration**

Set up S3 for distributed checkpointing by creating `/opt/cumulus-distributed/.env`:

```bash
CUMULUS_S3_BUCKET=your-bucket-name
CUMULUS_S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
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
    ├── test_complete_nn.py   # Complete neural network training with checkpointing
    ├── test_unified_s3_complete.py # Multi-framework S3 checkpointing test
    ├── test_artifact_store.py # Artifact storage and retrieval test
    ├── test_installation.py  # Local SDK installation test
    ├── test_remote.py        # Remote server connection test
    ├── test_ssh.py           # Server-side execution test
    ├── test_simple_pytorch.py # PyTorch CUDA functionality test
    └── test_gpu_benchmark.py # Comprehensive GPU performance test
```

## 🔧 Features

- **No Docker Required** - Direct Python execution on remote server
- **Cumulus Integration** - Automatic GPU partitioning and resource management
- **Framework-Agnostic Checkpointing** - Unified API for PyTorch, TensorFlow, scikit-learn, XGBoost, and LightGBM
- **Distributed Checkpointing** - L1 (local) + L2 (S3) cache system for cross-machine job resumption
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

### Current Test Suite

The test suite includes comprehensive tests for distributed execution and checkpointing:

#### **Complete Neural Network Training Test** (`test_complete_nn.py`)

- **Purpose**: End-to-end PyTorch training with checkpoint/resume workflow
- **Features**:
  - Unified checkpointing API (`save_checkpoint()`, `load_checkpoint()`)
  - Automatic S3 integration (if configured)
  - Training → checkpoint → resume → complete workflow
- **Usage**: `python cumulus/tests/test_complete_nn.py`

#### **Multi-Framework S3 Integration Test** (`test_unified_s3_complete.py`)

- **Purpose**: Comprehensive testing across all supported ML frameworks
- **Frameworks**: PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM
- **Features**:
  - Direct `DistributedCheckpointer` usage
  - Explicit S3 upload/download verification
  - Framework-specific adapter testing
- **Usage**: `python cumulus/tests/test_unified_s3_complete.py`

#### **Artifact Store Test** (`test_artifact_store.py`)

- **Purpose**: Test artifact storage and retrieval
- **Features**: Local and S3 artifact management
- **Usage**: `python cumulus/tests/test_artifact_store.py`

### Running Tests

#### **Local Testing (requires running worker)**

```bash
# Complete neural network training with checkpointing
python cumulus/tests/test_complete_nn.py

# Multi-framework checkpointing test
python cumulus/tests/test_unified_s3_complete.py

# Artifact store functionality
python cumulus/tests/test_artifact_store.py
```

#### **Server Testing**

```bash
# Test on the server directly
python tests/test_ssh.py

# Test PyTorch CUDA functionality
python tests/test_simple_pytorch.py

# Run comprehensive GPU benchmarks
python tests/test_gpu_benchmark.py
```

## 🎯 Use Cases

- **Machine Learning Training** - Send training scripts to GPU servers with automatic checkpointing
- **Cross-Machine Job Resumption** - Resume training on different machines using S3 checkpoint storage
- **Multi-Framework Support** - Train models in PyTorch, TensorFlow, scikit-learn, XGBoost, or LightGBM
- **Data Processing** - Run compute-intensive tasks remotely
- **Research Experiments** - Execute experiments on shared GPU resources with persistent state
- **Development** - Test code on remote hardware without local setup
