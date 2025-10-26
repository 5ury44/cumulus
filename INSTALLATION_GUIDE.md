# Cumulus Installation and Setup Guide

This guide walks you through setting up the complete Cumulus distributed execution system with Cumulus GPU partitioning.

## 🏗️ Architecture Overview

```
[Local Machine]                    [Remote GPU Server]
┌─────────────────┐               ┌─────────────────────┐
│   Cumulus SDK   │──────────────▶│  Cumulus Worker     │
│                 │   HTTP/JSON   │                     │
│  - Code Package │               │  - FastAPI Server   │
│  - Job Submit   │               │  - Cumulus Manager  │
│  - Results      │◀──────────────│  - Code Executor    │
└─────────────────┘               └─────────────────────┘
```

## 📋 Prerequisites

### Remote Server Requirements

- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with CUDA support
- Python 3.8+
- Cumulus installed and working (see `../chronos/INSTALL_GPU.md`)

### Local Machine Requirements

- Python 3.8+
- Internet connection to remote server

## 🚀 Quick Setup

### 1. Install Cumulus on Remote Server

```bash
# On your GPU server
cd /path/to/chronos
./install-chronos-gpu.sh

# Verify Cumulus is working
chronos_cli stats
```

### 2. Install Cumulus Worker on Remote Server

```bash
# On your GPU server
cd /path/to/cumulus
pip install -e .

# Start the worker server
python start_worker.py --host 0.0.0.0 --port 8080
```

### 3. Install Cumulus SDK on Local Machine

```bash
# On your local machine
cd /path/to/cumulus
pip install -e .

# Or install with ML dependencies
pip install -e ".[ml]"
```

## 🔧 Detailed Setup

### Remote Server Setup

#### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Install Cumulus (see ../chronos/INSTALL_GPU.md for details)
cd /path/to/chronos
./install-chronos-gpu.sh
```

#### Step 2: Install Cumulus Worker

```bash
# Clone or copy cumulus directory to server
cd /path/to/cumulus

# Create virtual environment (recommended)
python3 -m venv cumulus-env
source cumulus-env/bin/activate

# Install cumulus worker
pip install -e .

# Install additional dependencies if needed
pip install torch torchvision  # For ML workloads
```

#### Step 3: Configure and Start Worker

```bash
# Create job directory
sudo mkdir -p /tmp/cumulus_jobs
sudo chmod 777 /tmp/cumulus_jobs

# Start worker server
python start_worker.py --host 0.0.0.0 --port 8080 --workers 4

# Or run in background
nohup python start_worker.py --host 0.0.0.0 --port 8080 > worker.log 2>&1 &
```

#### Step 4: Verify Worker is Running

```bash
# Check if worker is responding
curl http://localhost:8080/health

# Check server info
curl http://localhost:8080/api/info
```

### Local Machine Setup

#### Step 1: Install Cumulus SDK

```bash
# Clone or copy cumulus directory
cd /path/to/cumulus

# Create virtual environment (recommended)
python3 -m venv cumulus-env
source cumulus-env/bin/activate

# Install cumulus SDK
pip install -e .

# Install ML dependencies if needed
pip install -e ".[ml]"
```

#### Step 2: Test Connection

```python
from cumulus.sdk import CumulusClient

# Test connection to remote server
client = CumulusClient("http://your-server:8080")
info = client.get_server_info()
print(f"Server info: {info}")
```

## 🎯 Usage Examples

### Basic Usage

```python
from cumulus.sdk import CumulusClient

# Create client
client = CumulusClient("http://your-server:8080")

# Define function to run remotely
def my_function():
    import math
    return {"pi": math.pi, "sqrt_2": math.sqrt(2)}

# Run on remote GPU
result = client.run(
    func=my_function,
    gpu_memory=0.5,  # 50% of GPU memory
    duration=3600,   # 1 hour
    requirements=["math"]
)

print(f"Result: {result}")
```

### Using Decorators

```python
from cumulus.sdk import CumulusClient, remote, gpu

client = CumulusClient("http://your-server:8080")

@remote(client, gpu_memory=0.8, duration=7200)
def train_model():
    import torch
    import torch.nn as nn

    # Your training code here
    model = nn.Linear(10, 1)
    # ... training logic ...

    return model.state_dict()

# Execute remotely
result = train_model()
```

### GPU Training Example

```python
from cumulus.sdk import CumulusClient, gpu

client = CumulusClient("http://your-server:8080")

@gpu(client, memory=0.9, duration=3600)
def pytorch_training():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Training code
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # ... training loop ...

    return {
        "model_state": model.state_dict(),
        "final_loss": 0.123,
        "epochs": 10
    }

# Run training
result = pytorch_training()
print(f"Training completed: {result}")
```

## 🔍 Monitoring and Debugging

### Check Server Status

```bash
# Check worker health
curl http://your-server:8080/health

# Get server info
curl http://your-server:8080/api/info

# List active jobs
curl http://your-server:8080/api/jobs
```

### Monitor Jobs

```python
# Get job status
job_status = client.get_job_status("job-id")
print(f"Job status: {job_status}")

# List all jobs
jobs = client.list_jobs()
for job in jobs:
    print(f"Job {job['job_id']}: {job['status']}")
```

### Debugging

```bash
# Check worker logs
tail -f worker.log

# Check Cumulus status
chronos_cli stats
chronos_cli list

# Check job directories
ls -la /tmp/cumulus_jobs/
```

## ⚙️ Configuration

### Environment Variables

```bash
# Worker configuration
export MAX_CONCURRENT_JOBS=5
export CHRONOS_PATH="/usr/local/bin/chronos_cli"
export JOB_TIMEOUT=3600

# Client configuration
export CUMULUS_SERVER_URL="http://your-server:8080"
export CUMULUS_API_KEY="your-api-key"  # Optional
```

### Worker Configuration

Create `worker_config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

chronos:
  path: "/usr/local/bin/chronos_cli"
  default_device: 0
  max_memory_fraction: 0.95

execution:
  timeout: 3600
  max_concurrent_jobs: 5
  job_directory: "/tmp/cumulus_jobs"

logging:
  level: "INFO"
  file: "worker.log"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Connection Refused

```bash
# Check if worker is running
ps aux | grep start_worker

# Check port availability
netstat -tlnp | grep 8080

# Check firewall
sudo ufw status
```

#### 2. Cumulus Not Found

```bash
# Check Cumulus installation
which chronos_cli
chronos_cli stats

# Check library path
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```

#### 3. GPU Not Detected

```bash
# Check GPU availability
nvidia-smi
clinfo -l

# Check Cumulus GPU detection
chronos_cli stats
```

#### 4. Job Execution Failed

```bash
# Check job logs
ls -la /tmp/cumulus_jobs/job-id/
cat /tmp/cumulus_jobs/job-id/error.json

# Check worker logs
tail -f worker.log
```

### Performance Optimization

#### 1. Increase Worker Processes

```bash
python start_worker.py --workers 8
```

#### 2. Optimize Cumulus Settings

```bash
# Check available memory
chronos_cli available 0

# Create test partition
chronos_cli create 0 0.5 60
chronos_cli list
```

#### 3. Monitor Resource Usage

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

## 🔒 Security Considerations

### Network Security

- Use HTTPS in production
- Implement API key authentication
- Use VPN or private networks
- Configure firewall rules

### Code Security

- Validate uploaded code
- Sandbox execution environments
- Monitor resource usage
- Implement job timeouts

### Data Security

- Encrypt sensitive data
- Use secure file transfers
- Implement access controls
- Regular security audits

## 📊 Performance Benchmarks

### Typical Performance

- **Job submission**: < 100ms
- **Code packaging**: < 500ms
- **GPU partition creation**: < 3s
- **Code execution**: Variable (depends on workload)
- **Result retrieval**: < 200ms

### Resource Usage

- **Worker memory**: ~50MB base + job memory
- **GPU memory**: As allocated by Cumulus
- **Disk usage**: ~100MB per job (temporary)

## 🎉 Success!

You now have a complete distributed execution system that can:

- ✅ Send code from local machine to remote GPU server
- ✅ Automatically manage GPU resources with Cumulus
- ✅ Execute code in isolated environments
- ✅ Return results back to local machine
- ✅ Handle multiple concurrent jobs
- ✅ Provide comprehensive monitoring and debugging

Perfect for machine learning training, data processing, and any compute-intensive tasks that need GPU acceleration!
