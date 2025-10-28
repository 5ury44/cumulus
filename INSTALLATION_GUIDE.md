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

## 🚀 Quick Setup (From Scratch)

### 1. Prepare Remote GPU Server

```bash
# On your GPU server
apt-get update
apt-get install -y git build-essential cmake ocl-icd-opencl-dev opencl-headers python3 python3-venv python3-pip curl

# Verify GPU
nvidia-smi
```

### 2. Get the Code and Build Chronos (GPU Partitioner)

```bash
# Clone repo to /opt
mkdir -p /opt && cd /opt
git clone https://github.com/5ury44/cumulus.git

# Build & install Chronos GPU partitioner
cd /opt/cumulus/chronos_core
bash scripts/install-quick.sh

# Verify Chronos CLI
chronos stats   # or /usr/local/bin/chronos_cli stats
```

### 3. Python Environment + Worker Install

```bash
# Create venv and install server
python3 -m venv /opt/cumulus-env
source /opt/cumulus-env/bin/activate
pip install --upgrade pip wheel setuptools

cd /opt/cumulus
pip install -e .

# Install CUDA-enabled PyTorch
pip install --upgrade torch torchvision
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)
PY
```

### 4. Configure S3 (Distributed Checkpointing + Artifact Store)

The worker automatically loads S3 config from `/opt/cumulus-distributed/.env` if present.

```bash
mkdir -p /opt/cumulus-distributed
cat >/opt/cumulus-distributed/.env << 'EOF'
CUMULUS_S3_BUCKET=your-bucket
CUMULUS_S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# L1 local cache (checkpoints)
CUMULUS_LOCAL_CACHE_DIR=/tmp/cumulus/checkpoints
CUMULUS_CACHE_SIZE_LIMIT_GB=10.0
CUMULUS_KEEP_CHECKPOINTS=5
CUMULUS_CHECKPOINT_EVERY_STEPS=100
CUMULUS_CHECKPOINT_EVERY_SECONDS=300
CUMULUS_AUTO_CLEANUP=true
CUMULUS_ENABLE_JOB_METADATA=true
EOF
```

Required IAM permissions on the bucket: s3:PutObject, s3:GetObject, s3:ListBucket, s3:DeleteObject (optional for cleanup).

### 3. Install Cumulus SDK on Local Machine

```bash
# On your local machine
cd /path/to/cumulus
pip install -e .

# Or install with ML dependencies
pip install -e ".[ml]"
```

## 🔧 Detailed Setup

### Remote Server Setup (Details)

#### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Build Chronos from source (if skipping quick setup above)
cd /opt/cumulus/chronos_core
bash scripts/install-quick.sh
```

#### Step 2: Install Cumulus Worker

```bash
# Clone or copy cumulus directory to server
cd /path/to/cumulus

# Create virtual environment (recommended)
python3 -m venv cumulus-env
source cumulus-env/bin/activate

# Install cumulus worker in venv
pip install -e .
```

#### Step 3: Configure and Start Worker

```bash
# Start worker on a free port (8081 used if 8080 busy)
source /opt/cumulus-env/bin/activate
cd /opt/cumulus
python -m uvicorn worker.server:create_app --factory --host 0.0.0.0 --port 8081
```

#### Step 4: Verify Worker is Running

```bash
# Health + server info
curl http://localhost:8081/health
curl http://localhost:8081/api/info
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

#### Step 2: Test Connection (with SSH tunnel)

````python
From your laptop:

```bash
# Forward local 8080 to server 8081
ssh -p <port> -N -f -L 8080:localhost:8081 root@<ip>
curl -s http://localhost:8080/health
````

```python
from cumulus.sdk import CumulusClient
client = CumulusClient("http://localhost:8080")
print(client.get_server_info())
```

````

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
````

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

## ✅ Validation

### GPU partitioning

```bash
chronos stats
chronos create 0 0.5 3600   # optional, creates 50% partition for 1h
chronos list
```

### End-to-end tests (from laptop)

```bash
# 1) Complete NN training + checkpoint/resume (uploads checkpoints to S3)
python cumulus/tests/test_complete_nn.py

# 2) Artifact store (uploads artifacts to S3 when configured)
python cumulus/tests/test_artifact_store.py
```

Check S3:

- Checkpoints: `s3://<bucket>/checkpoints/<job_id>/...`
- Artifacts: `s3://<bucket>/artifacts/<job_id>/...`

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
# Worker logs (if run with nohup)
tail -f /tmp/cumulus-worker.log 2>/dev/null || echo "Check your process manager logs"

# Chronos status
chronos stats
chronos list

# Check job directories
ls -la /tmp/cumulus_jobs/
```

## ⚙️ Configuration

### Environment Variables

```bash
# Worker configuration
export MAX_CONCURRENT_JOBS=5
export CUMULUS_CHRONOS_PATH="/path/to/chronos_cli"  # optional override
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
  path: "/usr/local/bin/chronos_cli" # or leave blank and use vendored
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
# Check Chronos resolution
cumulus-cli chronos-path

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
