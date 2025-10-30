# Central Orchestrator Design Document

## Overview

This document describes the design of the Central Orchestrator system for Cumulus, which acts as a relay between clients and GPU workers. The orchestrator receives complete job submissions, schedules containers on GPU workers using an external orchestration platform (Nomad or Docker Swarm), and streams results back to clients.

### Design Principles

1. **Orchestrator as Relay**: All communication flows through the orchestrator (no direct client-worker connections)
2. **Platform-Managed Workers**: Leverage existing orchestration platforms for worker management, health checks, and scheduling
3. **Container-Based Execution**: Jobs run in Docker containers with GPU access via Chronos partitions
4. **One-Script Onboarding**: GPU workers join the pool by running a single setup script
5. **Simplicity First**: Minimal custom code, maximum use of proven platforms

## Architecture

### System Components

```
┌─────────────────┐
│  Client SDK     │
│  (Python)       │
└────────┬────────┘
         │ 1. SubmitJob(code, requirements, gpu_memory, duration)
         ↓
┌──────────────────────────────────────────┐
│  Central Orchestrator (Go/Python)        │
│  - Receives job submissions              │
│  - Creates Chronos GPU partitions        │
│  - Schedules containers via platform API │
│  - Streams results back to client        │
└────────┬─────────────────────────────────┘
         │ 2. Create Chronos partition
         │ 3. Submit container spec
         ↓
┌──────────────────────────────────────────┐
│  Orchestration Platform                  │
│  (Nomad or Docker Swarm)                 │
│  - Worker pool management                │
│  - Health checks & heartbeats            │
│  - Container scheduling                  │
│  - Service discovery                     │
└────────┬─────────────────────────────────┘
         │ 4. Schedule container on worker
         ↓
┌─────────────────────────────────┐
│  GPU Worker (Docker Host)       │
│  - Chronos library installed    │
│  - Docker daemon                │
│  - Platform agent (Nomad/Swarm) │
└────────┬────────────────────────┘
         │ 5. Run container in partition
         ↓
┌─────────────────────────────────┐
│  Job Container                  │
│  - Python + OpenCL              │
│  - User code                    │
│  - Inherits GPU from partition  │
└─────────────────────────────────┘
         │ 6. Results → stdout
         ↓
    [Platform → Orchestrator → Client]
```

### Communication Flow

1. **Job Submission**
   - Client → Orchestrator: Submit complete job (code ZIP, requirements, GPU specs)
   - Orchestrator validates job and checks worker availability via platform API

2. **Partition Creation**
   - Orchestrator → Worker: Create Chronos partition (via SSH/API)
   - Worker: `chronos_cli create <device> <memory> <duration>`
   - Worker → Orchestrator: Return partition ID

3. **Container Scheduling**
   - Orchestrator → Platform: Submit container spec with partition context
   - Platform: Schedule container on worker with active partition
   - Worker: Launch container with GPU access from partition

4. **Execution & Results**
   - Container: Extract code, install requirements, execute
   - Container: Write results to stdout
   - Platform → Orchestrator: Stream container logs
   - Orchestrator → Client: Stream progress and results

5. **Cleanup**
   - Container exits (success or failure)
   - Platform: Remove container automatically
   - Orchestrator → Worker: Release Chronos partition
   - Worker: `chronos_cli release <partition_id>`

**Key Benefits:**
- ✅ Single client endpoint (orchestrator)
- ✅ Platform handles worker management
- ✅ Orchestrator can kill any job (via platform API)
- ✅ No custom authentication needed
- ✅ Proven reliability from platform

## Development Setup (Docker Context + Vast.ai)

### Overview

For development, we use **Docker Context** to run the orchestrator on a remote Vast.ai instance while developing locally. This solves networking issues (no port forwarding needed) and provides a seamless dev experience where local commands execute remotely.

### How It Works

Docker Context allows you to point your local Docker CLI to a remote Docker daemon via SSH. All `docker` commands execute on the remote machine transparently.

**Benefits:**
- ✅ No port forwarding or VPN needed
- ✅ Build and run on remote machine from local terminal
- ✅ Kill local process (Ctrl+C) → kills remote container
- ✅ Cheap (~$0.05-0.10/hour for CPU instance)
- ✅ Same platform as GPU workers (Vast.ai)
- ✅ Public IP for Swarm manager

### Setup Steps

#### 1. Rent Vast.ai CPU Instance (Orchestrator Host)

```bash
# Install Vast.ai CLI
pip install vastai

# Login
vastai set api-key <YOUR_API_KEY>

# Search for cheap CPU instances (no GPU needed)
vastai search offers "cpu_cores>=2 cpu_ram>=4 disk_space>=20" --order "dph+"

# Rent instance (note the instance ID)
vastai create instance <OFFER_ID> --image nvidia/cuda:12.0.0-base-ubuntu22.04 --disk 20

# Get SSH connection info
vastai show instance <INSTANCE_ID>
# Note the ssh_host and ssh_port
```

#### 2. Configure Docker Context

```bash
# Get SSH connection details from Vast.ai
VAST_SSH_HOST=$(vastai show instance <INSTANCE_ID> | jq -r '.ssh_host')
VAST_SSH_PORT=$(vastai show instance <INSTANCE_ID> | jq -r '.ssh_port')

# Create Docker context pointing directly to Vast.ai instance (no SSH config needed!)
docker context create vast-orchestrator --docker "host=ssh://root@$VAST_SSH_HOST:$VAST_SSH_PORT"

# Switch to remote context
docker context use vast-orchestrator

# Verify connection
docker info
# Should show remote machine details
```

#### 3. Initialize Docker Swarm on Remote Instance

```bash
# Get the public IP of the Vast.ai instance
VAST_PUBLIC_IP=$(vastai show instance <INSTANCE_ID> | jq -r '.public_ipaddr')

# Initialize Swarm (runs on remote machine via Docker context)
docker swarm init --advertise-addr $VAST_PUBLIC_IP

# Save the worker join token
SWARM_TOKEN=$(docker swarm join-token worker -q)
echo "Worker join command:"
echo "docker swarm join --token $SWARM_TOKEN $VAST_PUBLIC_IP:2377"
```

#### 4. Development Workflow

```bash
# All commands run on remote Vast.ai instance automatically

# Build orchestrator (builds remotely)
cd orchestrator
docker build -t cumulus-orchestrator .

# Run orchestrator (runs remotely)
docker run -d \
  --name orchestrator \
  -p 50051:50051 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  cumulus-orchestrator

# View logs (from remote)
docker logs -f orchestrator

# Make code changes locally, rebuild
docker build -t cumulus-orchestrator .
docker restart orchestrator

# Stop orchestrator
docker stop orchestrator

# Switch back to local Docker when done
docker context use default
```

#### 5. Add GPU Workers

GPU workers join the Swarm using the public IP and token:

```bash
# On GPU worker (Vast.ai GPU instance)
docker swarm join --token $SWARM_TOKEN $VAST_PUBLIC_IP:2377
```

### Development Scripts

**`scripts/dev-setup.sh`** - One-time setup
```bash
#!/bin/bash
set -e

echo "🚀 Setting up Cumulus development environment"

# Check if vastai is installed
if ! command -v vastai &> /dev/null; then
    echo "Installing Vast.ai CLI..."
    pip install vastai
fi

# Prompt for API key if not set
if [ -z "$VASTAI_API_KEY" ]; then
    echo "Enter your Vast.ai API key:"
    read -r VASTAI_API_KEY
    vastai set api-key "$VASTAI_API_KEY"
fi

# Search for cheap CPU instance
echo "🔍 Finding cheap CPU instance..."
OFFER_ID=$(vastai search offers "cpu_cores>=2 cpu_ram>=4 disk_space>=20" --order "dph+" --raw | jq -r '.[0].id')

echo "💰 Renting instance (Offer ID: $OFFER_ID)..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" --image nvidia/cuda:12.0.0-base-ubuntu22.04 --disk 20 --raw | jq -r '.new_contract')

echo "⏳ Waiting for instance to start..."
sleep 30

# Get connection info
INSTANCE_INFO=$(vastai show instance "$INSTANCE_ID" --raw)
SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host')
SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port')
PUBLIC_IP=$(echo "$INSTANCE_INFO" | jq -r '.public_ipaddr')

# Save instance info
cat > .dev-instance << EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
PUBLIC_IP=$PUBLIC_IP
EOF

echo "✅ Instance created: $INSTANCE_ID"
echo "📝 Connection info saved to .dev-instance"

# Wait for SSH to be ready
echo "⏳ Waiting for SSH..."
SSH_CONNECTION="ssh://root@$SSH_HOST:$SSH_PORT"
for i in {1..30}; do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "echo ready" &> /dev/null; then
        echo "✅ SSH ready"
        break
    fi
    sleep 2
done

# Install Docker on remote instance
echo "🐳 Installing Docker on remote instance..."
ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "curl -fsSL https://get.docker.com | sh"

# Configure Docker daemon for remote access (needed for Swarm)
echo "🔧 Configuring Docker daemon..."
ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "mkdir -p /etc/docker && cat > /etc/docker/daemon.json << 'DOCKER_EOF'
{
  \"hosts\": [\"unix:///var/run/docker.sock\", \"tcp://0.0.0.0:2375\"],
  \"log-driver\": \"json-file\",
  \"log-opts\": {
    \"max-size\": \"10m\",
    \"max-file\": \"3\"
  }
}
DOCKER_EOF
systemctl restart docker"

# Create Docker context (using direct SSH connection - no SSH config needed!)
echo "🔗 Creating Docker context..."
docker context create vast-orchestrator --docker "host=$SSH_CONNECTION" 2>/dev/null || true
docker context use vast-orchestrator

# Initialize Swarm
echo "🐝 Initializing Docker Swarm..."
docker swarm init --advertise-addr "$PUBLIC_IP" || true

# Get join token
SWARM_TOKEN=$(docker swarm join-token worker -q)

# Save to .dev-instance
cat >> .dev-instance << EOF
SWARM_TOKEN=$SWARM_TOKEN
SWARM_MANAGER=$PUBLIC_IP:2377
EOF

echo ""
echo "✅ Development environment ready!"
echo ""
echo "📋 Instance Info:"
echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP: $PUBLIC_IP"
echo "   SSH: ssh vast-orchestrator"
echo ""
echo "🐝 Swarm Info:"
echo "   Manager: $PUBLIC_IP:2377"
echo "   Worker join command:"
echo "   docker swarm join --token $SWARM_TOKEN $PUBLIC_IP:2377"
echo ""
echo "🚀 Next steps:"
echo "   1. Build orchestrator: docker build -t cumulus-orchestrator ./orchestrator"
echo "   2. Run orchestrator: docker run -d -p 50051:50051 cumulus-orchestrator"
echo "   3. Add GPU workers using the join command above"
echo ""
echo "💡 All docker commands now run on remote instance!"
```

**`scripts/dev-start.sh`** - Start orchestrator
```bash
#!/bin/bash
set -e

# Load instance info
source .dev-instance

# Switch to remote context
docker context use vast-orchestrator

echo "🏗️  Building orchestrator..."
docker build -t cumulus-orchestrator ./orchestrator

echo "🚀 Starting orchestrator..."
docker run -d \
  --name orchestrator \
  -p 50051:50051 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  cumulus-orchestrator

echo "✅ Orchestrator running on $PUBLIC_IP:50051"
echo "📋 View logs: docker logs -f orchestrator"
```

**`scripts/dev-stop.sh`** - Stop and cleanup
```bash
#!/bin/bash
set -e

# Load instance info
source .dev-instance

# Switch to remote context
docker context use vast-orchestrator

echo "🛑 Stopping orchestrator..."
docker stop orchestrator 2>/dev/null || true
docker rm orchestrator 2>/dev/null || true

echo "🗑️  Destroying Vast.ai instance..."
vastai destroy instance "$INSTANCE_ID"

# Remove Docker context
docker context use default
docker context rm vast-orchestrator 2>/dev/null || true

# Clean up files
rm -f .dev-instance

echo "✅ Development environment cleaned up"
```

**`scripts/dev-logs.sh`** - View logs
```bash
#!/bin/bash
docker context use vast-orchestrator
docker logs -f orchestrator
```

### Usage

```bash
# One-time setup
./scripts/dev-setup.sh

# Start orchestrator (after code changes)
./scripts/dev-start.sh

# View logs
./scripts/dev-logs.sh

# Stop and cleanup
./scripts/dev-stop.sh
```

### Cost Estimation

- **CPU Instance (Orchestrator)**: ~$0.05-0.10/hour
- **GPU Instance (Worker)**: ~$0.20-0.50/hour (depending on GPU)
- **Typical dev session (4 hours)**: ~$1-2 total

### Troubleshooting

**Docker context not connecting:**
```bash
# Test SSH connection
ssh vast-orchestrator

# Recreate context
docker context rm vast-orchestrator
docker context create vast-orchestrator --docker "host=ssh://vast-orchestrator"
```

**Swarm init fails:**
```bash
# Check if Swarm already initialized
docker info | grep Swarm

# Force leave and reinit
docker swarm leave --force
docker swarm init --advertise-addr $PUBLIC_IP
```

**Can't connect to orchestrator:**
```bash
# Check if port is open
docker ps | grep orchestrator

# Check logs
docker logs orchestrator
```


## Project Structure

### Current Structure
```
cumulus/
├── sdk/                    # Client SDK (Python) - MINOR UPDATES
│   ├── client.py          # Main client class - update endpoint
│   ├── code_packager.py   # Code packaging - keep as-is
│   ├── decorators.py      # Decorators - keep as-is
│   └── ...
├── worker/                 # Current Python worker - REMOVED
│   └── ...                # Replaced by platform + containers
├── cumulus_core/          # Chronos C++ library - UNCHANGED
│   └── ...                # GPU partitioning library
└── tests/                 # Tests - UPDATED
    └── ...
```

### New Structure (After Implementation)
```
cumulus/
├── orchestrator/          # NEW: Central orchestrator
│   ├── main.go (or main.py)
│   ├── platform/          # Platform API clients
│   │   ├── nomad.go       # Nomad client
│   │   └── swarm.go       # Docker Swarm client
│   ├── chronos/           # Chronos partition management
│   │   └── manager.go     # Create/release partitions via SSH
│   ├── jobs/              # Job lifecycle management
│   │   └── handler.go     # Job submission, monitoring
│   └── config.yaml        # Orchestrator configuration
├── containers/            # NEW: Job container image
│   ├── Dockerfile         # Base image with Python + OpenCL
│   ├── entrypoint.sh      # Container entry point
│   └── runner.py          # Job execution script
├── scripts/               # NEW: Worker setup
│   └── setup-worker.sh    # One-script worker onboarding
├── sdk/                   # UPDATED: Python SDK
│   ├── client.py          # Updated to call orchestrator
│   └── ...                # Minimal changes
├── cumulus_core/          # UNCHANGED: Chronos C++ library
└── tests/                 # UPDATED: New integration tests
    └── test_orchestrator.py
```

## Components and Interfaces

### 1. Central Orchestrator - New Component

**Location**: `orchestrator/`

#### Responsibilities
- Receive complete job submissions from clients
- Query orchestration platform for available workers
- Create Chronos GPU partitions on selected workers
- Schedule job containers via platform API
- Stream container logs and results back to clients
- Clean up partitions after job completion

#### Core Modules

**SwarmClient** (Docker Swarm integration)
```go
import (
    "github.com/docker/docker/client"
    "github.com/docker/docker/api/types/swarm"
)

type SwarmClient struct {
    cli *client.Client
}

// ListWorkers lists all Swarm nodes with GPU resources
func (s *SwarmClient) ListWorkers() ([]*WorkerInfo, error) {
    nodes, err := s.cli.NodeList(ctx, types.NodeListOptions{})
    // Filter nodes with GPU labels
    // Return worker info
}

// CreateService creates a Docker service (container) on Swarm
func (s *SwarmClient) CreateService(spec *ServiceSpec) (string, error) {
    serviceSpec := swarm.ServiceSpec{
        TaskTemplate: swarm.TaskSpec{
            ContainerSpec: &swarm.ContainerSpec{
                Image:   spec.Image,
                Command: spec.Command,
                Env:     spec.Environment,
            },
            Placement: &swarm.Placement{
                Constraints: []string{
                    fmt.Sprintf("node.id == %s", spec.NodeID),
                },
            },
        },
    }
    
    service, err := s.cli.ServiceCreate(ctx, serviceSpec, types.ServiceCreateOptions{})
    return service.ID, err
}

// GetServiceLogs streams logs from a service
func (s *SwarmClient) GetServiceLogs(serviceID string) (io.ReadCloser, error) {
    return s.cli.ServiceLogs(ctx, serviceID, types.ContainerLogsOptions{
        ShowStdout: true,
        ShowStderr: true,
        Follow:     true,
    })
}

// RemoveService stops and removes a service
func (s *SwarmClient) RemoveService(serviceID string) error {
    return s.cli.ServiceRemove(ctx, serviceID)
}

type WorkerInfo struct {
    ID         string
    Hostname   string
    Address    string
    GPUDevices []GPUDevice
    Status     string  // "ready", "down"
}

type ServiceSpec struct {
    Image       string
    Command     []string
    Environment []string
    NodeID      string  // Target Swarm node
    Mounts      []Mount
}
```

**ChronosManager** (manages partitions on workers)
```go
type ChronosManager struct {
    sshClients map[string]*ssh.Client  // Worker ID -> SSH connection
}

// CreatePartition creates a Chronos partition on a worker
func (m *ChronosManager) CreatePartition(workerID string, device int, 
                                         memory float64, duration int) (string, error) {
    // SSH to worker
    // Execute: chronos_cli create <device> <memory> <duration>
    // Parse partition ID from output
    // Return partition ID
}

// ReleasePartition releases a Chronos partition
func (m *ChronosManager) ReleasePartition(workerID string, partitionID string) error {
    // SSH to worker
    // Execute: chronos_cli release <partition_id>
}
```

**JobHandler** (manages job lifecycle)
```go
type JobHandler struct {
    platform PlatformClient
    chronos  *ChronosManager
    jobs     map[string]*JobInfo
    mu       sync.RWMutex
}

type JobInfo struct {
    ID          string
    WorkerID    string
    PartitionID string
    ContainerID string
    Status      string
    CreatedAt   time.Time
}

// SubmitJob handles a complete job submission
func (h *JobHandler) SubmitJob(req *JobSubmission) (*JobResult, error) {
    // 1. Select worker from platform
    // 2. Create Chronos partition on worker
    // 3. Build container spec with partition context
    // 4. Submit container to platform
    // 5. Stream logs to client
    // 6. Wait for completion
    // 7. Release partition
    // 8. Return results
}
```

#### gRPC Service Definition

```protobuf
service OrchestratorService {
    // Submit a job and stream results
    rpc SubmitJob(JobSubmission) returns (stream JobEvent);
    
    // Get job status
    rpc GetJobStatus(JobStatusRequest) returns (JobStatus);
    
    // Cancel a running job
    rpc CancelJob(JobCancelRequest) returns (Empty);
    
    // List workers (admin)
    rpc ListWorkers(Empty) returns (WorkerList);
    
    // Health check
    rpc HealthCheck(Empty) returns (HealthStatus);
}

message JobSubmission {
    bytes code_package = 1;  // ZIP file
    repeated string requirements = 2;
    float gpu_amount = 3;  // 0.1-0.9 (fractional) or 1-8 (whole GPUs)
    int32 duration = 4;    // seconds in range [60, 86400]
}

message JobEvent {
    string job_id = 1;
    JobState state = 2;
    string message = 3;
    bytes result = 4;  // Final result (JSON)
    float progress = 5;
}

enum JobState {
    SCHEDULING = 0;
    CREATING_PARTITION = 1;
    RUNNING = 2;
    COMPLETED = 3;
    FAILED = 4;
    CANCELLED = 5;
}
```


### 2. GPU Worker (Docker Host) - Simplified

**Note**: Workers are now just Docker hosts with Chronos installed. No custom worker service needed!

#### Responsibilities
- Run Docker daemon
- Run orchestration platform agent (Nomad client or Swarm node)
- Have Chronos library installed and accessible
- Report health to platform automatically
- Execute containers scheduled by platform

#### Setup Components

**Docker Swarm Node**
- Joins Swarm cluster as worker node
- Registers with Swarm manager automatically
- Reports node resources via labels (GPU info)
- Receives service scheduling requests
- Handles health checks and heartbeats automatically

**Chronos Installation**
- Chronos C++ library installed system-wide
- `chronos_cli` binary in PATH
- OpenCL drivers and GPU access configured

**Docker Configuration**
- Docker daemon with GPU support
- Access to Chronos partitions via environment variables
- Network connectivity to orchestrator

#### Worker Setup Script

```bash
#!/bin/bash
# setup-worker.sh - One-script worker onboarding for Docker Swarm

SWARM_MANAGER=$1
SWARM_TOKEN=$2

# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 1.5. Configure Docker daemon for Swarm communication
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'EOF'
{
  "hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2375"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
systemctl restart docker

# 2. Install Chronos
cd /tmp
git clone https://github.com/your-org/cumulus.git
cd cumulus/cumulus_core
./install-quick.sh

# 3. Detect GPU and add node labels
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# 4. Join Docker Swarm
docker swarm join --token $SWARM_TOKEN $SWARM_MANAGER

# 5. Label this node with GPU info (run on manager)
# Note: This needs to be run on the manager node after worker joins
# docker node update --label-add gpu.count=$GPU_COUNT <node-id>
# docker node update --label-add gpu.name="$GPU_NAME" <node-id>
# docker node update --label-add gpu.memory=$GPU_MEMORY <node-id>

echo "Worker setup complete! This node has joined the Swarm."
echo "Run these commands on the Swarm manager to label this node:"
echo "  NODE_ID=\$(docker node ls --filter role=worker -q | tail -1)"
echo "  docker node update --label-add gpu.count=$GPU_COUNT \$NODE_ID"
echo "  docker node update --label-add gpu.name=\"$GPU_NAME\" \$NODE_ID"
echo "  docker node update --label-add gpu.memory=$GPU_MEMORY \$NODE_ID"
```


### 3. Client SDK (Python) - `sdk/`

#### Updated Architecture

The existing `sdk/` directory contains the client-side Python SDK. It will be updated to submit complete jobs to the orchestrator and receive streaming results. The user-facing API remains largely unchanged.

#### Core Classes

**CumulusClient** (updated `sdk/client.py`)
```python
import grpc
from cumulus_proto import orchestrator_pb2, orchestrator_pb2_grpc

class CumulusClient:
    def __init__(self, orchestrator_url: str):
        self.channel = grpc.insecure_channel(orchestrator_url)
        self.stub = orchestrator_pb2_grpc.OrchestratorServiceStub(self.channel)
    
    def run(self, func: Callable, gpu_memory: float = 0.5, 
            duration: int = 3600, requirements: List[str] = None, 
            timeout: Optional[int] = None) -> Any:
        # 1. Package code
        packager = CodePackager()
        code_zip = packager.package_function(func, requirements or [])
        
        # 2. Submit job to orchestrator via gRPC
        submission = orchestrator_pb2.JobSubmission(
            code_package=code_zip,
            requirements=requirements or [],
            gpu_memory=gpu_memory,
            duration=duration
        )
        
        # 3. Stream results from orchestrator
        result = self._stream_results(submission, timeout)
        
        return result
    
    def _stream_results(self, submission, timeout: Optional[int]) -> Any:
        # Stream events from orchestrator
        for event in self.stub.SubmitJob(submission, timeout=timeout):
            if event.state == orchestrator_pb2.COMPLETED:
                return json.loads(event.result)
            elif event.state == orchestrator_pb2.FAILED:
                raise RuntimeError(f"Job failed: {event.message}")
            elif event.state == orchestrator_pb2.RUNNING:
                # Optional: callback for progress updates
                pass
        
        raise RuntimeError("Stream ended unexpectedly")
```

**Key Changes:**
- ✅ gRPC for type-safe communication
- ✅ Single endpoint (orchestrator only)
- ✅ Complete job submission in one call
- ✅ Streaming results through orchestrator
- ✅ No worker connections needed
- ✅ Minimal API changes for users

### 4. Job Container - New Component

**Location**: `containers/`

#### Responsibilities
- Provide consistent execution environment for all jobs
- Extract and execute user code
- Install Python requirements
- Inherit GPU access from Chronos partition
- Write results to stdout for orchestrator capture

#### Container Image

**Dockerfile**
```dockerfile
FROM python:3.10-slim

# Install OpenCL and common dependencies
RUN apt-get update && apt-get install -y \
    ocl-icd-opencl-dev \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# Install common ML libraries (optional, can be in requirements)
RUN pip install --no-cache-dir \
    numpy \
    torch \
    tensorflow

# Copy job runner script
COPY runner.py /app/runner.py
COPY entrypoint.sh /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
```

**entrypoint.sh**
```bash
#!/bin/bash
set -e

# Job code is mounted at /job
cd /job

# Extract code package
unzip -q code.zip

# Install requirements
if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
fi

# Run job (GPU access inherited from Chronos partition)
python runner.py

# Results written to stdout in JSON format
```

**runner.py**
```python
import json
import sys
import traceback

def main():
    try:
        # Import and execute user's main function
        from main import main as user_main
        
        result = user_main()
        
        # Write result to stdout
        print(json.dumps({
            "status": "success",
            "result": result
        }))
        
    except Exception as e:
        # Write error to stdout
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
```


## Data Models

### GPU Resource Allocation

**Allowed GPU Values:**
- Fractional: `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`, `0.8`, `0.9`
- Whole GPUs: `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`

**Invalid Values:** `0`, `2.3`, `0.1482`, `1.5`, etc.

**Resource Tracking:**
```go
type GPUDevice struct {
    Index           int
    Name            string
    TotalCapacity   float64  // Always 1.0 per physical GPU
    AllocatedAmount float64  // Sum of all active partitions
    ActivePartitions []*PartitionInfo
}

type PartitionInfo struct {
    ID          string
    JobID       string
    Amount      float64  // 0.1-1.0 for fractional, 1+ for multi-GPU
    CreatedAt   time.Time
    ExpiresAt   time.Time
}

// Example: Worker with 2 GPUs
// GPU 0: Total=1.0, Allocated=0.7 (0.5 + 0.2), Available=0.3
// GPU 1: Total=1.0, Allocated=0.0, Available=1.0
```

**Scheduling Logic:**
```go
func (w *WorkerInfo) CanAllocate(gpuAmount float64) (deviceIndex int, ok bool) {
    if gpuAmount < 1.0 {
        // Fractional GPU: find device with enough free capacity
        for i, gpu := range w.GPUDevices {
            available := gpu.TotalCapacity - gpu.AllocatedAmount
            if available >= gpuAmount {
                return i, true
            }
        }
    } else {
        // Whole GPU(s): find contiguous free GPUs
        needed := int(gpuAmount)
        freeCount := 0
        startIdx := -1
        
        for i, gpu := range w.GPUDevices {
            if gpu.AllocatedAmount == 0 {
                if startIdx == -1 {
                    startIdx = i
                }
                freeCount++
                if freeCount == needed {
                    return startIdx, true
                }
            } else {
                freeCount = 0
                startIdx = -1
            }
        }
    }
    return -1, false
}
```

### Orchestrator Configuration (YAML)

```yaml
orchestrator:
  grpc_port: 50051
  swarm_manager: "tcp://localhost:2377"
  
  # SSH config for Chronos partition management
  ssh:
    key_path: "/root/.ssh/id_rsa"
    user: "root"
    timeout: 10s

  # Job container config
  container:
    image: "cumulus-job:latest"
    registry: "docker.io/cumulus"

  # Resource limits
  max_job_duration: 86400  # 24 hours
  default_duration: 3600   # 1 hour

logging:
  level: "info"
  format: "json"
```

### Worker Node Labels (Docker Swarm)

Workers are labeled in Swarm with GPU information:

```bash
# Labels added to Swarm nodes
docker node update --label-add gpu.count=2 worker-1
docker node update --label-add gpu.0.name="NVIDIA A100" worker-1
docker node update --label-add gpu.0.memory=40960 worker-1
docker node update --label-add gpu.1.name="NVIDIA A100" worker-1
docker node update --label-add gpu.1.memory=40960 worker-1
docker node update --label-add chronos.installed=true worker-1
docker node update --label-add ssh.address="10.0.1.10" worker-1
```

### Job Tracking

```go
type JobInfo struct {
    ID          string
    ServiceID   string  // Docker service ID
    WorkerID    string  // Swarm node ID
    PartitionID string  // Chronos partition ID
    GPUDevice   int     // GPU device index
    GPUAmount   float64 // Allocated GPU amount
    Status      JobState
    CreatedAt   time.Time
    StartedAt   time.Time
    CompletedAt time.Time
}

// In-memory job tracking
type JobTracker struct {
    jobs map[string]*JobInfo
    mu   sync.RWMutex
}
```

### Container Volume Structure

```
# Mounted into container at /job
/job/
├── code.zip           # User code package
├── main.py            # Extracted entry point (after unzip)
├── requirements.txt   # Python dependencies (after unzip)
└── ...                # Other user files (after unzip)

# Container writes to stdout:
{"status": "success", "result": {...}}
# or
{"status": "error", "error": "...", "traceback": "..."}
```


## Error Handling

### Error Types and Recovery

**1. No Workers Available**
- gRPC Code: `UNAVAILABLE`
- Message: "No GPU workers available in Swarm cluster"
- Client Action: Retry with exponential backoff or fail

**2. Insufficient GPU Resources**
- gRPC Code: `RESOURCE_EXHAUSTED`
- Message: "No workers with sufficient GPU capacity (requested: X, max available: Y)"
- Client Action: Reduce GPU requirements or wait

**3. Invalid GPU Amount**
- gRPC Code: `INVALID_ARGUMENT`
- Message: "Invalid GPU amount: X. Must be 0.1-0.9 or whole number 1-8"
- Client Action: Fix GPU amount and retry

**4. Worker SSH Failure**
- gRPC Code: `UNAVAILABLE`
- Message: "Failed to create Chronos partition on worker: SSH connection failed"
- Client Action: Orchestrator retries on different worker

**5. Chronos Partition Creation Failed**
- gRPC Code: `INTERNAL`
- Message: "Chronos partition creation failed: <chronos_cli error>"
- Client Action: Orchestrator retries on different worker

**6. Container Scheduling Failed**
- gRPC Code: `INTERNAL`
- Message: "Docker Swarm failed to schedule container: <swarm error>"
- Client Action: Orchestrator retries or returns error

**7. Job Execution Failure**
- gRPC Code: `INTERNAL`
- Message: Detailed error from container stdout
- Client Action: Fix code and retry

**8. Job Timeout**
- gRPC Code: `DEADLINE_EXCEEDED`
- Message: "Job exceeded allocated duration of X seconds"
- Client Action: Increase duration or optimize code

### Retry Logic

**Orchestrator → Worker SSH**
```go
func (cm *ChronosManager) CreatePartitionWithRetry(workerID string, device int, 
                                                   amount float64, duration int) (string, error) {
    maxRetries := 3
    backoff := time.Second
    
    for i := 0; i < maxRetries; i++ {
        partitionID, err := cm.CreatePartition(workerID, device, amount, duration)
        if err == nil {
            return partitionID, nil
        }
        
        log.Warn().Err(err).Int("attempt", i+1).Msg("Chronos partition creation failed")
        
        if i < maxRetries-1 {
            time.Sleep(backoff)
            backoff *= 2
        }
    }
    
    return "", fmt.Errorf("failed to create partition after %d attempts", maxRetries)
}
```

**Client → Orchestrator**
```python
def run_with_retry(self, func: Callable, gpu_amount: float, duration: int,
                   max_retries: int = 3) -> Any:
    backoff = 1.0
    
    for i in range(max_retries):
        try:
            return self.run(func, gpu_amount, duration)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                if i < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise
            else:
                # Don't retry on other errors
                raise
```

### Resource Cleanup on Failure

```go
func (h *JobHandler) cleanupJob(jobID string) {
    job, ok := h.jobs[jobID]
    if !ok {
        return
    }
    
    // Release Chronos partition
    if job.PartitionID != "" {
        err := h.chronos.ReleasePartition(job.WorkerID, job.PartitionID)
        if err != nil {
            log.Error().Err(err).Str("partition_id", job.PartitionID).Msg("Failed to release partition")
        }
    }
    
    // Remove Docker service
    if job.ServiceID != "" {
        err := h.swarm.RemoveService(job.ServiceID)
        if err != nil {
            log.Error().Err(err).Str("service_id", job.ServiceID).Msg("Failed to remove service")
        }
    }
    
    // Update job status
    job.Status = JobState_FAILED
    job.CompletedAt = time.Now()
}
```


## Testing Strategy

### Unit Tests

**Orchestrator (Go)**
- GPU resource allocation algorithm (fractional and whole GPU)
- Worker selection with various GPU availability scenarios
- GPU amount validation (0.1-0.9, 1-8)
- Resource tracking and cleanup
- gRPC message serialization

**ChronosManager (Go)**
- SSH connection management
- Partition creation command parsing
- Partition release logic
- Error handling for SSH failures

**SwarmClient (Go)**
- Node listing and filtering
- Service creation with constraints
- Log streaming
- Service removal

**Client SDK (Python)**
- Code packaging (existing tests)
- gRPC stub creation
- Event stream handling
- Error handling and retries

### Integration Tests

**End-to-End Flow**
1. Start local Docker Swarm (single node)
2. Start orchestrator connected to Swarm
3. Client submits simple job (return 2+2)
4. Verify job executes and returns result
5. Verify partition created and released
6. Verify service created and removed

**GPU Resource Allocation**
1. Submit job requesting 0.5 GPU
2. Submit another job requesting 0.3 GPU (should succeed)
3. Submit third job requesting 0.5 GPU (should fail or queue)
4. First job completes, third job should start
5. Verify GPU tracking is accurate

**Multi-Worker Scenarios**
1. Add second worker to Swarm
2. Submit multiple jobs simultaneously
3. Verify jobs distributed across workers
4. Verify each worker tracks GPU independently

**Error Scenarios**
1. Submit job with invalid GPU amount (2.3)
2. Submit job when no workers available
3. Kill container mid-execution
4. SSH failure to worker
5. Chronos partition creation failure

### Performance Tests

**Orchestrator Throughput**
- Measure job submissions per second
- Target: 100+ concurrent jobs

**GPU Allocation Efficiency**
- Submit mix of fractional and whole GPU jobs
- Measure GPU utilization across workers
- Target: >80% GPU utilization

**End-to-End Latency**
- Job submission to container start: < 5 seconds
- Partition creation: < 2 seconds
- Small job (1 second compute): < 10 seconds total
- Log streaming latency: < 1 second


## Security Considerations

### Token Security

**Token Generation**
```go
func generateJobToken() string {
    // Use crypto/rand for secure random bytes
    b := make([]byte, 32)  // 256 bits
    rand.Read(b)
    return base64.URLEncoding.EncodeToString(b)
}
```

**Token Validation**
- Tokens are single-use (invalidated after job completion)
- Tokens expire after configured duration (default: 1 hour)
- Tokens are bound to specific worker ID
- Tokens include job resource limits

### Network Security

**TLS Support**
```go
// Orchestrator with TLS
creds, err := credentials.NewServerTLSFromFile(certFile, keyFile)
server := grpc.NewServer(grpc.Creds(creds))

// Worker with TLS
creds, err := credentials.NewClientTLSFromFile(caFile, "")
conn, err := grpc.Dial(address, grpc.WithTransportCredentials(creds))
```

**Mutual TLS (mTLS)**
- Workers authenticate to orchestrator with client certificates
- Prevents rogue workers from joining pool
- Orchestrator validates worker certificates

### Code Execution Isolation

**Process Isolation**
- Each job runs in separate process
- Resource limits via cgroups (optional)
- Chronos provides GPU memory isolation

**Filesystem Isolation**
- Jobs run in isolated directories
- No access to other job directories
- Cleanup after job completion

**Network Isolation (Future)**
- Jobs run in network namespace
- No outbound network access by default
- Whitelist allowed endpoints


## Monitoring and Observability

### Metrics

**Orchestrator Metrics**
```go
// Prometheus metrics
var (
    workerAssignments = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "cumulus_worker_assignments_total",
            Help: "Total number of worker assignments",
        },
        []string{"status"},  // success, no_workers, insufficient_resources
    )
    
    activeWorkers = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "cumulus_active_workers",
            Help: "Number of active workers",
        },
    )
    
    assignmentDuration = prometheus.NewHistogram(
        prometheus.HistogramOpts{
            Name: "cumulus_assignment_duration_seconds",
            Help: "Time to assign worker",
            Buckets: prometheus.DefBuckets,
        },
    )
)
```

**Worker Metrics**
```go
var (
    activeJobs = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "cumulus_worker_active_jobs",
            Help: "Number of active jobs on worker",
        },
    )
    
    jobDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "cumulus_job_duration_seconds",
            Help: "Job execution duration",
            Buckets: []float64{1, 5, 10, 30, 60, 300, 600, 1800, 3600},
        },
        []string{"status"},  // completed, failed, cancelled
    )
    
    gpuMemoryUsed = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "cumulus_gpu_memory_used_bytes",
            Help: "GPU memory used",
        },
        []string{"device"},
    )
)
```

### Logging

**Structured Logging**
```go
// Use zerolog or zap for structured logging
log.Info().
    Str("job_id", jobID).
    Str("worker_id", workerID).
    Float64("gpu_memory", gpuMemory).
    Int32("duration", duration).
    Msg("Worker assigned")

log.Error().
    Err(err).
    Str("job_id", jobID).
    Msg("Job execution failed")
```

**Log Levels**
- DEBUG: Detailed flow, token validation, health checks
- INFO: Worker assignments, job lifecycle events
- WARN: Retry attempts, worker unavailable
- ERROR: Job failures, worker crashes, system errors

### Tracing

**OpenTelemetry Integration**
```go
// Trace worker assignment flow
ctx, span := tracer.Start(ctx, "RequestWorker")
defer span.End()

span.SetAttributes(
    attribute.Float64("gpu_memory", req.GpuMemory),
    attribute.Int("duration", int(req.Duration)),
)

// Trace spans:
// - RequestWorker
//   - SelectWorker
//   - CreatePartition
//   - GenerateToken
```


## Deployment

### Build Process

**Orchestrator Binary**
```bash
cd orchestrator
go build -o cumulus-orchestrator main.go

# Or with version info
go build -ldflags "-X main.version=1.0.0" -o cumulus-orchestrator main.go
```

**Container Image**
```bash
cd containers
docker build -t cumulus-job:latest .
docker tag cumulus-job:latest your-registry/cumulus-job:latest
docker push your-registry/cumulus-job:latest
```

**Release Artifacts**
- `cumulus-orchestrator-linux-amd64` - Orchestrator binary
- `cumulus-job:latest` - Job container image
- `setup-worker.sh` - Worker onboarding script
- `config.yaml.example` - Configuration template
- `orchestrator.proto` - gRPC definitions

### Docker Swarm Setup

**Initialize Swarm Manager (Orchestrator Host)**
```bash
# Initialize Swarm
docker swarm init --advertise-addr <MANAGER-IP>

# Save join token for workers
docker swarm join-token worker
```

**Orchestrator Systemd Service**
```ini
[Unit]
Description=Cumulus Orchestrator
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/cumulus
ExecStart=/opt/cumulus/cumulus-orchestrator --config /opt/cumulus/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Configuration Files

**Orchestrator Config**
```yaml
# /opt/cumulus/config.yaml
orchestrator:
  grpc_port: 50051
  swarm_manager: "unix:///var/run/docker.sock"  # Local Swarm manager
  
ssh:
  key_path: "/root/.ssh/id_rsa"
  user: "root"
  timeout: 10s

container:
  image: "your-registry/cumulus-job:latest"
  
max_job_duration: 86400  # 24 hours
default_duration: 3600   # 1 hour

logging:
  level: "info"
  format: "json"
```

### Worker Onboarding

**One-Script Setup (Vast.ai or any GPU instance)**
```bash
# On Swarm manager, get join token
SWARM_TOKEN=$(docker swarm join-token worker -q)
SWARM_MANAGER="<manager-ip>:2377"

# On GPU worker instance
curl -sSL https://your-domain.com/setup-worker.sh | bash -s -- \
  $SWARM_MANAGER \
  $SWARM_TOKEN

# Script installs:
# - Docker
# - Chronos
# - Joins Swarm
# - Labels node with GPU info
```

**Manual Worker Setup**
```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sh

# 2. Install Chronos
cd /tmp
git clone https://github.com/your-org/cumulus.git
cd cumulus/cumulus_core
./install-quick.sh

# 3. Join Swarm
docker swarm join --token <WORKER-TOKEN> <MANAGER-IP>:2377

# 4. On manager, label the node
NODE_ID=$(docker node ls --filter role=worker -q | tail -1)
docker node update --label-add gpu.count=2 $NODE_ID
docker node update --label-add gpu.0.name="NVIDIA A100" $NODE_ID
docker node update --label-add gpu.0.memory=40960 $NODE_ID
docker node update --label-add chronos.installed=true $NODE_ID
docker node update --label-add ssh.address="<worker-ip>" $NODE_ID
```


## Component Mapping

### Existing → New Architecture

**SDK (`sdk/`)**
- `sdk/client.py`: Update `CumulusClient` to use gRPC orchestrator
- `sdk/code_packager.py`: Keep as-is (still packages Python code to ZIP)
- `sdk/decorators.py`: Update to work with new client flow
- `sdk/distributed_checkpointer.py`: Keep as-is (checkpointing logic unchanged)
- `sdk/runtime.py`: Keep as-is (runtime helpers for jobs)

**Worker (`worker/` Python → Go)**
- `worker/server.py` (FastAPI) → `worker/cmd/main.go` (gRPC server)
- `worker/cumulus_manager.py` → `worker/internal/chronos/manager.go`
- `worker/executor.py` → `worker/internal/executor/executor.go`

**Chronos Integration (`cumulus_core/`)**
- Remains unchanged
- Go worker calls `chronos_cli` binary (same as Python worker does now)
- Alternative: Use CGO to call C++ library directly (future optimization)

**Tests (`tests/`)**
- Update existing tests to work with orchestrator
- Add new integration tests for orchestrator ↔ worker communication
- Keep checkpoint/artifact tests (functionality unchanged)

## Migration Path

### Phase 1: Build New Components
1. Create `orchestrator/` directory with Go implementation
2. Create new `worker/` Go implementation (can coexist with Python version)
3. Define Protocol Buffer schemas in `proto/`
4. Generate gRPC stubs for Go and Python

### Phase 2: Update SDK
1. Add gRPC dependencies to `sdk/`
2. Update `sdk/client.py` to support orchestrator mode
3. Keep existing direct-worker mode for backward compatibility
4. Add configuration option to choose mode

### Phase 3: Testing
1. Test orchestrator with new Go workers
2. Test updated SDK with orchestrator
3. Run existing test suite to ensure compatibility
4. Add new integration tests

### Phase 4: Deployment
1. Deploy orchestrator on central server
2. Deploy Go workers on GPU machines
3. Update client SDK in production
4. Monitor and validate

### Phase 5: Cleanup (Optional)
1. Remove old Python worker code from `worker/`
2. Remove backward compatibility from SDK
3. Update all documentation
4. Archive old implementation

## Future Enhancements

### Short Term (Next 3 months)
1. **Job Queuing**: Queue jobs when all workers busy
2. **Priority Scheduling**: Support job priorities
3. **Worker Auto-Discovery**: Dynamic worker registration
4. **Metrics Dashboard**: Web UI for monitoring

### Medium Term (3-6 months)
1. **Multi-GPU Jobs**: Support jobs spanning multiple GPUs
2. **Checkpoint Integration**: Resume jobs on different workers
3. **Resource Reservations**: Pre-book workers for future jobs
4. **Worker Pools**: Group workers by capabilities

### Long Term (6+ months)
1. **Auto-Scaling**: Automatically start/stop workers
2. **Spot Instance Support**: Use cloud spot instances
3. **Federation**: Multiple orchestrators for geo-distribution
4. **Advanced Scheduling**: Gang scheduling, bin packing

