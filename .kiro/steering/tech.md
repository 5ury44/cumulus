---
inclusion: always
---

# Technology Stack

## Languages & Frameworks

**Go 1.23+** - Central orchestrator
- gRPC for client communication
- Docker Engine API for Swarm integration
- SSH client for Chronos partition management
- YAML for configuration

**Python 3.8+** - SDK, CLI, and tests
- grpcio + grpcio-tools for gRPC client
- Existing code packaging and checkpointing
- Boto3 for S3 integration

**C++ 14** - Chronos core library
- OpenCL for GPU abstraction
- yaml-cpp for configuration (optional)
- Catch2 for testing

**Docker** - Container orchestration
- Docker Swarm for worker pool management
- Job containers with Python + OpenCL + ML frameworks

## Build Systems

**Go modules** - Go dependency management
- Orchestrator uses standard Go toolchain
- `go build` for binary compilation

**Protocol Buffers (protoc)** - gRPC code generation
- Generates Go code: `protoc --go_out=. --go-grpc_out=.`
- Generates Python code: `python -m grpc_tools.protoc`

**Docker** - Container image builds
- Job container image with Python + OpenCL + ML frameworks
- Multi-stage builds for optimization

**CMake 3.10+** - C++ build system for Chronos
- Builds shared library (libchronos.so/dylib/dll)
- Builds CLI tool (chronos_cli)
- Optional: tests, examples, benchmarks

**setuptools** - Python packaging
- Cumulus SDK: standard Python package
- Chronos: custom build_py command that invokes CMake

## Key Libraries

**Go:**
- `github.com/docker/docker/client` - Docker Engine API client
- `golang.org/x/crypto/ssh` - SSH client for Chronos management
- `google.golang.org/grpc` - gRPC server
- `google.golang.org/protobuf` - Protocol Buffers
- `gopkg.in/yaml.v3` - YAML configuration

**Python:**
- `grpcio` - gRPC client
- `grpcio-tools` - Protocol Buffers compiler

**System:**
- **OpenCL** - Cross-vendor GPU access
- **Docker Swarm** - Container orchestration platform
- **PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM** - ML framework support (in containers)

## Common Commands

### Orchestrator (Go)

```bash
# Build orchestrator
cd orchestrator
go build -o cumulus-orchestrator .

# Generate gRPC code
cd proto
protoc --go_out=. --go-grpc_out=. orchestrator.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. orchestrator.proto

# Run orchestrator
./cumulus-orchestrator --config config.yaml

# Development with Docker Context (remote Vast.ai instance)
./scripts/dev-setup.sh      # One-time setup
./scripts/dev-start.sh       # Build and run remotely
./scripts/dev-logs.sh        # View logs
./scripts/dev-stop.sh        # Cleanup
```

### Docker Swarm

```bash
# Initialize Swarm (on manager node)
docker swarm init --advertise-addr <PUBLIC_IP>

# Get worker join token
docker swarm join-token worker

# Add GPU worker (on worker node)
./scripts/setup-worker.sh <SWARM_MANAGER> <TOKEN>

# List nodes
docker node ls

# Label worker with GPU info
docker node update --label-add gpu.count=2 <NODE_ID>
docker node update --label-add gpu.0.name="NVIDIA A100" <NODE_ID>
```

### Chronos (C++)

```bash
# Build from source
cd cumulus_core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
sudo make install

# Quick install script
./cumulus_core/install-quick.sh

# CLI usage (on worker nodes)
chronos_cli stats
chronos_cli create <device> <memory> <duration>
chronos_cli list
chronos_cli release <partition_id>
```

### Cumulus SDK (Python)

```bash
# Install
pip install -e .

# Use SDK (connects to orchestrator via gRPC)
python
>>> from sdk.client import CumulusClient
>>> client = CumulusClient("orchestrator.example.com:50051")
>>> result = client.run(my_function, gpu_memory=0.5, duration=3600)

# Run tests
python tests/test_orchestrator.py
python tests/test_complete_nn.py
```

### Job Containers

```bash
# Build job container image
cd containers
docker build -t cumulus-job:latest .

# Test locally
docker run -v /path/to/code:/job cumulus-job:latest
```

## Platform Support

- **Linux** - Primary platform, full support
- **macOS** - Supported via Apple OpenCL framework
- **Windows** - Supported with platform-specific code paths

## Environment Variables

**Orchestrator:**
- `ORCHESTRATOR_CONFIG` - Path to config.yaml (default: ./config.yaml)
- `ORCHESTRATOR_PORT` - gRPC port (default: 50051)
- `SWARM_MANAGER` - Docker Swarm manager endpoint
- `SSH_KEY_PATH` - SSH private key for worker access

**SDK:**
- `CUMULUS_ORCHESTRATOR_URL` - Orchestrator endpoint (e.g., "orchestrator.example.com:50051")
- `CUMULUS_S3_BUCKET` - S3 bucket for checkpoints
- `CUMULUS_S3_REGION` - AWS region
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS credentials

**Workers (legacy):**
- `CUMULUS_CHRONOS_PATH` - Path to chronos_cli binary
- `MAX_CONCURRENT_JOBS` - Server job limit (default: 5)
