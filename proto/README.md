# Cumulus Orchestrator gRPC API

This directory contains the Protocol Buffers definitions for the Cumulus Orchestrator gRPC API.

## Overview

The orchestrator exposes a gRPC service that allows clients to:
- Submit jobs for execution on GPU workers
- Monitor job progress via streaming updates
- Cancel running jobs
- Query worker pool status
- Check orchestrator health

## Service Definition

See `orchestrator.proto` for the complete service definition.

### Key RPCs

- **SubmitJob**: Submit a job and receive streaming progress updates
- **GetJobStatus**: Query the current status of a job
- **CancelJob**: Cancel a running or pending job
- **ListWorkers**: List available GPU workers (admin)
- **HealthCheck**: Check orchestrator health

## Code Generation

### Prerequisites

Install Protocol Buffers compiler and plugins:

```bash
# Install all tools
make install-proto-tools

# Or manually:
# macOS
brew install protobuf

# Linux
sudo apt-get install -y protobuf-compiler

# Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Python tools
pip install grpcio grpcio-tools
```

### Generate Code

```bash
# Generate both Go and Python code
make proto

# Or generate individually
make proto-go      # Generate Go code only
make proto-python  # Generate Python code only

# Clean generated files
make clean-proto
```

### Generated Files

**Go** (in `orchestrator/proto/`):
- `orchestrator.pb.go` - Message definitions
- `orchestrator_grpc.pb.go` - Service stubs

**Python** (in `sdk/proto/`):
- `orchestrator_pb2.py` - Message definitions
- `orchestrator_pb2_grpc.py` - Service stubs
- `__init__.py` - Package initialization

## Usage Examples

### Go (Server)

```go
import (
    pb "github.com/cumulus/orchestrator/proto"
    "google.golang.org/grpc"
)

type orchestratorServer struct {
    pb.UnimplementedOrchestratorServiceServer
}

func (s *orchestratorServer) SubmitJob(req *pb.JobSubmission, stream pb.OrchestratorService_SubmitJobServer) error {
    // Handle job submission
    // Stream progress updates
    return nil
}

func main() {
    lis, _ := net.Listen("tcp", ":50051")
    grpcServer := grpc.NewServer()
    pb.RegisterOrchestratorServiceServer(grpcServer, &orchestratorServer{})
    grpcServer.Serve(lis)
}
```

### Python (Client)

```python
import grpc
from sdk.proto import orchestrator_pb2, orchestrator_pb2_grpc

# Connect to orchestrator
channel = grpc.insecure_channel('localhost:50051')
stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)

# Submit job
submission = orchestrator_pb2.JobSubmission(
    code_package=code_zip,
    requirements=['torch', 'numpy'],
    gpu_memory=0.5,
    duration=3600
)

# Stream results
for event in stub.SubmitJob(submission):
    print(f"Job {event.job_id}: {event.state}")
    if event.state == orchestrator_pb2.COMPLETED:
        print(f"Result: {event.result}")
```

## Message Types

### JobSubmission
Complete job package including code, requirements, and resource specs.

### JobEvent
Streaming updates about job progress (SCHEDULING → RUNNING → COMPLETED/FAILED).

### JobStatus
Detailed information about a job's current state.

### WorkerInfo
Information about GPU workers in the pool.

### GPUDevice
Details about a specific GPU including capacity and active partitions.

## GPU Resource Allocation

GPU memory can be specified as:
- **Fractional**: 0.1-0.9 (time-sliced sharing via Chronos)
- **Whole GPUs**: 1, 2, 3, 4, 5, 6, 7, 8

Invalid values (e.g., 0, 2.3, 1.5) will be rejected.

## Job Lifecycle

1. **SCHEDULING** - Finding available worker
2. **CREATING_PARTITION** - Creating Chronos GPU partition
3. **RUNNING** - Container executing user code
4. **COMPLETED** - Job finished successfully
5. **FAILED** - Job encountered an error
6. **CANCELLED** - Job was cancelled by user

## Error Handling

The orchestrator returns standard gRPC status codes:
- `OK` - Success
- `INVALID_ARGUMENT` - Invalid job parameters
- `UNAVAILABLE` - No workers available
- `NOT_FOUND` - Job not found
- `CANCELLED` - Job was cancelled
- `INTERNAL` - Internal error

## Version

Protocol version: v1 (initial release)
