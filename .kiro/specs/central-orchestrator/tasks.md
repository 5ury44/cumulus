# Implementation Plan - MVP Focus

## Overview

This implementation plan focuses on building a **Minimum Viable Product (MVP)** that demonstrates the core orchestrator functionality. The goal is to get a working system quickly, then iterate and improve.

## MVP Goals (SIMPLIFIED)

1. **Client sends everything to orchestrator**: Code, requirements, GPU needs via gRPC
2. **Orchestrator is the relay**: All communication goes through it
3. **Orchestrator controls containers**: Creates, monitors, kills containers via Docker Swarm
4. **One-script worker setup**: Vast.ai instance joins Swarm with one command
5. **Docker Swarm handles everything**: Health checks, scheduling, container management

## Architecture Decision: Go + Docker Swarm + gRPC

**Technology Stack:**
- **Orchestrator**: Go 1.23+ with gRPC server
- **Platform**: Docker Swarm for worker management
- **Client SDK**: Python with gRPC (grpcio)
- **Containers**: Docker with GPU access via Chronos

**Simplified Flow:**
```
Client (gRPC) → Orchestrator (Go) → Docker Swarm → Worker Node → Container → Results
```

**Benefits:**
- ✅ gRPC for type-safe, efficient communication
- ✅ Docker Swarm handles worker management automatically
- ✅ Go for performance and concurrency
- ✅ Python SDK stays familiar for users
- ✅ Orchestrator controls everything (can kill jobs anytime)
- ✅ Client only needs one endpoint
- ✅ Simpler security model
- ✅ Easier to debug (all traffic through orchestrator)

## Docker Swarm Benefits

**What Swarm gives us for free:**
- ✅ Worker pool management
- ✅ Container orchestration
- ✅ Health checks and heartbeats
- ✅ Service discovery
- ✅ Load balancing
- ✅ Rolling updates
- ✅ Secrets management

## Development Setup

**For development, we use Docker Context + Vast.ai to run the orchestrator remotely while developing locally.**

This approach:
- ✅ Solves networking issues (no port forwarding needed)
- ✅ Provides seamless dev experience (local commands → remote execution)
- ✅ Cheap (~$0.05-0.10/hour for CPU instance)
- ✅ Kill local terminal → kills remote container

See the Design document for detailed setup instructions and scripts.

## Task List

- [x] 0. Development environment setup
- [x] 0.1 Create development setup scripts
  - Write `scripts/dev-setup.sh` for one-time Vast.ai + Docker context setup
  - Write `scripts/dev-start.sh` to build and run orchestrator remotely
  - Write `scripts/dev-stop.sh` to cleanup and destroy instance
  - Write `scripts/dev-logs.sh` to view remote logs
  - Test full workflow: setup → start → logs → stop
  - _Requirements: Development workflow, Docker Context integration_

- [x] 1. Set up Protocol Buffers and gRPC definitions
- [x] 1.1 Create proto/ directory and define orchestrator service
  - Define OrchestratorService with SubmitJob, GetJobStatus, CancelJob, ListWorkers, HealthCheck
  - Define JobSubmission message (code_package, requirements, gpu_memory, duration)
  - Define JobEvent message for streaming (job_id, state, message, result, progress)
  - Define JobState enum (SCHEDULING, CREATING_PARTITION, RUNNING, COMPLETED, FAILED, CANCELLED)
  - _Requirements: 1.1, 3.1, 5.1, 6.1_

- [x] 1.2 Generate Go and Python gRPC code
  - Install protoc and Go/Python plugins
  - Generate Go code: protoc --go_out=. --go-grpc_out=. orchestrator.proto
  - Generate Python code: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. orchestrator.proto
  - Verify generated code compiles
  - _Requirements: 13.1, 13.4, 13.5_

- [-] 2. Build Docker Swarm client wrapper (Go)
- [x] 2.1 Create SwarmClient struct with Docker API client
  - Initialize Docker client: client.NewClientWithOpts()
  - Implement ListWorkers() to query Swarm nodes with GPU labels
  - Implement GetNodeInfo() to get node details
  - Test connection to local Swarm
  - _Requirements: 2.1, 2.2, 8.1, 8.3_

- [x] 2.2 Implement service creation for job containers
  - Implement CreateService() to create Docker service with constraints
  - Build swarm.ServiceSpec with container image, command, environment
  - Add placement constraints to target specific node
  - Add mount for job code package
  - Return service ID
  - _Requirements: 4.2, 4.3, 8.4_

- [x] 2.3 Implement service monitoring and log streaming
  - Implement GetServiceLogs() to stream container logs
  - Implement GetServiceStatus() to check service state
  - Implement RemoveService() to clean up completed services
  - Test log streaming from running container
  - _Requirements: 5.1, 5.2, 5.3, 8.5_


- [ ] 3. Build Chronos partition manager (Go)
- [x] 3.1 Create ChronosManager with SSH client pool
  - Implement SSH connection management to worker nodes
  - Store worker address → SSH client mapping
  - Handle SSH authentication (key-based)
  - Test SSH connection to worker
  - _Requirements: 4.1, 7.1, 14.3_

- [x] 3.2 Implement CreatePartition via SSH
  - SSH to worker node
  - Execute: chronos_cli create <device> <memory> <duration>
  - Parse partition ID from command output
  - Store partition info (worker, partition ID, expiry)
  - Return partition ID
  - _Requirements: 4.1, 7.2_

- [x] 3.3 Implement ReleasePartition via SSH
  - SSH to worker node
  - Execute: chronos_cli release <partition_id>
  - Remove partition from tracking
  - Handle errors gracefully
  - _Requirements: 7.4, 9.1, 9.2_

- [ ] 4. Build orchestrator gRPC service (Go)
- [x] 4.1 Create OrchestratorServer struct
  - Embed UnimplementedOrchestratorServiceServer
  - Add SwarmClient and ChronosManager fields
  - Add job tracking map (job ID → JobInfo)
  - Initialize with config
  - _Requirements: 1.1, 11.1_

- [x] 4.2 Implement SubmitJob RPC handler
  - Receive JobSubmission from client
  - Validate request fields
  - Select available worker from Swarm
  - Create Chronos partition on worker
  - Build service spec with partition context
  - Submit service to Swarm
  - Stream JobEvents back to client
  - _Requirements: 3.1, 3.2, 3.4, 4.1, 4.2, 6.1, 6.2_

- [x] 4.3 Implement job monitoring and streaming
  - Poll Swarm for service status
  - Stream container logs to client
  - Send progress updates (SCHEDULING, RUNNING, etc.)
  - Handle job completion
  - Release Chronos partition
  - Remove service from Swarm
  - Send final result to client
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 9.1, 9.2_

- [ ] 4.4 Implement GetJobStatus RPC handler
  - Look up job in tracking map
  - Query Swarm for service status
  - Return current job state
  - _Requirements: 10.3_

- [ ] 4.5 Implement CancelJob RPC handler
  - Look up job in tracking map
  - Stop service via Swarm API
  - Release Chronos partition
  - Update job state to CANCELLED
  - _Requirements: 9.3, 12.5_

- [ ] 4.6 Implement ListWorkers and HealthCheck RPCs
  - ListWorkers: Query Swarm nodes, return worker info
  - HealthCheck: Return orchestrator status
  - _Requirements: 10.1, 10.4_

- [ ] 4.7 Create main.go for orchestrator
  - Load configuration from YAML
  - Initialize SwarmClient
  - Initialize ChronosManager
  - Create gRPC server
  - Register OrchestratorService
  - Start listening on port
  - _Requirements: 11.1, 11.2, 11.4, 13.1, 13.5_

- [ ] 5. Build job container image
- [x] 5.1 Create Dockerfile for job execution
  - Base image: python:3.10-slim
  - Install OpenCL libraries (ocl-icd-opencl-dev)
  - Install common ML libraries (numpy, torch, tensorflow) - optional
  - Copy entrypoint script
  - Set working directory
  - _Requirements: 7.1, 13.3, 15.1_

- [x] 5.2 Create container entrypoint script
  - Accept code package as mounted volume
  - Extract ZIP file to working directory
  - Install requirements: pip install -r requirements.txt
  - Execute user code: python main.py
  - Write results to stdout in JSON format
  - Handle errors and write to stdout
  - _Requirements: 15.2, 15.3, 15.5_

- [x] 5.3 Build and test container image locally
  - Build image: docker build -t cumulus-job:latest
  - Test with sample job (simple Python script)
  - Verify GPU access works (if GPU available)
  - Push to container registry
  - _Requirements: 7.3, 15.4_

- [ ] 6. Update Python SDK for gRPC
- [ ] 6.1 Add gRPC dependencies to SDK
  - Add grpcio and grpcio-tools to requirements
  - Copy generated Python proto files to sdk/proto/
  - Update __init__.py to import proto modules
  - _Requirements: 6.1, 13.4, 13.5_

- [ ] 6.2 Update CumulusClient for orchestrator relay
  - Update __init__ to create gRPC channel to orchestrator
  - Create OrchestratorServiceStub
  - Remove old HTTP client code
  - _Requirements: 6.1, 6.2_

- [ ] 6.3 Update CumulusClient.run() for gRPC streaming
  - Package code with existing CodePackager
  - Create JobSubmission message
  - Call stub.SubmitJob() and iterate over stream
  - Handle JobEvent states (SCHEDULING, RUNNING, COMPLETED, FAILED)
  - Return final result or raise error
  - _Requirements: 6.2, 6.3, 6.4, 6.5_

- [ ] 6.4 Keep code_packager.py and other modules unchanged
  - No changes needed to ZIP packaging
  - decorators.py, distributed_checkpointer.py stay the same
  - _Requirements: All existing functionality_

- [ ] 7. Create worker setup script
- [ ] 7.1 Write setup-worker.sh script
  - Install Docker
  - Install Chronos (clone repo, run install-quick.sh)
  - Detect GPU info (nvidia-smi)
  - Join Docker Swarm with provided token
  - Print commands for manager to label node
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 7.2 Test worker setup on fresh instance
  - Spin up Vast.ai GPU instance
  - Run setup script
  - Verify worker joins Swarm
  - Verify Chronos installed
  - Label node on manager
  - _Requirements: 14.5_

- [ ] 8. Create end-to-end integration test
- [ ] 8.1 Write minimal test that verifies basic flow
  - Initialize Docker Swarm locally
  - Start orchestrator
  - Use SDK to submit job that returns 2+2
  - Verify result is 4
  - _Requirements: All core requirements_

- [ ]* 8.2 Test GPU resource allocation
  - Submit job requesting 0.5 GPU
  - Submit job requesting 0.3 GPU (should succeed)
  - Verify both jobs run
  - Verify GPU tracking is accurate
  - _Requirements: 4.1, 4.2, 7.1_

- [ ]* 8.3 Test error scenarios
  - Submit job with invalid GPU amount (2.3)
  - Submit job when no workers available
  - Verify appropriate error messages
  - _Requirements: 12.1, 12.2, 12.3_

- [ ] 9. Create minimal deployment setup
- [ ] 9.1 Write simple build script
  - Build orchestrator binary
  - Build container image
  - Generate proto files
  - _Requirements: 13.1_

- [ ] 9.2 Create basic config.yaml.example
  - Minimal working configuration
  - Essential fields only
  - _Requirements: 11.1, 11.2_

- [ ]* 9.3 Write comprehensive deployment documentation
  - How to initialize Docker Swarm
  - How to deploy orchestrator
  - How to onboard workers
  - How to use SDK
  - Troubleshooting guide
  - _Requirements: 14.1, 14.2, 14.4_


## Optional Tasks (Post-MVP)

These tasks can be added after the MVP is working:

- [ ]* 10. Add structured logging
- [ ]* 10.1 Add zerolog or zap to orchestrator
- [ ]* 10.2 Make log level configurable
- [ ]* 10.3 Add request tracing

- [ ]* 11. Add metrics and monitoring
- [ ]* 11.1 Add Prometheus metrics to orchestrator
- [ ]* 11.2 Create Grafana dashboard
- [ ]* 11.3 Add alerting rules

- [ ]* 12. Add security features
- [ ]* 12.1 Add TLS/mTLS support
- [ ]* 12.2 Add authentication tokens
- [ ]* 12.3 Add rate limiting

- [ ]* 13. Add advanced features
- [ ]* 13.1 Job queuing when workers busy
- [ ]* 13.2 Priority scheduling
- [ ]* 13.3 Multi-region support
- [ ]* 13.4 Job persistence (database)
- [ ]* 13.5 Web UI for monitoring

- [ ]* 14. Improve testing
- [ ]* 14.1 Add unit tests for all modules
- [ ]* 14.2 Add load testing
- [ ]* 14.3 Add chaos testing
- [ ]* 14.4 Update existing SDK test suite

- [ ]* 15. Production readiness
- [ ]* 15.1 Create systemd service files
- [ ]* 15.2 Add configuration validation
- [ ]* 15.3 Write operations runbook
- [ ]* 15.4 Add backup/restore procedures

## Implementation Order (MVP)

### Phase 1: Core Infrastructure (3-5 days)
- **Tasks 1-2**: Protocol Buffers, Docker Swarm client
- **Deliverable**: gRPC definitions, Swarm integration working
- **Success**: Can list Swarm nodes, create services

### Phase 2: Orchestrator Service (5-7 days)
- **Tasks 3-4**: Chronos manager, gRPC service implementation
- **Deliverable**: Working orchestrator that accepts jobs
- **Success**: Can submit job via gRPC, create partition, schedule container

### Phase 3: Container & SDK (3-5 days)
- **Tasks 5-6**: Job container image, Python SDK updates
- **Deliverable**: Container executes jobs, SDK submits via gRPC
- **Success**: End-to-end job execution works

### Phase 4: Worker Setup & Testing (3-5 days)
- **Tasks 7-8**: Worker setup script, integration tests
- **Deliverable**: One-script worker onboarding, test suite
- **Success**: Vast.ai instance joins pool in < 5 minutes

### Phase 5: Minimal Deployment (1-2 days)
- **Task 9**: Build script, basic config
- **Deliverable**: Can build and run orchestrator
- **Success**: Can deploy orchestrator and onboard workers

**Total Estimated Time: 13-19 days (MVP)**

## Vast.ai Worker Setup Script (Goal)

The ultimate goal is a single script that runs on a fresh Vast.ai GPU instance:

```bash
# On Vast.ai instance:
curl -sSL https://your-domain.com/setup-worker.sh | bash -s -- \
  --orchestrator-url "nomad.example.com:4646" \
  --worker-token "secret-token"

# Script does:
# 1. Install dependencies (Docker/Nomad client, Python, Chronos)
# 2. Configure GPU access
# 3. Join orchestrator pool
# 4. Start accepting jobs
# 
# Worker is now in pool and accepting jobs!
```

## Success Criteria (MVP)

- [ ] Docker Swarm manages worker pool automatically
- [ ] Worker setup script works on fresh Vast.ai instance
- [ ] Worker joins Swarm and is ready for jobs in < 5 minutes
- [ ] Swarm handles heartbeats and health checks automatically
- [ ] SDK can submit jobs via gRPC to orchestrator
- [ ] Orchestrator creates Chronos partitions via SSH
- [ ] Orchestrator schedules containers on Swarm workers
- [ ] Jobs execute with GPU access from Chronos partitions
- [ ] Container logs stream back through orchestrator to client
- [ ] Results return to SDK via gRPC stream
- [ ] GPU resource tracking works (fractional and whole GPUs)
- [ ] Partitions and services cleaned up after job completion
- [ ] Dead workers automatically removed from Swarm pool
- [ ] Can handle multiple concurrent jobs across multiple workers

## Key Simplifications for MVP

1. **No authentication**: Plain gRPC without TLS (add mTLS later)
2. **No job queuing**: Return error if no GPU capacity available
3. **No metrics**: Use structured logging for debugging
4. **Simple GPU selection**: First-fit algorithm for GPU allocation
5. **SSH-based partition management**: Direct SSH to workers (could use agent later)
6. **No job persistence**: In-memory job tracking only
7. **No multi-region**: Single Swarm cluster
8. **Basic error handling**: Retry SSH/Swarm calls, fail job on persistent errors

## Notes

- Focus on getting something working end-to-end
- Use existing libraries where possible (research task 1)
- Keep code simple and readable
- Add complexity only when needed
- Test frequently with real GPU instances
