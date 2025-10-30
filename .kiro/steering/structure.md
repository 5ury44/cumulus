---
inclusion: always
---

# Project Structure

## Top-Level Organization

```
cumulus/
├── orchestrator/          # NEW: Go-based central orchestrator
│   ├── main.go           # Orchestrator entry point
│   ├── platform/         # Docker Swarm client
│   ├── chronos/          # Chronos partition manager
│   ├── jobs/             # Job lifecycle handler
│   └── proto/            # Generated gRPC code
├── containers/            # NEW: Job container image
│   ├── Dockerfile        # Base image with Python + OpenCL
│   ├── entrypoint.sh     # Container entry point
│   └── runner.py         # Job execution script
├── proto/                 # NEW: gRPC/Protocol Buffers definitions
│   └── orchestrator.proto
├── cumulus_core/          # C++ Chronos GPU partitioning library
├── cumulus_vendor/        # Vendored Chronos builds (optional)
├── sdk/                   # Python client SDK (updated for gRPC)
├── worker/                # DEPRECATED: Old Python worker (being replaced)
├── tests/                 # Integration tests
├── scripts/               # Deployment and setup scripts
│   ├── dev-setup.sh      # Development environment setup
│   ├── dev-start.sh      # Start orchestrator remotely
│   ├── dev-stop.sh       # Cleanup dev environment
│   ├── dev-logs.sh       # View orchestrator logs
│   └── setup-worker.sh   # One-script worker onboarding
├── cli.py                 # Main CLI entry point
├── setup.py               # Python package setup
└── start_worker.py        # DEPRECATED: Old worker startup
```

## Cumulus SDK (Python)

**sdk/** - Client-side Python SDK for remote execution
- `client.py` - CumulusClient and CumulusJob classes
- `code_packager.py` - Code packaging and serialization
- `decorators.py` - @remote, @gpu, @async_remote decorators
- `config.py` - Configuration management
- `distributed_checkpointer.py` - Framework-agnostic checkpointing
- `framework_adapters.py` - ML framework adapters
- `runtime.py` - Runtime utilities (save_checkpoint, load_checkpoint)
- `env_loader.py` - Environment setup

**orchestrator/** - Central orchestrator (Go)
- `main.go` - Orchestrator entry point and gRPC server
- `platform/swarm.go` - Docker Swarm client wrapper
- `chronos/manager.go` - Chronos partition management via SSH
- `jobs/handler.go` - Job lifecycle management
- `proto/` - Generated gRPC code from .proto files
- `config.yaml` - Orchestrator configuration

**containers/** - Job execution containers
- `Dockerfile` - Base image with Python, OpenCL, ML libraries
- `entrypoint.sh` - Container startup script
- `runner.py` - Job execution wrapper

**proto/** - gRPC service definitions
- `orchestrator.proto` - OrchestratorService definition

**worker/** - DEPRECATED: Old Python worker (being replaced by containers)

## Chronos Core (C++)

**cumulus_core/** - C++ GPU partitioning library

**include/** - Public headers
- `chronos.h` - Main C++ API
- `chronos_c.h` - C API for bindings
- `chronos_utils.h` - Utility functions
- `cli/formatter.h` - CLI output formatting

**src/** - Implementation
- `partitioner.cpp` - Main partitioner logic
- `chronos_c.cpp` - C API implementation
- `chronos_utils.cpp` - Utilities
- `core/` - Core components (device_info, gpu_partition, memory_enforcer)
- `platform/` - Platform-specific code (unix_platform, windows_platform)
- `utils/` - Utilities (lock_file, time_utils)
- `config/` - Configuration management (optional)

**apps/cli/** - Command-line interface
- `main.cpp` - CLI entry point
- `commands.cpp` - Command implementations

**tests/** - C++ unit tests (Catch2)
**examples/** - C++ usage examples
**benchmarks/** - Performance benchmarks
**python/** - Python bindings

## Tests

**tests/** - Integration and end-to-end tests
- `test_complete_nn.py` - Full PyTorch training workflow
- `test_unified_s3_complete.py` - Multi-framework S3 checkpointing
- `test_artifact_store.py` - Artifact storage tests
- `submission.py` - Test submission utilities

## Key Patterns

### Code Organization
- C++ follows header/implementation split
- Python uses module-based organization
- Clear separation between client (sdk) and server (worker)

### API Layers
1. **C++ Core** - Low-level GPU partitioning (chronos.h)
2. **C API** - Language bindings layer (chronos_c.h)
3. **gRPC API** - Orchestrator service (orchestrator.proto)
4. **Python SDK** - High-level client API (client.py)
5. **HTTP API** - DEPRECATED: Old REST endpoints (server.py)

### Configuration
- C++: YAML files (optional, via yaml-cpp)
- Python: Environment variables + .env files
- S3: `/opt/cumulus-distributed/.env`

### Build Artifacts
- C++: `build/lib/libchronos.{so,dylib,dll}`, `build/bin/chronos_cli`
- Python: Installed via pip, entry points: `cumulus-cli`, `cumulus-worker`
