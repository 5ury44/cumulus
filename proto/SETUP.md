# Protocol Buffers Setup Guide

## Quick Start

```bash
# Install all required tools
make install-proto-tools

# Generate gRPC code
make proto
```

## Manual Installation

### macOS

```bash
# Install protoc
brew install protobuf

# Verify installation
protoc --version

# Install Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Install Python tools
pip install grpcio grpcio-tools
```

### Linux (Ubuntu/Debian)

```bash
# Install protoc
sudo apt-get update
sudo apt-get install -y protobuf-compiler

# Verify installation
protoc --version

# Install Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Install Python tools
pip install grpcio grpcio-tools
```

### Verify Installation

```bash
# Check protoc
protoc --version
# Should show: libprotoc 3.x.x or higher

# Check Go plugins
which protoc-gen-go
which protoc-gen-go-grpc

# Check Python tools
python3 -c "import grpc_tools; print('OK')"
```

## Generating Code

Once tools are installed:

```bash
# Generate all code (Go + Python)
make proto

# Or use the script directly
./proto/generate.sh

# Or generate manually
make proto-go      # Go only
make proto-python  # Python only
```

## Troubleshooting

### protoc not found

Install Protocol Buffers compiler using your package manager (see above).

### protoc-gen-go not found

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Make sure $GOPATH/bin is in your PATH
export PATH="$PATH:$(go env GOPATH)/bin"
```

### grpcio-tools not found

```bash
pip install grpcio grpcio-tools
```

### Permission denied on generate.sh

```bash
chmod +x proto/generate.sh
```

## Next Steps

After generating code:

1. **Go**: Import generated code in orchestrator
   ```go
   import pb "github.com/cumulus/orchestrator/proto"
   ```

2. **Python**: Import in SDK
   ```python
   from sdk.proto import orchestrator_pb2, orchestrator_pb2_grpc
   ```

3. Implement the gRPC service (Go) and client (Python)
