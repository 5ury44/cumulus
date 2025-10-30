#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔧 Generating gRPC code from proto definitions..."

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "❌ protoc not found. Please install Protocol Buffers compiler"
    exit 1
fi

# Check protoc version
PROTOC_VERSION=$(protoc --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "✅ Found protoc version $PROTOC_VERSION"

# Generate Go code
echo ""
echo "📦 Generating Go code..."

# Check if Go plugins are installed
if ! command -v protoc-gen-go &> /dev/null; then
    echo "⚠️  protoc-gen-go not found. Installing..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "⚠️  protoc-gen-go-grpc not found. Installing..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

# Create output directory for Go
mkdir -p "$PROJECT_ROOT/orchestrator/proto"

# Generate Go code
protoc \
    --proto_path="$SCRIPT_DIR" \
    --go_out="$PROJECT_ROOT/orchestrator/proto" \
    --go_opt=paths=source_relative \
    --go-grpc_out="$PROJECT_ROOT/orchestrator/proto" \
    --go-grpc_opt=paths=source_relative \
    "$SCRIPT_DIR/orchestrator.proto"

echo "✅ Go code generated in orchestrator/proto/"

# Generate Python code
echo ""
echo "🐍 Generating Python code..."

# Check if Python gRPC tools are installed
if ! python3 -c "import grpc_tools" &> /dev/null; then
    echo "⚠️  grpcio-tools not found. Installing..."
    pip install grpcio grpcio-tools
fi

# Create output directory for Python
mkdir -p "$PROJECT_ROOT/sdk/proto"

# Generate Python code
python3 -m grpc_tools.protoc \
    --proto_path="$SCRIPT_DIR" \
    --python_out="$PROJECT_ROOT/sdk/proto" \
    --grpc_python_out="$PROJECT_ROOT/sdk/proto" \
    "$SCRIPT_DIR/orchestrator.proto"

# Create __init__.py for Python package
cat > "$PROJECT_ROOT/sdk/proto/__init__.py" << 'EOF'
"""
Generated gRPC code for Cumulus Orchestrator
"""
from . import orchestrator_pb2
from . import orchestrator_pb2_grpc

__all__ = ['orchestrator_pb2', 'orchestrator_pb2_grpc']
EOF

echo "✅ Python code generated in sdk/proto/"

echo ""
echo "✅ Code generation complete!"
echo ""
echo "Generated files:"
echo "  Go:     orchestrator/proto/orchestrator.pb.go"
echo "  Go:     orchestrator/proto/orchestrator_grpc.pb.go"
echo "  Python: sdk/proto/orchestrator_pb2.py"
echo "  Python: sdk/proto/orchestrator_pb2_grpc.py"
