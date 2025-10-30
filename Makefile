.PHONY: proto proto-go proto-python clean-proto install-proto-tools help

# Default target
help:
	@echo "Cumulus Build Targets:"
	@echo "  make proto              - Generate all gRPC code (Go + Python)"
	@echo "  make proto-go           - Generate Go gRPC code only"
	@echo "  make proto-python       - Generate Python gRPC code only"
	@echo "  make install-proto-tools - Install protoc and plugins"
	@echo "  make clean-proto        - Remove generated proto files"

# Install Protocol Buffers tools
install-proto-tools:
	@echo "📦 Installing Protocol Buffers tools..."
	@if ! command -v protoc &> /dev/null; then \
		echo "Installing protoc..."; \
		if [[ "$$OSTYPE" == "darwin"* ]]; then \
			brew install protobuf; \
		elif [[ "$$OSTYPE" == "linux-gnu"* ]]; then \
			sudo apt-get update && sudo apt-get install -y protobuf-compiler; \
		else \
			echo "❌ Unsupported OS. Please install protoc manually."; \
			exit 1; \
		fi; \
	fi
	@echo "Installing Go plugins..."
	@go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	@go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "Installing Python tools..."
	@pip install grpcio grpcio-tools
	@echo "✅ All tools installed"

# Generate all proto code
proto: proto-go proto-python
	@echo "✅ All gRPC code generated"

# Generate Go code
proto-go:
	@echo "📦 Generating Go gRPC code..."
	@mkdir -p orchestrator/proto
	@protoc \
		--proto_path=proto \
		--go_out=orchestrator/proto \
		--go_opt=paths=source_relative \
		--go-grpc_out=orchestrator/proto \
		--go-grpc_opt=paths=source_relative \
		proto/orchestrator.proto
	@echo "✅ Go code generated in orchestrator/proto/"

# Generate Python code
proto-python:
	@echo "🐍 Generating Python gRPC code..."
	@mkdir -p sdk/proto
	@python3 -m grpc_tools.protoc \
		--proto_path=proto \
		--python_out=sdk/proto \
		--grpc_python_out=sdk/proto \
		proto/orchestrator.proto
	@echo "# Generated gRPC code for Cumulus Orchestrator" > sdk/proto/__init__.py
	@echo "from . import orchestrator_pb2" >> sdk/proto/__init__.py
	@echo "from . import orchestrator_pb2_grpc" >> sdk/proto/__init__.py
	@echo "" >> sdk/proto/__init__.py
	@echo "__all__ = ['orchestrator_pb2', 'orchestrator_pb2_grpc']" >> sdk/proto/__init__.py
	@echo "✅ Python code generated in sdk/proto/"

# Clean generated files
clean-proto:
	@echo "🗑️  Removing generated proto files..."
	@rm -rf orchestrator/proto/*.pb.go
	@rm -rf sdk/proto/*_pb2*.py
	@rm -f sdk/proto/__init__.py
	@echo "✅ Cleaned"
