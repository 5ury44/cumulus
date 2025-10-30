# Cumulus Job Container

This directory contains the Docker container image used to execute user jobs in the Cumulus distributed execution system.

## Overview

The job container provides a standardized execution environment for user code with:
- Python 3.10 runtime
- OpenCL support for GPU access
- Common ML libraries pre-installed
- Automatic code extraction and dependency installation
- Structured result output in JSON format

## Container Structure

```
containers/
├── Dockerfile          # Container image definition
├── entrypoint.sh       # Container startup script
├── runner.py           # Python job execution wrapper
├── build.sh           # Build script
├── test-container.sh  # Test script
├── test-job/          # Test job example
│   ├── main.py        # Test job code
│   └── requirements.txt
└── README.md          # This file
```

## Building the Container

```bash
cd containers
./build.sh
```

This builds the `cumulus-job:latest` image.

## Job Code Requirements

User jobs must provide:
- `main.py` with a `main()` function that returns the result
- Optional `requirements.txt` for Python dependencies

### Example Job

```python
# main.py
def main():
    import numpy as np
    
    # Your computation here
    result = np.array([1, 2, 3, 4, 5]).sum()
    
    return {
        "message": "Computation completed",
        "result": result
    }
```

## Container Execution

The container expects:
1. Job code mounted at `/job/code.zip`
2. Environment variables:
   - `JOB_ID`: Unique job identifier
   - `CHRONOS_PARTITION_ID`: GPU partition ID
   - `GPU_DEVICE`: GPU device index

### Manual Execution

```bash
# Create job directory
mkdir -p /tmp/test-job
cd /tmp/test-job

# Create job code
cat > main.py << 'EOF'
def main():
    return {"message": "Hello from Cumulus!", "result": 42}
EOF

# Package job
zip code.zip main.py

# Run container
docker run --rm \
    -v /tmp/test-job:/job \
    -e JOB_ID="test-123" \
    -e CHRONOS_PARTITION_ID="partition-456" \
    -e GPU_DEVICE="0" \
    cumulus-job:latest
```

## Output Format

The container outputs results to stdout as JSON:

### Success Response
```json
{
  "status": "success",
  "result": { "your": "result" },
  "execution_time": 1.23,
  "timestamp": 1640995200,
  "system_info": {
    "python_version": "3.10.0",
    "job_id": "test-123",
    "chronos_partition": "partition-456",
    "gpu_device": "0"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Error message",
  "error_type": "RuntimeError",
  "traceback": "Full traceback...",
  "execution_time": 0.5,
  "timestamp": 1640995200,
  "system_info": { ... }
}
```

## Testing

Run the test suite:

```bash
./test-container.sh
```

This builds the container and runs a test job to verify functionality.

## GPU Support

The container includes OpenCL support for GPU access. GPU resources are managed by the Chronos partition system:

1. Orchestrator creates a Chronos partition on the worker
2. Container inherits GPU access from the partition
3. User code can access GPU via OpenCL or ML frameworks
4. Partition is released when job completes

## Pre-installed Libraries

The container includes common ML libraries:
- numpy, scipy, pandas
- scikit-learn
- matplotlib, seaborn
- jupyter, requests

Additional dependencies can be specified in `requirements.txt`.

## Security

The container runs user code in an isolated environment:
- No network access by default
- Limited filesystem access
- GPU access controlled by Chronos partitions
- Automatic cleanup after execution

## Integration with Orchestrator

The orchestrator uses this container by:
1. Writing job code to worker filesystem
2. Creating Docker service with this image
3. Mounting job directory into container
4. Monitoring container execution
5. Collecting results from stdout
6. Cleaning up resources