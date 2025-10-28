# Cumulus Tests

This directory contains comprehensive tests for the Cumulus distributed execution and checkpointing system.

## Test Files

### `test_unified_checkpointing.py`

**Local unified checkpointing tests** - Tests the framework-agnostic checkpointing API locally for all supported ML frameworks:

- PyTorch
- TensorFlow/Keras
- scikit-learn
- XGBoost
- LightGBM

Includes both local-only tests and S3 integration tests (when S3 is configured).

### `test_unified_s3_complete.py`

**Remote unified checkpointing with S3** - Comprehensive test that runs on the remote GPU server via CumulusClient, testing:

- All frameworks with unified checkpointing API
- Full S3 integration (L1 local + L2 S3 caching)
- Cross-machine checkpoint resumption
- Framework-agnostic save/load operations

### `test_remote_unified.py`

**Basic remote unified checkpointing** - Simple test of unified checkpointing running remotely without S3 integration.

### `test_complete_nn.py`

**Complete neural network training** - End-to-end PyTorch training with checkpointing and resume functionality.

### `test_artifact_store.py`

**Artifact storage tests** - Tests the artifact store functionality for saving and retrieving files.

## Running Tests

### Local Tests

```bash
cd cumulus/tests
python3 test_unified_checkpointing.py
```

### Remote Tests (requires GPU server setup)

```bash
cd cumulus/tests
python3 test_unified_s3_complete.py
```

### All Tests

```bash
cd cumulus/tests
python3 test_*.py
```

## Prerequisites

### For Local Tests

- Python 3.8+
- PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM (optional - tests skip missing frameworks)

### For Remote Tests

- GPU server running Cumulus worker
- SSH tunnel: `ssh -p 44027 root@96.241.192.5 -L 8080:localhost:8081 -N`
- S3 credentials configured (for S3 tests)

## Test Results

All tests return JSON results with framework status:

```json
[
  {
    "framework": "pytorch",
    "status": "ok",
    "s3": true,
    "save_path": "/tmp/remote-s3/ckpt_1_10.pt"
  }
]
```

Status values:

- `"ok"`: Test passed
- `"skipped"`: Framework not available or S3 not configured
- `"error"`: Test failed with error details
