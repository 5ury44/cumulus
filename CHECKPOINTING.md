# Cumulus Checkpointing and Pause/Resume

This document describes the checkpointing and pause/resume functionality in the Cumulus SDK.

## Overview

The Cumulus SDK now supports:

- **Automatic checkpointing** during training
- **Pause/resume** functionality for long-running jobs
- **Cooperative pausing** at batch boundaries
- **Checkpoint management** and listing

## Key Features

### 1. Automatic Checkpointing

- Save model state, optimizer state, and training progress
- Configurable checkpoint frequency (by steps or time)
- Automatic RNG state preservation for reproducibility

### 2. Pause/Resume

- Pause running jobs gracefully at batch boundaries
- Resume from the exact point where paused
- Control via API endpoints or client methods

### 3. Checkpoint Management

- List available checkpoints for any job
- Resume from specific checkpoints
- Automatic cleanup and organization

## Usage

### Hands-Free Automatic Checkpointing

Every job that runs through `CumulusClient.run()` automatically imports the runtime helper and initializes the `AutoCheckpointManager`. The manager hooks into the active ML framework and checkpoints transparently—no extra code required.

For example, a plain PyTorch loop:

```python
def train_pytorch():
    import torch
    model = torch.nn.Linear(16, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(500):
        loss = model(torch.randn(32, 16)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()      # AutoCheckpointManager intercepts this call
    return {"loss": float(loss)}

client = CumulusClient("http://localhost:8081")
result = client.run(func=train_pytorch, gpu_memory=0.4, duration=600)
```

Behind the scenes the packaged runtime will:

- Register optimizer and model ownership automatically
- Save checkpoints every `CUMULUS_CHECKPOINT_EVERY_STEPS` (default `100`) or `CUMULUS_CHECKPOINT_EVERY_SECONDS` (default `0`, disabled)
- Persist to local disk or S3 (when distributed checkpointing is configured)
- Record pause events if `control.json` requests a cooperative stop

Environment variables (set on the worker or passed into the job) control cadence:

| Variable                           | Default | Meaning                                         |
| ---------------------------------- | ------- | ----------------------------------------------- |
| `CUMULUS_AUTO_CHECKPOINT`          | `true`  | Disable auto mode by setting to `false`         |
| `CUMULUS_CHECKPOINT_EVERY_STEPS`   | `100`   | Save every N optimizer/training steps           |
| `CUMULUS_CHECKPOINT_EVERY_SECONDS` | `0`     | Minimum seconds between auto saves (0 = ignore) |

#### Cooperative Pause via `control.json`

- Each packaged job runs in `CUMULUS_JOB_DIR` (typically `/tmp/cumulus_jobs/<job_id>/`).
- The runtime watches `control.json` inside that directory:

```json
{ "pause": true }
```

- Writing the file (either manually or via `job.pause()`) triggers:
  1. An immediate checkpoint on the next safe boundary.
  2. A `runtime.CooperativePause` exception.
  3. A structured result (`{"status": "paused", "checkpoint": {...}}`) saved to `result.json`.
- Clearing the flag (`{"pause": false}`) or deleting the file lets `job.resume()` continue.

Current framework hooks:

- **PyTorch** – wraps `torch.nn.Module` registration and `Optimizer.step`. Fully supported for local and distributed checkpoints.
- **TensorFlow / Keras** – injects a callback into `model.fit` and watches batch boundaries.
- **scikit-learn** – wraps `BaseEstimator.fit`, saving full estimators.
- **XGBoost** – adds a training callback and auto-populates `xgb_model` for resuming.
- **LightGBM** – installs a callback and auto-populates `init_model` for resuming.

If a framework is unavailable, the manager logs a warning and falls back to manual usage without interrupting execution.

### Declarative Checkpoint API (Manual Control)

Manual calls still work and can be mixed with automatic checkpoints:

```python
from runtime import save_checkpoint, load_checkpoint, should_pause

def train_with_manual_saves():
    import torch
    model = torch.nn.Linear(784, 10)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(5):
        # ... training steps ...
        if epoch % 2 == 0:
            save_checkpoint(model, opt, epoch=epoch, step=0, framework="pytorch")
        if should_pause():
            save_checkpoint(model, opt, epoch=epoch, step=0, framework="pytorch")
            return {"status": "paused"}
    return {"status": "done"}
```

### Manual Pause/Resume Workflow

### Pause/Resume Control

```python
# Start a job
job = client.run_async(func=train_with_checkpoints, gpu_memory=0.8, duration=600)

# Pause the job
job.pause()

# Check available checkpoints
checkpoints = job.get_checkpoints()
print(f"Available checkpoints: {len(checkpoints)}")

# Resume the job
job.resume()

# Wait for completion
result = job.result()
```

### Resuming from Checkpoint

```python
def resume_training(checkpoint_path):
    # Load checkpoint and resume training
    ckpt = Checkpointer(checkpoint_path)
    if ckpt.exists():
        state = ckpt.load(model, optimizer)
        start_epoch = state['epoch']
        start_step = state['step']
        # Continue training from this point
    # ... rest of training code

# Resume from specific checkpoint
result = client.run(
    func=resume_training,
    args=[checkpoint_path],
    gpu_memory=0.8,
    duration=300
)
```

## API Reference

### Runtime Helper Functions

#### `Checkpointer(fname='checkpoint.pt')`

Main checkpointing class.

**Methods:**

- `exists()` - Check if checkpoint exists
- `save(model, optimizer, epoch, step, extra=None)` - Save checkpoint
- `load(model, optimizer)` - Load checkpoint
- `time_to_checkpoint(step, every_steps=None, every_seconds=None)` - Check if time to checkpoint

#### `should_pause() -> bool`

Check if job should pause (called by training loop).

#### `list_checkpoints() -> List[Dict]`

List all available checkpoints in job directory.

### Client Methods

#### `client.pause_job(job_id) -> Dict`

Pause a running job.

#### `client.resume_job(job_id) -> Dict`

Resume a paused job.

#### `client.get_checkpoints(job_id) -> List[Dict]`

Get available checkpoints for a job.

### Job Object Methods

#### `job.pause() -> Dict`

Pause this job.

#### `job.resume() -> Dict`

Resume this job.

#### `job.get_checkpoints() -> List[Dict]`

Get checkpoints for this job.

## Examples

### Example 1: Basic Checkpointed Training

```bash
python cumulus/examples/checkpointed_training.py
```

### Example 2: Pause/Resume Demonstration

```bash
python cumulus/examples/pause_resume_demo.py
```

### Example 3: Test Checkpointing

```bash
python cumulus/test_checkpointing.py
```

## Best Practices

1. **Leave auto-checkpointing enabled** unless you have a specific reason to opt out.
2. **Tune cadence with environment variables** to balance robustness and throughput.
3. **Call `save_checkpoint()` manually** at milestone events where you need deterministic snapshots.
4. **Trigger pauses via `job.pause()` or `control.json`** instead of killing the process.
5. **Resume with `find_latest_checkpoint()`** to pick up the most recent automatic save.
6. **Monitor storage** (local cache + S3) and prune old checkpoints if needed.

## Limitations

- **Pause granularity**: Can only pause at batch boundaries, not mid-operation
- **Checkpoint size**: Large models may create large checkpoint files
- **Storage**: Checkpoints are stored locally on the worker (not persistent across restarts)
- **Concurrency**: Only one pause/resume operation per job at a time

## Troubleshooting

### Common Issues

1. **Job doesn't pause**: For manual loops, ensure `should_pause()` is called frequently; for auto mode confirm `control.json` is writable and the framework hook is active.
2. **Resume fails**: Check that checkpoint file exists and is valid
3. **Large checkpoints**: Consider reducing checkpoint frequency or model size
4. **Storage issues**: Monitor disk space on worker nodes

### Debug Tips

- Check job status: `job.status()`
- List checkpoints: `job.get_checkpoints()`
- Verify pause signal: Check `control.json` file in job directory
- Monitor logs: Check worker logs for errors

## Future Enhancements

- **Distributed checkpointing** across multiple workers
- **Checkpoint compression** to reduce storage requirements
- **Automatic checkpoint cleanup** based on age or count
- **Checkpoint validation** to ensure integrity
- **Resume from any checkpoint** in training history
