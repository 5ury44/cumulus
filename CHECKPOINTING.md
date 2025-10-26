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

### Basic Checkpointed Training

```python
from cumulus.sdk import CumulusClient
from runtime import Checkpointer, should_pause

def train_with_checkpoints():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model and optimizer
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32)

    # Initialize checkpointing
    ckpt = Checkpointer()

    # Training loop
    for epoch in range(10):
        for step, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Check for pause signal
            if should_pause():
                ckpt.save(model, optimizer, epoch, step)
                return {'status': 'paused', 'checkpoint': ckpt.path}

            # Periodic checkpointing
            if ckpt.time_to_checkpoint(step, every_steps=100):
                ckpt.save(model, optimizer, epoch, step)

    return {'status': 'completed'}

# Run training
client = CumulusClient("http://localhost:8081")
result = client.run(func=train_with_checkpoints, gpu_memory=0.8, duration=300)
```

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

1. **Check for pause signals frequently** - Call `should_pause()` in your training loop
2. **Use appropriate checkpoint frequency** - Balance between safety and performance
3. **Handle pause gracefully** - Save state and return appropriate status
4. **Test pause/resume** - Verify your training can resume correctly
5. **Monitor checkpoint storage** - Clean up old checkpoints as needed

## Limitations

- **Pause granularity**: Can only pause at batch boundaries, not mid-operation
- **Checkpoint size**: Large models may create large checkpoint files
- **Storage**: Checkpoints are stored locally on the worker (not persistent across restarts)
- **Concurrency**: Only one pause/resume operation per job at a time

## Troubleshooting

### Common Issues

1. **Job doesn't pause**: Ensure `should_pause()` is called in training loop
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
