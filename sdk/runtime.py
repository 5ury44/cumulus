"""
Cumulus Runtime Helper - Provides checkpointing and pause/resume functionality
"""

import os
import json
import time
import torch
from typing import Dict, Any, Optional, List

# Import distributed checkpointing if available
try:
    from .distributed_checkpointer import DistributedCheckpointer, create_distributed_checkpointer
    DISTRIBUTED_CHECKPOINTING_AVAILABLE = True
except ImportError:
    DISTRIBUTED_CHECKPOINTING_AVAILABLE = False


def job_dir():
    """Get the job directory from environment variable."""
    return os.getenv('CUMULUS_JOB_DIR', os.getcwd())


def _control_path():
    """Get the path to the control file."""
    return os.path.join(job_dir(), 'control.json')


def should_pause() -> bool:
    """Check if the job should pause."""
    p = _control_path()
    if not os.path.exists(p): 
        return False
    try:
        with open(p, 'r') as f:
            return bool(json.load(f).get('pause', False))
    except Exception:
        return False


class Checkpointer:
    """Handles model checkpointing and resuming."""
    
    def __init__(self, fname: str = 'checkpoint.pt'):
        self.path = os.path.join(job_dir(), fname)
        self._last_ts = 0

    def exists(self):
        """Check if checkpoint exists."""
        return os.path.exists(self.path)

    def save(self, model, optimizer, epoch: int, step: int, extra: dict = None):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'step': step,
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'rng_cpu': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'extra': extra or {}
        }
        torch.save(state, self.path)
        self._last_ts = time.time()
        return self.path

    def load(self, model, optimizer):
        """Load model checkpoint."""
        state = torch.load(self.path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        return state

    def time_to_checkpoint(self, step: int, every_steps: int = None, every_seconds: int = None):
        """Check if it's time to checkpoint."""
        by_step = (every_steps is not None and step > 0 and step % every_steps == 0)
        by_time = (every_seconds is not None and (time.time() - self._last_ts) >= every_seconds)
        return by_step or by_time


def list_checkpoints() -> List[Dict[str, Any]]:
    """List available checkpoints in the job directory."""
    checkpoints = []
    job_d = job_dir()
    
    for fname in os.listdir(job_d):
        if fname.endswith('.pt'):
            fpath = os.path.join(job_d, fname)
            try:
                state = torch.load(fpath, map_location='cpu')
                checkpoints.append({
                    'filename': fname,
                    'path': fpath,
                    'epoch': state.get('epoch', 0),
                    'step': state.get('step', 0),
                    'timestamp': os.path.getmtime(fpath)
                })
            except Exception:
                continue
    
    return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)


def get_distributed_checkpointer(job_id: str = None, **kwargs) -> Optional[DistributedCheckpointer]:
    """
    Get a distributed checkpointer instance if available and configured.
    
    Args:
        job_id: Job ID (defaults to CUMULUS_JOB_ID env var)
        **kwargs: Additional arguments for DistributedCheckpointer
        
    Returns:
        DistributedCheckpointer instance or None if not available/configured
    """
    if not DISTRIBUTED_CHECKPOINTING_AVAILABLE:
        return None
    
    # Get job ID from environment if not provided
    if job_id is None:
        job_id = os.getenv('CUMULUS_JOB_ID')
        if not job_id:
            return None
    
    # Check if S3 is configured
    s3_bucket = os.getenv('CUMULUS_S3_BUCKET')
    if not s3_bucket:
        return None
    
    try:
        return create_distributed_checkpointer(job_id, **kwargs)
    except Exception as e:
        print(f"Warning: Failed to create distributed checkpointer: {e}")
        return None


def get_checkpointer(job_id: str = None, use_distributed: bool = True, **kwargs):
    """
    Get the appropriate checkpointer (distributed or local).
    
    Args:
        job_id: Job ID for distributed checkpointing
        use_distributed: Whether to prefer distributed checkpointing
        **kwargs: Additional arguments for checkpointers
        
    Returns:
        Checkpointer instance (DistributedCheckpointer or Checkpointer)
    """
    if use_distributed:
        distributed_ckpt = get_distributed_checkpointer(job_id, **kwargs)
        if distributed_ckpt:
            return distributed_ckpt
    
    # Fall back to local checkpointer
    return Checkpointer()


# Convenience functions for backward compatibility
def save_checkpoint(model, optimizer, epoch: int, step: int, extra: dict = None, **kwargs):
    """Save checkpoint using the appropriate checkpointer."""
    checkpointer = get_checkpointer(**kwargs)
    return checkpointer.save(model, optimizer, epoch, step, extra)


def load_checkpoint(model, optimizer, checkpoint_path: str = None, **kwargs):
    """Load checkpoint using the appropriate checkpointer."""
    checkpointer = get_checkpointer(**kwargs)
    return checkpointer.load(model, optimizer, checkpoint_path)


def find_latest_checkpoint(**kwargs):
    """Find the latest checkpoint using the appropriate checkpointer."""
    checkpointer = get_checkpointer(**kwargs)
    if hasattr(checkpointer, 'find_latest_checkpoint'):
        return checkpointer.find_latest_checkpoint()
    elif hasattr(checkpointer, 'exists') and checkpointer.exists():
        return checkpointer.path
    return None
