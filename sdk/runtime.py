"""
Cumulus Runtime Helper - Provides checkpointing and pause/resume functionality
"""

import os
import json
import time
import torch
from typing import Dict, Any, Optional, List


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
