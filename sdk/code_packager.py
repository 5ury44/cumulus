"""
CodePackager - Handles packaging of Python code and dependencies
"""

import zipfile
import tempfile
import os
import inspect
import ast
import importlib.util
import json
from typing import List, Dict, Any, Callable, Set
import sys


class CodePackager:
    """
    Packages Python functions and their dependencies into a ZIP file for remote execution.
    """
    
    def __init__(self):
        self.imported_modules: Set[str] = set()
        self.source_files: Dict[str, str] = {}
    
    def package_function(self, 
                        func: Callable, 
                        requirements: List[str],
                        job_id: str,
                        args: List[Any] = None,
                        **kwargs) -> bytes:
        """
        Package a function and its dependencies into a ZIP file.
        
        Args:
            func: Function to package
            requirements: List of required packages
            job_id: Unique job identifier
            args: List of positional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            ZIP file as bytes
        """
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add runtime helper
                runtime_content = self._get_runtime_helper()
                zip_file.writestr('runtime.py', runtime_content)
                
                # Add distributed checkpointing module
                distributed_checkpointer_content = self._get_distributed_checkpointer()
                zip_file.writestr('distributed_checkpointer.py', distributed_checkpointer_content)
                
                # Add main execution script
                main_script = self._generate_main_script(func, job_id, args or [], **kwargs)
                zip_file.writestr('main.py', main_script)
                
                # Add requirements
                requirements_content = '\n'.join(requirements) if requirements else ''
                zip_file.writestr('requirements.txt', requirements_content)
                
                # Add function source code
                func_source = self._extract_function_source(func)
                zip_file.writestr('function.py', func_source)
                
                # Add any additional source files
                for filename, content in self.source_files.items():
                    zip_file.writestr(filename, content)
                
                # Add job configuration
                job_config = {
                    'job_id': job_id,
                    'function_name': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                zip_file.writestr('job_config.json', json.dumps(job_config, indent=2))
            
            # Read ZIP file as bytes
            with open(temp_file.name, 'rb') as f:
                zip_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return zip_data
    
    def _generate_main_script(self, func: Callable, job_id: str, args: List[Any], **kwargs) -> str:
        """Generate the main execution script."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated execution script for job {job_id}
"""

import sys
import json
import os
import traceback
from function import {func.__name__}

def main():
    """Main execution function."""
    try:
        # Load job configuration
        with open('job_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"🚀 Starting job {{config['job_id']}}")
        print(f"Function: {{config['function_name']}}")
        
        # Import and call the function
        args = config.get('args', [])
        kwargs = config.get('kwargs', {{}})
        result = {func.__name__}(*args, **kwargs)
        
        # Save result
        with open('result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print("✅ Job completed successfully")
        
    except Exception as e:
        error_info = {{
            'error': str(e),
            'traceback': traceback.format_exc()
        }}
        
        with open('error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"❌ Job failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _extract_function_source(self, func: Callable) -> str:
        """Extract the source code of a function and its dependencies."""
        try:
            # Get the source code of the function
            source = inspect.getsource(func)
            
            # Parse the AST to find imports and other dependencies
            tree = ast.parse(source)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Build the complete source file
            complete_source = '\n'.join(imports) + '\n\n' + source
            
            return complete_source
            
        except Exception as e:
            # Fallback: just return the function source with proper indentation
            source = inspect.getsource(func)
            # Remove any leading indentation issues
            lines = source.split('\n')
            if lines and lines[0].startswith('    '):
                # Remove common indentation
                min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
                lines = [line[min_indent:] if line.strip() else line for line in lines]
            return '\n'.join(lines)
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.names:
                    names = ', '.join([alias.name for alias in node.names])
                    imports.append(f"from {module} import {names}")
        
        return list(set(imports))  # Remove duplicates
    
    def add_source_file(self, filename: str, content: str):
        """Add an additional source file to the package."""
        self.source_files[filename] = content
    
    def _get_runtime_helper(self) -> str:
        """Get the runtime helper code."""
        return '''"""
Cumulus Runtime Helper - Provides checkpointing and pause/resume functionality
"""

import os
import json
import time
import signal
import torch
from typing import Dict, Any, Optional, List, Callable

# Import distributed checkpointing if available
try:
    from distributed_checkpointer import DistributedCheckpointer, create_distributed_checkpointer
    DISTRIBUTED_CHECKPOINTING_AVAILABLE = True
except ImportError:
    DISTRIBUTED_CHECKPOINTING_AVAILABLE = False

# Optional boto3 import for S3 artifact storage
try:
    import boto3
    _BOTO3_AVAILABLE = True
except Exception:
    boto3 = None
    _BOTO3_AVAILABLE = False


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
def save_checkpoint(model, optimizer=None, epoch: int = 0, step: int = 0, extra: dict = None, framework: str = None, **kwargs):
    """Save checkpoint with unified, framework-agnostic API.

    Uses distributed checkpointer when available; falls back to local.
    """
    checkpointer = get_checkpointer(**kwargs)
    if hasattr(checkpointer, 'save_checkpoint'):
        return checkpointer.save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, step=step, extra=extra, framework=framework)
    return checkpointer.save(model, optimizer, epoch, step, extra)


def load_checkpoint(model=None, optimizer=None, checkpoint_path: str = None, framework: str = None, **kwargs):
    """Load checkpoint with unified, framework-agnostic API."""
    checkpointer = get_checkpointer(**kwargs)
    if hasattr(checkpointer, 'load_checkpoint'):
        return checkpointer.load_checkpoint(model=model, optimizer=optimizer, checkpoint_path=checkpoint_path, framework=framework)
    return checkpointer.load(model, optimizer, checkpoint_path)


def find_latest_checkpoint(**kwargs):
    """Find the latest checkpoint using the appropriate checkpointer."""
    checkpointer = get_checkpointer(**kwargs)
    if hasattr(checkpointer, 'find_latest_checkpoint'):
        return checkpointer.find_latest_checkpoint()
    elif hasattr(checkpointer, 'exists') and checkpointer.exists():
        return checkpointer.path
    return None


class _SaveOnInterrupt:
    """Context manager to save a checkpoint on SIGINT/SIGTERM/KeyboardInterrupt.

    Usage:
        with save_on_interrupt(ckpt, model, optimizer, lambda: epoch, lambda: step):
            ... training loop ...
    """

    def __init__(self,
                 checkpointer: Checkpointer,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 get_epoch: Callable[[], int],
                 get_step: Callable[[], int]):
        self.checkpointer = checkpointer
        self.model = model
        self.optimizer = optimizer
        self.get_epoch = get_epoch
        self.get_step = get_step
        self._prev_sigint = None
        self._prev_sigterm = None

    def __enter__(self):
        def _handler(signum, frame):
            try:
                epoch = int(self.get_epoch())
                step = int(self.get_step())
                self.checkpointer.save(self.model, self.optimizer, epoch, step)
            except Exception:
                pass
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                raise SystemExit(0)

        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
        return False


def save_on_interrupt(checkpointer: Checkpointer,
                      model: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      get_epoch: Callable[[], int],
                      get_step: Callable[[], int]) -> _SaveOnInterrupt:
    """Return a context manager that saves a checkpoint on Ctrl-C/terminate."""
    return _SaveOnInterrupt(checkpointer, model, optimizer, get_epoch, get_step)


class LocalArtifactStore:
    """Local artifact store that mirrors the distributed interface (L1 only)."""

    def __init__(self, local_dir: Optional[str] = None):
        self.local_dir = local_dir or job_dir()
        os.makedirs(self.local_dir, exist_ok=True)

    def _artifact_local_path(self, name: str) -> str:
        safe_name = os.path.basename(name)
        return os.path.join(self.local_dir, safe_name)

    def save_artifact_file(self, name: str, src_path: str) -> Dict[str, Optional[str]]:
        import shutil
        dst = self._artifact_local_path(name)
        shutil.copy2(src_path, dst)
        return {"local_path": dst, "s3_key": None}

    def save_artifact_bytes(self, name: str, data: bytes) -> Dict[str, Optional[str]]:
        dst = self._artifact_local_path(name)
        with open(dst, 'wb') as f:
            f.write(data)
        return {"local_path": dst, "s3_key": None}

    def load_artifact_to_path(self, name: str, dst_path: Optional[str] = None) -> str:
        src = self._artifact_local_path(name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Artifact not found: {name}")
        if dst_path:
            import shutil
            shutil.copy2(src, dst_path)
            return dst_path
        return src

    def load_artifact_bytes(self, name: str) -> bytes:
        src = self._artifact_local_path(name)
        with open(src, 'rb') as f:
            return f.read()

    def list_artifacts(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for fname in os.listdir(self.local_dir):
            if prefix and not fname.startswith(prefix):
                continue
            fpath = os.path.join(self.local_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith('ckpt_') and not fname.endswith('.pt'):
                items.append({
                    'name': fname,
                    'path': fpath,
                    'source': 'L1 (local)',
                    'timestamp': os.path.getmtime(fpath)
                })
        return sorted(items, key=lambda x: x['timestamp'], reverse=True)


class S3ArtifactStore(LocalArtifactStore):
    """S3-backed artifact store (L2) with optional local L1 cache."""

    def __init__(self,
                 job_id: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 s3_region: Optional[str] = None,
                 local_dir: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        # Use local L1 for caching and mirroring writes
        super().__init__(local_dir=local_dir or os.getenv('CUMULUS_LOCAL_CACHE_DIR', job_dir()))
        self.job_id = job_id or os.getenv('CUMULUS_JOB_ID') or 'unknown_job'
        self.s3_bucket = s3_bucket or os.getenv('CUMULUS_S3_BUCKET')
        self.s3_region = s3_region or os.getenv('CUMULUS_S3_REGION', 'us-east-1')

        self.s3_client = None
        if _BOTO3_AVAILABLE and self.s3_bucket:
            try:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.s3_region
                )
                self.s3_client = session.client('s3')
            except Exception:
                self.s3_client = None

    def _s3_key(self, name: str) -> str:
        safe_name = os.path.basename(name)
        return f"artifacts/{self.job_id}/{safe_name}"

    def save_artifact_file(self, name: str, src_path: str) -> Dict[str, Optional[str]]:
        info = super().save_artifact_file(name, src_path)
        s3_key = None
        if self.s3_client and self.s3_bucket:
            try:
                s3_key = self._s3_key(name)
                self.s3_client.upload_file(info['local_path'], self.s3_bucket, s3_key)
            except Exception:
                s3_key = None
        return {"local_path": info['local_path'], "s3_key": s3_key}

    def save_artifact_bytes(self, name: str, data: bytes) -> Dict[str, Optional[str]]:
        info = super().save_artifact_bytes(name, data)
        s3_key = None
        if self.s3_client and self.s3_bucket:
            try:
                s3_key = self._s3_key(name)
                import io
                self.s3_client.upload_fileobj(io.BytesIO(data), self.s3_bucket, s3_key)
            except Exception:
                s3_key = None
        return {"local_path": info['local_path'], "s3_key": s3_key}

    def load_artifact_to_path(self, name: str, dst_path: Optional[str] = None) -> str:
        # Try local first
        try:
            return super().load_artifact_to_path(name, dst_path)
        except FileNotFoundError:
            pass

        # Fallback to S3
        if not (self.s3_client and self.s3_bucket):
            raise FileNotFoundError(f"Artifact not found locally and S3 not configured: {name}")

        s3_key = self._s3_key(name)
        local_target = dst_path or self._artifact_local_path(name)
        os.makedirs(os.path.dirname(local_target), exist_ok=True)
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, local_target)
            return local_target
        except Exception as e:
            raise FileNotFoundError(f"Failed to download artifact from S3 {s3_key}: {e}")

    def load_artifact_bytes(self, name: str) -> bytes:
        # Try local first
        try:
            return super().load_artifact_bytes(name)
        except Exception:
            pass

        # Fallback to S3
        if not (self.s3_client and self.s3_bucket):
            raise FileNotFoundError(f"Artifact not found locally and S3 not configured: {name}")

        s3_key = self._s3_key(name)
        try:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return obj['Body'].read()
        except Exception as e:
            raise FileNotFoundError(f"Failed to load artifact from S3 {s3_key}: {e}")

    def list_artifacts(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        # Start with local list
        items = super().list_artifacts(prefix)

        # Merge with S3 list if available
        if self.s3_client and self.s3_bucket:
            try:
                s3_prefix = f"artifacts/{self.job_id}/"
                if prefix:
                    s3_prefix = s3_prefix + prefix
                resp = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=s3_prefix)
                if 'Contents' in resp:
                    for obj in resp['Contents']:
                        fname = os.path.basename(obj['Key'])
                        if not fname:
                            continue
                        items.append({
                            'name': fname,
                            'path': f"s3://{self.s3_bucket}/{obj['Key']}",
                            'source': 'L2 (S3)',
                            'timestamp': obj['LastModified'].timestamp()
                        })
            except Exception:
                pass
        # Newest first
        return sorted(items, key=lambda x: x['timestamp'], reverse=True)


def get_artifact_store(job_id: str = None, use_distributed: bool = True, **kwargs):
    """Get S3 artifact store if configured, else local store.

    If use_distributed is True and S3 is configured, returns S3ArtifactStore.
    Otherwise returns LocalArtifactStore.
    """
    if use_distributed:
        # Prefer S3 artifact store if configured
        if _BOTO3_AVAILABLE and os.getenv('CUMULUS_S3_BUCKET'):
            return S3ArtifactStore(job_id=job_id, **kwargs)
    return LocalArtifactStore()
'''
    
    def _get_distributed_checkpointer(self) -> str:
        """Get the distributed checkpointing module code."""
        return '''"""
Distributed Checkpointing System with L1 (Local) and L2 (S3) Caches

This module provides cross-machine checkpointing capabilities, allowing jobs
to be resumed on different machines by leveraging S3 as a shared checkpoint store.
"""

import os
import json
import time
import socket
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import boto3, but don't fail if it's not available
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None


class DistributedCheckpointer:
    """
    Distributed checkpointing system with L1 (local) and L2 (S3) caches.
    
    Features:
    - L1 Cache: Fast local disk storage for recent checkpoints
    - L2 Cache: S3 storage for cross-machine access and persistence
    - Automatic checkpoint discovery and loading
    - Job metadata tracking for coordination
    """
    
    def __init__(self, 
                 job_id: str,
                 s3_bucket: Optional[str] = None,
                 s3_region: str = "us-west-2",
                 local_dir: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        """
        Initialize distributed checkpointer.
        
        Args:
            job_id: Unique identifier for the job
            s3_bucket: S3 bucket name for L2 cache (optional)
            s3_region: AWS region for S3 bucket
            local_dir: Local directory for L1 cache
            aws_access_key_id: AWS access key (optional, can use env vars)
            aws_secret_access_key: AWS secret key (optional, can use env vars)
        """
        self.job_id = job_id
        self.s3_bucket = s3_bucket or os.getenv('CUMULUS_S3_BUCKET')
        self.s3_region = s3_region or os.getenv('CUMULUS_S3_REGION', 'us-west-2')
        self.local_dir = local_dir or os.getenv('CUMULUS_LOCAL_CACHE_DIR', '/tmp/cumulus/checkpoints')
        self.machine_id = socket.gethostname()
        
        # Ensure local directory exists
        os.makedirs(self.local_dir, exist_ok=True)
        
        # Initialize S3 client if bucket is configured and boto3 is available
        self.s3_client = None
        if self.s3_bucket and BOTO3_AVAILABLE:
            try:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.s3_region
                )
                self.s3_client = session.client('s3')
                logger.info(f"Initialized S3 client for bucket: {self.s3_bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
                self.s3_client = None
        elif self.s3_bucket and not BOTO3_AVAILABLE:
            logger.warning("S3 bucket configured but boto3 not available. Install boto3 for S3 support.")
        
        # Job metadata tracking
        self.metadata_key = f"jobs/{self.job_id}/metadata.json"
        
    def save(self, 
             model, 
             optimizer, 
             epoch: int, 
             step: int,
             extra: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Save checkpoint to both L1 (local) and L2 (S3) caches.
        
        Args:
            model: PyTorch model to checkpoint
            optimizer: Optimizer to checkpoint
            epoch: Current epoch number
            step: Current step number
            extra: Additional data to save
            
        Returns:
            Dict with 'local_path' and 's3_key' of saved checkpoint
        """
        checkpoint_id = f"ckpt_{epoch}_{step}"
        
        # Create checkpoint data
        import torch
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'rng_cpu': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'extra': extra or {},
            'metadata': {
                'job_id': self.job_id,
                'machine_id': self.machine_id,
                'timestamp': time.time(),
                'checkpoint_id': checkpoint_id,
                'created_at': datetime.utcnow().isoformat()
            }
        }
        
        # Save to L1 cache (local)
        local_path = os.path.join(self.local_dir, f"{checkpoint_id}.pt")
        torch.save(checkpoint, local_path)
        logger.info(f"✅ Checkpoint saved to L1 cache: {local_path}")
        
        # Save to L2 cache (S3) if available
        s3_key = None
        if self.s3_client:
            s3_key = f"checkpoints/{self.job_id}/{checkpoint_id}.pt"
            try:
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                logger.info(f"✅ Checkpoint synced to L2 cache: s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"❌ Failed to sync checkpoint to S3: {e}")
                s3_key = None
        
        # Update job metadata
        self._update_job_metadata(epoch, step, checkpoint_id, local_path, s3_key)
        
        return {
            'local_path': local_path,
            's3_key': s3_key,
            'checkpoint_id': checkpoint_id
        }
    
    def load(self, 
             model, 
             optimizer,
             checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint from L1 or L2 cache.
        
        Args:
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Specific checkpoint path (optional)
            
        Returns:
            Loaded checkpoint state
        """
        if checkpoint_path:
            # Load specific checkpoint
            local_path = self._ensure_local_from_checkpoint_path(checkpoint_path)
        else:
            # Find latest checkpoint
            local_path = self._find_latest_checkpoint()
        
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"No checkpoint found at: {local_path}")
        
        # Load checkpoint
        import torch
        state = torch.load(local_path, map_location='cpu')
        
        # Restore model and optimizer states
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        
        # Restore RNG states
        torch.random.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        
        logger.info(f"✅ Checkpoint loaded from: {local_path}")
        logger.info(f"📊 Resuming from epoch {state['epoch']}, step {state['step']}")
        
        return state

    # -------------------------------
    # Unified, framework-agnostic API
    # -------------------------------
    def _write_sidecar_metadata(self, data_file: str, framework: str, epoch: int, step: int, extra: Dict[str, Any]):
        meta = {
            'framework': framework,
            'epoch': epoch,
            'step': step,
            'job_id': self.job_id,
            'created_at': time.time(),
            'extra': extra or {}
        }
        meta_path = f"{data_file}.meta.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            if self.s3_client:
                key = f"checkpoints/{self.job_id}/{os.path.basename(meta_path)}"
                self.s3_client.upload_file(meta_path, self.s3_bucket, key)
        except Exception:
            pass

    def _infer_framework_from_model(self, model) -> str:
        mod = type(model).__module__ if model is not None else ''
        if mod.startswith('torch.'):
            return 'pytorch'
        if mod.startswith(('tensorflow.', 'keras.')):
            return 'tensorflow'
        if mod.startswith('sklearn.'):
            return 'sklearn'
        if mod.startswith('xgboost.'):
            return 'xgboost'
        if mod.startswith('lightgbm.'):
            return 'lightgbm'
        return 'pytorch'

    def save_checkpoint(self,
                        model=None,
                        optimizer=None,
                        epoch: int = 0,
                        step: int = 0,
                        extra: Dict[str, Any] = None,
                        framework: Optional[str] = None) -> Dict[str, Optional[str]]:
        fw = (framework or self._infer_framework_from_model(model)).lower()
        ckpt_id = f"ckpt_{epoch}_{step}"
        if fw in ('pytorch', 'torch'):
            info = self.save(model, optimizer, epoch, step, extra)
            # ensure sidecar present
            self._write_sidecar_metadata(info['local_path'], 'pytorch', epoch, step, extra or {})
            return info
        elif fw in ('tensorflow', 'tf', 'keras'):
            # weights-only for TF/Keras
            fname = f"{ckpt_id}.weights.h5"
            local_path = os.path.join(self.local_dir, fname)
            os.makedirs(self.local_dir, exist_ok=True)
            model.save_weights(local_path)
            self._write_sidecar_metadata(local_path, 'tensorflow', epoch, step, extra or {})
            s3_key = None
            if self.s3_client:
                s3_key = f"checkpoints/{self.job_id}/{fname}"
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                except Exception:
                    s3_key = None
            self._update_job_metadata(epoch, step, ckpt_id, local_path, s3_key)
            return {'local_path': local_path, 's3_key': s3_key, 'checkpoint_id': ckpt_id}
        elif fw == 'sklearn':
            import joblib
            fname = f"{ckpt_id}.pkl"
            local_path = os.path.join(self.local_dir, fname)
            os.makedirs(self.local_dir, exist_ok=True)
            joblib.dump(model, local_path)
            self._write_sidecar_metadata(local_path, 'sklearn', epoch, step, extra or {})
            s3_key = None
            if self.s3_client:
                s3_key = f"checkpoints/{self.job_id}/{fname}"
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                except Exception:
                    s3_key = None
            self._update_job_metadata(epoch, step, ckpt_id, local_path, s3_key)
            return {'local_path': local_path, 's3_key': s3_key, 'checkpoint_id': ckpt_id}
        elif fw == 'xgboost':
            fname = f"{ckpt_id}.json"
            local_path = os.path.join(self.local_dir, fname)
            os.makedirs(self.local_dir, exist_ok=True)
            model.save_model(local_path)
            self._write_sidecar_metadata(local_path, 'xgboost', epoch, step, extra or {})
            s3_key = None
            if self.s3_client:
                s3_key = f"checkpoints/{self.job_id}/{fname}"
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                except Exception:
                    s3_key = None
            self._update_job_metadata(epoch, step, ckpt_id, local_path, s3_key)
            return {'local_path': local_path, 's3_key': s3_key, 'checkpoint_id': ckpt_id}
        elif fw == 'lightgbm':
            fname = f"{ckpt_id}.txt"
            local_path = os.path.join(self.local_dir, fname)
            os.makedirs(self.local_dir, exist_ok=True)
            model.save_model(local_path)
            self._write_sidecar_metadata(local_path, 'lightgbm', epoch, step, extra or {})
            s3_key = None
            if self.s3_client:
                s3_key = f"checkpoints/{self.job_id}/{fname}"
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                except Exception:
                    s3_key = None
            self._update_job_metadata(epoch, step, ckpt_id, local_path, s3_key)
            return {'local_path': local_path, 's3_key': s3_key, 'checkpoint_id': ckpt_id}
        else:
            # default to PyTorch behavior
            info = self.save(model, optimizer, epoch, step, extra)
            self._write_sidecar_metadata(info['local_path'], 'pytorch', epoch, step, extra or {})
            return info

    def load_checkpoint(self,
                        model=None,
                        optimizer=None,
                        checkpoint_path: Optional[str] = None,
                        framework: Optional[str] = None) -> Dict[str, Any]:
        # Resolve path
        local_path = (self._ensure_local_from_checkpoint_path(checkpoint_path)
                      if checkpoint_path else self._find_latest_checkpoint())
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"No checkpoint found at: {local_path}")
        fw = (framework or '').lower()
        base = os.path.basename(local_path).lower()
        if not fw:
            if base.endswith('.pt'):
                fw = 'pytorch'
            elif base.endswith(('.h5', 'weights.h5')):
                fw = 'tensorflow'
            elif base.endswith('.pkl'):
                fw = 'sklearn'
            elif base.endswith('.json'):
                fw = 'xgboost'
            elif base.endswith('.txt'):
                fw = 'lightgbm'
            else:
                fw = self._infer_framework_from_model(model)
        if fw in ('pytorch', 'torch'):
            state = self.load(model, optimizer, local_path)
            return {'local_path': local_path, 'framework': 'pytorch', 'state': state}
        elif fw in ('tensorflow', 'tf', 'keras'):
            model.load_weights(local_path)
            return {'local_path': local_path, 'framework': 'tensorflow', 'state': {'model': 'loaded'}}
        elif fw == 'sklearn':
            import joblib
            obj = joblib.load(local_path)
            return {'local_path': local_path, 'framework': 'sklearn', 'state': obj}
        elif fw == 'xgboost':
            import xgboost as xgb
            booster = model or xgb.Booster()
            booster.load_model(local_path)
            return {'local_path': local_path, 'framework': 'xgboost', 'state': booster}
        elif fw == 'lightgbm':
            import lightgbm as lgb
            if model is None:
                booster = lgb.Booster(model_file=local_path)
            else:
                model.load_model(local_path)
                booster = model
            return {'local_path': local_path, 'framework': 'lightgbm', 'state': booster}
        else:
            # fallback to pytorch loader
            state = self.load(model, optimizer, local_path)
            return {'local_path': local_path, 'framework': 'pytorch', 'state': state}

    def _ensure_local_from_checkpoint_path(self, checkpoint_path: str) -> str:
        """Resolve a checkpoint path which may be:
        - a local filesystem path
        - a full S3 URL (s3://bucket/key)
        - an S3 key (checkpoints/<job_id>/ckpt_X_Y.pt)
        - a bare filename (ckpt_X_Y.pt) which will be resolved to checkpoints/<job_id>/filename
        Returns a local filesystem path to the checkpoint file, downloading if necessary.
        """
        # If it's already a local file, use it
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        # If it's an S3 URL, download it
        if checkpoint_path.startswith('s3://'):
            return self._download_from_s3(checkpoint_path)

        # If we have S3 configured, treat it as an S3 key or filename
        if self.s3_client and self.s3_bucket:
            # If it's a bare filename, prefix with standard checkpoints path
            if '/' not in checkpoint_path:
                key = f"checkpoints/{self.job_id}/{checkpoint_path}"
            else:
                key = checkpoint_path

            # Download to local cache
            local_path = os.path.join(self.local_dir, os.path.basename(key))
            try:
                self.s3_client.download_file(self.s3_bucket, key, local_path)
                logger.info(f"Downloaded checkpoint from S3 key: s3://{self.s3_bucket}/{key} -> {local_path}")
                return local_path
            except Exception as e:
                raise FileNotFoundError(f"Failed to download checkpoint from S3 key '{key}': {e}")

        # Fallback: return as-is (will error later if invalid)
        return checkpoint_path
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint across L1 and L2 caches."""
        return self._find_latest_checkpoint()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        # Check L1 cache
        for fname in os.listdir(self.local_dir):
            if fname.startswith('ckpt_') and fname.endswith('.pt'):
                fpath = os.path.join(self.local_dir, fname)
                try:
                    state = torch.load(fpath, map_location='cpu')
                    checkpoints.append({
                        'filename': fname,
                        'path': fpath,
                        'epoch': state.get('epoch', 0),
                        'step': state.get('step', 0),
                        'timestamp': os.path.getmtime(fpath),
                        'source': 'L1 (local)',
                        'metadata': state.get('metadata', {})
                    })
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {fname}: {e}")
        
        # Check L2 cache (S3)
        if self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"checkpoints/{self.job_id}/"
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if obj['Key'].endswith('.pt'):
                            # Extract checkpoint info from S3 metadata
                            s3_key = obj['Key']
                            filename = os.path.basename(s3_key)
                            
                            checkpoints.append({
                                'filename': filename,
                                'path': f"s3://{self.s3_bucket}/{s3_key}",
                                'epoch': 0,  # Would need to download to get full info
                                'step': 0,
                                'timestamp': obj['LastModified'].timestamp(),
                                'source': 'L2 (S3)',
                                'metadata': {}
                            })
            except Exception as e:
                logger.warning(f"Failed to list S3 checkpoints: {e}")
        
        # Sort by timestamp (newest first)
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint, checking L1 first, then L2."""
        
        # Check L1 cache first (fastest)
        local_checkpoints = []
        for fname in os.listdir(self.local_dir):
            if fname.startswith('ckpt_') and fname.endswith('.pt'):
                fpath = os.path.join(self.local_dir, fname)
                local_checkpoints.append((fpath, os.path.getmtime(fpath)))
        
        if local_checkpoints:
            latest_local = max(local_checkpoints, key=lambda x: x[1])[0]
            logger.info(f"Found latest checkpoint in L1 cache: {latest_local}")
            return latest_local
        
        # Check L2 cache (S3) if no local checkpoints
        if self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"checkpoints/{self.job_id}/"
                )
                
                if 'Contents' in response:
                    # Find latest checkpoint in S3
                    latest_s3 = max(response['Contents'], key=lambda x: x['LastModified'])
                    s3_key = latest_s3['Key']
                    
                    # Download to L1 cache
                    local_path = os.path.join(self.local_dir, os.path.basename(s3_key))
                    self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                    
                    logger.info(f"Found latest checkpoint in L2 cache, downloaded to L1: {local_path}")
                    return local_path
                    
            except Exception as e:
                logger.error(f"Error accessing S3 for checkpoint discovery: {e}")
        
        return None
    
    def _download_from_s3(self, s3_path: str) -> str:
        """Download checkpoint from S3 to local cache."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        # Parse S3 path: s3://bucket/key
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        path_parts = s3_path[5:].split('/', 1)  # Remove 's3://' and split
        bucket = path_parts[0]
        key = path_parts[1]
        
        # Download to local cache
        local_path = os.path.join(self.local_dir, os.path.basename(key))
        self.s3_client.download_file(bucket, key, local_path)
        
        logger.info(f"Downloaded checkpoint from S3: {s3_path} -> {local_path}")
        return local_path
    
    def _update_job_metadata(self, 
                            epoch: int, 
                            step: int, 
                            checkpoint_id: str,
                            local_path: str,
                            s3_key: Optional[str]):
        """Update job metadata in S3 for cross-machine coordination."""
        if not self.s3_client:
            return
        
        metadata = {
            'job_id': self.job_id,
            'status': 'running',
            'current_machine': self.machine_id,
            'last_checkpoint': checkpoint_id,
            'last_epoch': epoch,
            'last_step': step,
            'checkpoint_paths': {
                'local': local_path,
                's3': s3_key
            },
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=self.metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            logger.debug(f"Updated job metadata: {self.metadata_key}")
        except Exception as e:
            logger.warning(f"Failed to update job metadata: {e}")
    
    def get_job_metadata(self) -> Optional[Dict[str, Any]]:
        """Get job metadata from S3."""
        if not self.s3_client:
            return None
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=self.metadata_key
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.debug(f"No job metadata found: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()
        
        # Keep only the most recent checkpoints
        to_keep = checkpoints[:keep_last]
        to_remove = checkpoints[keep_last:]
        
        for checkpoint in to_remove:
            try:
                if checkpoint['source'] == 'L1 (local)':
                    os.remove(checkpoint['path'])
                    logger.info(f"Removed old checkpoint: {checkpoint['path']}")
                elif checkpoint['source'] == 'L2 (S3)' and self.s3_client:
                    # Parse S3 path and delete
                    s3_path = checkpoint['path']
                    path_parts = s3_path[5:].split('/', 1)
                    bucket = path_parts[0]
                    key = path_parts[1]
                    
                    self.s3_client.delete_object(Bucket=bucket, Key=key)
                    logger.info(f"Removed old checkpoint from S3: {s3_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")


def create_distributed_checkpointer(job_id: str, **kwargs) -> DistributedCheckpointer:
    """
    Factory function to create a distributed checkpointer.
    
    Args:
        job_id: Unique job identifier
        **kwargs: Additional arguments passed to DistributedCheckpointer
        
    Returns:
        Configured DistributedCheckpointer instance
    """
    return DistributedCheckpointer(job_id, **kwargs)
'''
    
    def package_directory(self, directory_path: str, requirements: List[str]) -> bytes:
        """
        Package an entire directory for remote execution.
        
        Args:
            directory_path: Path to directory to package
            requirements: List of required packages
            
        Returns:
            ZIP file as bytes
        """
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add all Python files from directory
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, directory_path)
                            zip_file.write(file_path, arcname)
                
                # Add requirements
                requirements_content = '\n'.join(requirements) if requirements else ''
                zip_file.writestr('requirements.txt', requirements_content)
                
                # Add execution script
                exec_script = self._generate_directory_exec_script()
                zip_file.writestr('main.py', exec_script)
            
            # Read ZIP file as bytes
            with open(temp_file.name, 'rb') as f:
                zip_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return zip_data
    
    def _generate_directory_exec_script(self) -> str:
        """Generate execution script for directory packages."""
        return '''#!/usr/bin/env python3
"""
Auto-generated execution script for directory package
"""

import sys
import json
import os
import traceback

def main():
    """Main execution function."""
    try:
        print("🚀 Starting directory package execution")
        
        # Look for main.py or __main__.py
        if os.path.exists('main.py'):
            exec(open('main.py').read())
        elif os.path.exists('__main__.py'):
            exec(open('__main__.py').read())
        else:
            print("❌ No main.py or __main__.py found")
            sys.exit(1)
        
        print("✅ Package execution completed successfully")
        
    except Exception as e:
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        
        with open('error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"❌ Package execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
