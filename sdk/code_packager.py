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
import runtime
from function import {func.__name__}

def main():
    """Main execution function."""
    try:
        # Load job configuration
        with open('job_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"🚀 Starting job {{config['job_id']}}")
        print(f"Function: {{config['function_name']}}")
        
        # Ensure auto-checkpoint hooks are initialized
        runtime.get_auto_checkpoint_manager()

        # Import and call the function
        args = config.get('args', [])
        kwargs = config.get('kwargs', {{}})
        result = {func.__name__}(*args, **kwargs)
        
        # Save result
        with open('result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print("✅ Job completed successfully")
        
    except runtime.CooperativePause as pause:
        pause_info = runtime.get_last_checkpoint_info() or {{}}
        pause_payload = {{
            'status': 'paused',
            'reason': str(pause) or 'pause requested',
            'checkpoint': pause_info
        }}
        with open('result.json', 'w') as f:
            json.dump(pause_payload, f, indent=2, default=str)
        print("⏸️ Job paused cooperatively")
        sys.exit(0)
        
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
import importlib
import importlib.util
import functools
import types
import weakref
try:
    import torch
except Exception:
    torch = None
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
        if torch is None:
            raise RuntimeError("PyTorch is required for local checkpointing")
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
        if torch is None:
            raise RuntimeError("PyTorch is required for local checkpointing")
        state = torch.load(self.path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        return state

    # Unified wrappers for local checkpoints
    def save_checkpoint(self, model, optimizer=None, epoch: int = 0, step: int = 0, extra: dict = None, framework: str = None):
        path = self.save(model, optimizer, epoch, step, extra)
        return {"local_path": path, "s3_key": None, "checkpoint_id": f"ckpt_{epoch}_{step}"}

    def load_checkpoint(self, model=None, optimizer=None, checkpoint_path: str = None, framework: str = None):
        if checkpoint_path:
            self.path = checkpoint_path
        state = self.load(model, optimizer) if (model is not None and optimizer is not None) else torch.load(self.path, map_location='cpu')
        return {"local_path": self.path, "framework": (framework or "pytorch"), "state": state}

    def time_to_checkpoint(self, step: int, every_steps: int = None, every_seconds: int = None):
        """Check if it's time to checkpoint."""
        by_step = (every_steps is not None and step > 0 and step % every_steps == 0)
        by_time = (every_seconds is not None and (time.time() - self._last_ts) >= every_seconds)
        return by_step or by_time


def list_checkpoints() -> List[Dict[str, Any]]:
    """List available checkpoints in the job directory."""
    if torch is None:
        return []
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


_LAST_CHECKPOINT_INFO: Optional[Dict[str, Any]] = None


def _update_last_checkpoint(info: Optional[Dict[str, Any]]):
    """Store metadata about the most recent checkpoint save."""
    global _LAST_CHECKPOINT_INFO
    if info is not None:
        _LAST_CHECKPOINT_INFO = info


def get_last_checkpoint_info() -> Optional[Dict[str, Any]]:
    """Retrieve metadata for the most recently saved checkpoint."""
    return _LAST_CHECKPOINT_INFO


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


# Unified convenience functions
def save_checkpoint(model, optimizer=None, epoch: int = 0, step: int = 0, extra: dict = None, framework: str = None, **kwargs):
    """Save checkpoint with unified, framework-agnostic API."""
    checkpointer = get_checkpointer(**kwargs)
    info = checkpointer.save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, step=step, extra=extra, framework=framework)
    _update_last_checkpoint(info)
    return info


def load_checkpoint(model=None, optimizer=None, checkpoint_path: str = None, framework: str = None, **kwargs):
    """Load checkpoint with unified, framework-agnostic API."""
    checkpointer = get_checkpointer(**kwargs)
    return checkpointer.load_checkpoint(model=model, optimizer=optimizer, checkpoint_path=checkpoint_path, framework=framework)


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
                # Always use unified API
                self.checkpointer.save_checkpoint(model=self.model, optimizer=self.optimizer, epoch=epoch, step=step, framework="pytorch")
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


class CooperativePause(RuntimeError):
    """Raised when cooperative pause is requested via control signal."""


class AutoCheckpointManager:
    """Coordinates transparent checkpointing for supported frameworks."""

    def __init__(self):
        self.enabled = os.getenv('CUMULUS_AUTO_CHECKPOINT', 'true').lower() not in ('0', 'false', 'no', 'off')
        try:
            self.every_steps = int(os.getenv('CUMULUS_CHECKPOINT_EVERY_STEPS', '0') or '0')
        except Exception:
            self.every_steps = 0
        try:
            self.every_seconds = float(os.getenv('CUMULUS_CHECKPOINT_EVERY_SECONDS', '0') or '0')
        except Exception:
            self.every_seconds = 0.0

        if self.every_steps <= 0:
            self.every_steps = 100
        if self.every_seconds < 0:
            self.every_seconds = 0.0

        self.checkpointer = None
        self.distributed = False
        self._torch_param_to_module: Dict[int, weakref.ReferenceType] = {}
        self._torch_param_finalizers: Dict[int, weakref.finalize] = {}
        self._torch_optimizer_state = weakref.WeakKeyDictionary()
        self.framework_state: Dict[str, Dict[str, Any]] = {}
        self._resume_cache: Dict[str, Optional[str]] = {}
        self._paused_frameworks: Dict[str, bool] = {}
        self.last_checkpoint_info: Optional[Dict[str, Any]] = None
        self.framework_status: Dict[str, str] = {}

        if not self.enabled:
            return

        try:
            self.checkpointer = get_checkpointer()
            self.distributed = not isinstance(self.checkpointer, Checkpointer)
        except Exception as exc:
            print(f"[cumulus.auto] Failed to initialize checkpointer: {exc}")
            self.enabled = False
            return

        self._enable_frameworks()

    def _enable_frameworks(self):
        self._safe_enable('pytorch', self._enable_torch)
        if self.distributed:
            self._safe_enable('tensorflow', self._enable_tensorflow)
            self._safe_enable('sklearn', self._enable_sklearn)
            self._safe_enable('xgboost', self._enable_xgboost)
            self._safe_enable('lightgbm', self._enable_lightgbm)
        else:
            for name in ('tensorflow', 'sklearn', 'xgboost', 'lightgbm'):
                self.framework_status.setdefault(name, 'skipped (distributed checkpointing disabled)')

    def _safe_enable(self, name: str, func: Callable[[], None]):
        try:
            func()
        except Exception as exc:
            self.framework_status[name] = f'error: {exc}'
            print(f"[cumulus.auto] Failed to enable {name} auto-checkpointing: {exc}")

    @staticmethod
    def _module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _framework_state_for(self, name: str) -> Dict[str, Any]:
        state = self.framework_state.setdefault(name, {})
        state.setdefault('step', 0)
        state.setdefault('last_ts', time.time())
        state.setdefault('resumed', False)
        return state

    def _record_checkpoint(self, info: Optional[Dict[str, Any]]):
        if info:
            self.last_checkpoint_info = info
            _update_last_checkpoint(info)

    def _should_save(self, step: int, last_ts: float) -> bool:
        save_by_step = self.every_steps and step > 0 and (step % self.every_steps == 0)
        save_by_time = self.every_seconds and ((time.time() - last_ts) >= self.every_seconds)
        return bool(save_by_step or save_by_time)

    def _record_pause(self, framework: str):
        self._paused_frameworks[framework] = True

    def pause_requested(self, framework: Optional[str] = None) -> bool:
        if framework:
            return self._paused_frameworks.get(framework, False)
        return any(self._paused_frameworks.values())

    def _raise_pause(self):
        raise CooperativePause("Pause requested by Cumulus scheduler")

    def get_framework_status(self) -> Dict[str, str]:
        return dict(self.framework_status)

    def _latest_checkpoint(self, framework: str) -> Optional[str]:
        if framework in self._resume_cache:
            return self._resume_cache[framework]
        if hasattr(self.checkpointer, 'list_checkpoints'):
            try:
                entries = self.checkpointer.list_checkpoints()
            except Exception:
                entries = []
            for item in entries:
                meta = item.get('metadata') or {}
                fw = (meta.get('framework') or item.get('framework') or '').lower()
                if fw == framework:
                    path = item.get('path')
                    if path:
                        self._resume_cache[framework] = path
                        return path
                    s3_key = item.get('s3_key')
                    if s3_key:
                        self._resume_cache[framework] = s3_key
                        return s3_key
            return None
        return None

    def _enable_torch(self):
        if torch is None:
            self.framework_status['pytorch'] = 'skipped (torch not available)'
            return
        module_cls = torch.nn.Module
        if getattr(module_cls, "_cumulus_auto_ckpt", False):
            self.framework_status['pytorch'] = 'ok'
            return
        manager = self

        original_add_module = module_cls.add_module
        original_register_parameter = module_cls.register_parameter

        def add_module_wrapper(self_module, name, module):
            result = original_add_module(self_module, name, module)
            if module is not None:
                try:
                    module.__dict__["_cumulus_parent"] = weakref.ref(self_module)
                except Exception:
                    pass
            return result

        def register_parameter_wrapper(self_module, name, param):
            result = original_register_parameter(self_module, name, param)
            if param is not None:
                try:
                    key = id(param)
                    manager._torch_param_to_module[key] = weakref.ref(self_module)
                    try:
                        manager._torch_param_finalizers[key] = weakref.finalize(
                            param,
                            lambda k=key: manager._torch_param_to_module.pop(k, None)
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
            return result

        module_cls.add_module = add_module_wrapper
        module_cls.register_parameter = register_parameter_wrapper
        module_cls._cumulus_auto_ckpt = True
        self.framework_status['pytorch'] = 'ok'

        opt_cls = torch.optim.Optimizer
        if getattr(opt_cls, "_cumulus_auto_ckpt", False):
            return
        original_init = opt_cls.__init__
        original_step = opt_cls.step

        def init_wrapper(opt_self, params, defaults):
            original_init(opt_self, params, defaults)
            manager._register_torch_optimizer(opt_self)

        def step_wrapper(opt_self, *args, **kwargs):
            result = original_step(opt_self, *args, **kwargs)
            manager._after_torch_step(opt_self)
            return result

        opt_cls.__init__ = init_wrapper
        opt_cls.step = step_wrapper
        opt_cls._cumulus_auto_ckpt = True

    def _register_torch_optimizer(self, optimizer):
        if optimizer in self._torch_optimizer_state:
            return
        state = {'step': 0, 'last_ts': time.time(), 'model': None, 'resumed': False}
        self._torch_optimizer_state[optimizer] = state
        model = self._infer_torch_model(optimizer)
        if model is not None:
            state['model'] = model
            self._maybe_resume_torch(model, optimizer, state)

    def _infer_torch_model(self, optimizer):
        modules = []
        for group in getattr(optimizer, 'param_groups', []):
            for param in group.get('params', []):
                module_ref = self._torch_param_to_module.get(id(param))
                if module_ref is None:
                    continue
                module = module_ref() if isinstance(module_ref, weakref.ReferenceType) else module_ref
                if module is None:
                    continue
                root = module
                visited = set()
                while hasattr(root, "_cumulus_parent") and getattr(root, "_cumulus_parent") is not None and root not in visited:
                    visited.add(root)
                    parent = getattr(root, "_cumulus_parent")
                    if isinstance(parent, weakref.ReferenceType):
                        parent = parent()
                    if parent is None:
                        break
                    root = parent
                if root not in modules:
                    modules.append(root)
        if len(modules) == 1:
            return modules[0]
        if modules:
            try:
                return modules[0]
            except Exception:
                return None
        return None

    def _maybe_resume_torch(self, model, optimizer, state):
        if state.get('resumed'):
            return
        try:
            if isinstance(self.checkpointer, Checkpointer):
                if self.checkpointer.exists():
                    info = self.checkpointer.load_checkpoint(model=model, optimizer=optimizer, framework="pytorch")
                    state['step'] = info['state'].get('step', 0)
                    state['last_ts'] = time.time()
                    self._record_checkpoint(info)
            else:
                path = self._latest_checkpoint('pytorch')
                if path:
                    info = self.checkpointer.load_checkpoint(model=model, optimizer=optimizer, checkpoint_path=path, framework="pytorch")
                    state['step'] = info['state'].get('step', 0)
                    state['last_ts'] = time.time()
                    self._record_checkpoint(info)
        except Exception as exc:
            print(f"[cumulus.auto] Failed to resume PyTorch checkpoint: {exc}")
        state['resumed'] = True

    def _save_torch_checkpoint(self, model, optimizer, step):
        if model is None:
            return None
        try:
            info = self.checkpointer.save_checkpoint(model=model, optimizer=optimizer, epoch=0, step=step, framework="pytorch", extra={'auto': True})
            self._record_checkpoint(info)
            return info
        except Exception as exc:
            print(f"[cumulus.auto] Failed to save PyTorch checkpoint: {exc}")
            return None

    def _after_torch_step(self, optimizer):
        if optimizer not in self._torch_optimizer_state:
            self._register_torch_optimizer(optimizer)
        state = self._torch_optimizer_state.get(optimizer)
        if not state:
            return
        state['step'] += 1
        if state.get('model') is None:
            state['model'] = self._infer_torch_model(optimizer)
            if state['model'] is not None:
                self._maybe_resume_torch(state['model'], optimizer, state)
        model = state.get('model')
        if model is None:
            return
        if self._should_save(state['step'], state.get('last_ts', 0.0)):
            info = self._save_torch_checkpoint(model, optimizer, state['step'])
            if info:
                state['last_ts'] = time.time()
        if should_pause():
            info = self._save_torch_checkpoint(model, optimizer, state['step'])
            if info:
                state['last_ts'] = time.time()
            self._record_pause('pytorch')
            self._raise_pause()

    def _enable_tensorflow(self):
        if not self._module_available('tensorflow'):
            self.framework_status['tensorflow'] = 'skipped (tensorflow not available)'
            return
        import tensorflow as tf
        if getattr(tf.keras.Model, "_cumulus_auto_ckpt", False):
            self.framework_status['tensorflow'] = 'ok'
            return
        manager = self

        class AutoCheckpointCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self._total_step = 0

            def on_train_begin(self, logs=None):
                manager._tf_before_train(self.model)

            def on_train_batch_end(self, batch, logs=None):
                self._total_step += 1
                manager._tf_after_batch(self.model, self._total_step)

        original_fit = tf.keras.Model.fit

        def fit_wrapper(model_self, *args, **kwargs):
            if not manager.enabled:
                return original_fit(model_self, *args, **kwargs)
            callbacks = list(kwargs.get('callbacks') or [])
            if not any(isinstance(cb, AutoCheckpointCallback) for cb in callbacks):
                callbacks.append(AutoCheckpointCallback())
            kwargs['callbacks'] = callbacks
            result = original_fit(model_self, *args, **kwargs)
            if manager.pause_requested('tensorflow'):
                manager._raise_pause()
            return result

        tf.keras.Model.fit = fit_wrapper
        tf.keras.Model._cumulus_auto_ckpt = True
        self.framework_status['tensorflow'] = 'ok'

    def _tf_before_train(self, model):
        state = self._framework_state_for('tensorflow')
        if state.get('resumed'):
            return
        path = self._latest_checkpoint('tensorflow')
        if path:
            try:
                info = self.checkpointer.load_checkpoint(model=model, checkpoint_path=path, framework='tensorflow')
                self._record_checkpoint(info)
            except Exception as exc:
                print(f"[cumulus.auto] Failed to resume TensorFlow checkpoint: {exc}")
        state['resumed'] = True

    def _tf_after_batch(self, model, step):
        state = self._framework_state_for('tensorflow')
        state['step'] = step
        if self._should_save(step, state.get('last_ts', 0.0)):
            try:
                info = self.checkpointer.save_checkpoint(model=model, epoch=0, step=step, framework='tensorflow', extra={'auto': True})
                self._record_checkpoint(info)
                state['last_ts'] = time.time()
            except Exception as exc:
                print(f"[cumulus.auto] Failed to save TensorFlow checkpoint: {exc}")
        if should_pause():
            try:
                info = self.checkpointer.save_checkpoint(model=model, epoch=0, step=step, framework='tensorflow', extra={'auto': True, 'reason': 'pause'})
                self._record_checkpoint(info)
            except Exception as exc:
                print(f"[cumulus.auto] Failed to save TensorFlow checkpoint during pause: {exc}")
            model.stop_training = True
            self._record_pause('tensorflow')

    def _enable_xgboost(self):
        if not self._module_available('xgboost'):
            self.framework_status['xgboost'] = 'skipped (xgboost not available)'
            return
        import xgboost as xgb
        if getattr(xgb, "_cumulus_auto_ckpt", False):
            self.framework_status['xgboost'] = 'ok'
            return
        manager = self

        class _AutoXGBCallback(xgb.callback.TrainingCallback):
            def __init__(self):
                self._last_step = 0

            def after_iteration(self, model, epoch, evals_log):
                step = epoch + 1
                self._last_step = step
                manager._xgb_after_iteration(model, step)
                if should_pause():
                    manager._xgb_after_iteration(model, step, force=True)
                    manager._record_pause('xgboost')
                    return True
                return False

            def after_training(self, model):
                manager._xgb_after_iteration(model, self._last_step, force=True)
                return model

        original_train = xgb.train

        def train_wrapper(params, dtrain, num_boost_round=10, *args, **kwargs):
            if not manager.enabled:
                return original_train(params, dtrain, num_boost_round, *args, **kwargs)
            callbacks = list(kwargs.get('callbacks') or [])
            callbacks.append(_AutoXGBCallback())
            kwargs['callbacks'] = callbacks
            resume = manager._latest_checkpoint('xgboost')
            if resume and 'xgb_model' not in kwargs:
                kwargs['xgb_model'] = resume
            booster = original_train(params, dtrain, num_boost_round, *args, **kwargs)
            if manager.pause_requested('xgboost'):
                manager._raise_pause()
            return booster

        xgb.train = train_wrapper
        xgb._cumulus_auto_ckpt = True
        self.framework_status['xgboost'] = 'ok'

    def _xgb_after_iteration(self, booster, step, force=False):
        state = self._framework_state_for('xgboost')
        if step > state.get('step', 0):
            state['step'] = step
        if not force and not self._should_save(step, state.get('last_ts', 0.0)):
            return
        try:
            info = self.checkpointer.save_checkpoint(model=booster, epoch=0, step=step, framework='xgboost', extra={'auto': True})
            self._record_checkpoint(info)
            state['last_ts'] = time.time()
        except Exception as exc:
            print(f"[cumulus.auto] Failed to save XGBoost checkpoint: {exc}")

    def _enable_lightgbm(self):
        if not self._module_available('lightgbm'):
            self.framework_status['lightgbm'] = 'skipped (lightgbm not available)'
            return
        import lightgbm as lgb
        if getattr(lgb, "_cumulus_auto_ckpt", False):
            self.framework_status['lightgbm'] = 'ok'
            return
        manager = self

        def auto_callback():
            def _callback(env):
                step = env.iteration + 1
                manager._lgb_after_iteration(env.model, step)
                if should_pause():
                    manager._lgb_after_iteration(env.model, step, force=True)
                    env.model.stop_training = True
                    manager._record_pause('lightgbm')
            _callback.order = 60
            return _callback

        original_train = lgb.train

        def train_wrapper(params, train_set, num_boost_round=100, *args, **kwargs):
            if not manager.enabled:
                return original_train(params, train_set, num_boost_round, *args, **kwargs)
            callbacks = list(kwargs.get('callbacks') or [])
            callbacks.append(auto_callback())
            kwargs['callbacks'] = callbacks
            resume = manager._latest_checkpoint('lightgbm')
            if resume and 'init_model' not in kwargs:
                kwargs['init_model'] = resume
            booster = original_train(params, train_set, num_boost_round, *args, **kwargs)
            try:
                manager._lgb_after_iteration(booster, booster.current_iteration(), force=True)
            except Exception:
                pass
            if manager.pause_requested('lightgbm'):
                manager._raise_pause()
            return booster

        lgb.train = train_wrapper
        lgb._cumulus_auto_ckpt = True
        self.framework_status['lightgbm'] = 'ok'

    def _lgb_after_iteration(self, booster, step, force=False):
        state = self._framework_state_for('lightgbm')
        if step > state.get('step', 0):
            state['step'] = step
        if not force and not self._should_save(step, state.get('last_ts', 0.0)):
            return
        try:
            info = self.checkpointer.save_checkpoint(model=booster, epoch=0, step=step, framework='lightgbm', extra={'auto': True})
            self._record_checkpoint(info)
            state['last_ts'] = time.time()
        except Exception as exc:
            print(f"[cumulus.auto] Failed to save LightGBM checkpoint: {exc}")

    def _enable_sklearn(self):
        if not self._module_available('sklearn'):
            self.framework_status['sklearn'] = 'skipped (sklearn not available)'
            return
        from sklearn.base import BaseEstimator
        if getattr(BaseEstimator, "_cumulus_auto_ckpt", False):
            self.framework_status['sklearn'] = 'ok'
            return
        manager = self
        original_getattribute = BaseEstimator.__getattribute__

        def getattribute_wrapper(estimator_self, name):
            attr = original_getattribute(estimator_self, name)
            if name == 'fit' and callable(attr):
                func = getattr(attr, '__func__', attr)
                if getattr(func, '_cumulus_wrapped', False):
                    return attr

                @functools.wraps(func)
                def wrapped(*args, **kwargs):
                    if manager.enabled:
                        manager._sklearn_before_fit(estimator_self)
                    result = func(estimator_self, *args, **kwargs)
                    if manager.enabled:
                        manager._sklearn_after_fit(estimator_self)
                        if manager.pause_requested('sklearn'):
                            manager._raise_pause()
                    return result

                setattr(func, '_cumulus_wrapped', True)
                return types.MethodType(wrapped, estimator_self)
            return attr

        BaseEstimator.__getattribute__ = getattribute_wrapper
        BaseEstimator._cumulus_auto_ckpt = True
        self.framework_status['sklearn'] = 'ok'

    def _sklearn_before_fit(self, estimator):
        state = self._framework_state_for('sklearn')
        if state.get('resumed'):
            return
        path = self._latest_checkpoint('sklearn')
        if path:
            try:
                info = self.checkpointer.load_checkpoint(checkpoint_path=path, framework='sklearn')
                restored = info.get('state')
                if restored is not None:
                    if hasattr(restored, '__dict__'):
                        estimator.__dict__.update(restored.__dict__)
                    elif isinstance(restored, dict):
                        estimator.__dict__.update(restored)
                self._record_checkpoint(info)
            except Exception as exc:
                print(f"[cumulus.auto] Failed to resume sklearn checkpoint: {exc}")
        state['resumed'] = True

    def _sklearn_after_fit(self, estimator):
        state = self._framework_state_for('sklearn')
        step = state.get('step', 0) + 1
        state['step'] = step
        try:
            info = self.checkpointer.save_checkpoint(model=estimator, epoch=0, step=step, framework='sklearn', extra={'auto': True})
            self._record_checkpoint(info)
            state['last_ts'] = time.time()
        except Exception as exc:
            print(f"[cumulus.auto] Failed to save sklearn checkpoint: {exc}")


_AUTO_MANAGER: Optional[AutoCheckpointManager] = None


def get_auto_checkpoint_manager() -> AutoCheckpointManager:
    """Return the singleton auto-checkpoint manager, creating it if needed."""
    global _AUTO_MANAGER
    if _AUTO_MANAGER is None:
        _AUTO_MANAGER = AutoCheckpointManager()
    return _AUTO_MANAGER


# Ensure auto-checkpointing hooks are registered on import.
get_auto_checkpoint_manager()
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
            # Save PyTorch checkpoint without legacy API
            fname = f"{ckpt_id}.pt"
            local_path = os.path.join(self.local_dir, fname)
            os.makedirs(self.local_dir, exist_ok=True)
            import torch
            state = {
                'epoch': epoch,
                'step': step,
                'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'optimizer': optimizer.state_dict() if optimizer is not None else {},
                'rng_cpu': torch.random.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'extra': extra or {}
            }
            torch.save(state, local_path)
            self._write_sidecar_metadata(local_path, 'pytorch', epoch, step, extra or {})
            s3_key = None
            if self.s3_client:
                s3_key = f"checkpoints/{self.job_id}/{fname}"
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                except Exception:
                    s3_key = None
            self._update_job_metadata(epoch, step, ckpt_id, local_path, s3_key)
            return {'local_path': local_path, 's3_key': s3_key, 'checkpoint_id': ckpt_id}
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
            # Default to PyTorch behavior via unified path
            fname = f"{ckpt_id}.pt"
            local_path = os.path.join(self.local_dir, fname)
            os.makedirs(self.local_dir, exist_ok=True)
            import torch
            state = {
                'epoch': epoch,
                'step': step,
                'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'optimizer': optimizer.state_dict() if optimizer is not None else {},
                'rng_cpu': torch.random.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'extra': extra or {}
            }
            torch.save(state, local_path)
            self._write_sidecar_metadata(local_path, 'pytorch', epoch, step, extra or {})
            s3_key = None
            if self.s3_client:
                s3_key = f"checkpoints/{self.job_id}/{fname}"
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                except Exception:
                    s3_key = None
            self._update_job_metadata(epoch, step, ckpt_id, local_path, s3_key)
            return {'local_path': local_path, 's3_key': s3_key, 'checkpoint_id': ckpt_id}

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
        
    except runtime.CooperativePause as pause:
        pause_info = runtime.get_last_checkpoint_info() or {}
        pause_payload = {
            'status': 'paused',
            'reason': str(pause) or 'pause requested',
            'checkpoint': pause_info
        }
        with open('result.json', 'w') as f:
            json.dump(pause_payload, f, indent=2, default=str)
        print("⏸️ Job paused cooperatively")
        sys.exit(0)

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
