"""
Distributed Checkpointing System with L1 (Local) and L2 (S3) Caches

This module provides cross-machine checkpointing capabilities, allowing jobs
to be resumed on different machines by leveraging S3 as a shared checkpoint store.
"""

import os
import json
import time
import socket
import boto3
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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
        
        # Initialize S3 client if bucket is configured
        self.s3_client = None
        if self.s3_bucket:
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
        
        # Job metadata tracking
        self.metadata_key = f"jobs/{self.job_id}/metadata.json"
        
    def save(self, 
             model: torch.nn.Module, 
             optimizer: torch.optim.Optimizer, 
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
             model: torch.nn.Module, 
             optimizer: torch.optim.Optimizer,
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

    # -------------------------------
    # Generic Artifact Store (L1/L2)
    # -------------------------------
    def _artifact_local_path(self, name: str) -> str:
        safe_name = os.path.basename(name)
        return os.path.join(self.local_dir, safe_name)

    def _artifact_s3_key(self, name: str) -> Optional[str]:
        if not self.s3_client:
            return None
        safe_name = os.path.basename(name)
        return f"artifacts/{self.job_id}/{safe_name}"

    def save_artifact_file(self, name: str, src_path: str) -> Dict[str, Optional[str]]:
        """Save an arbitrary file as an artifact to L1 and optionally L2.

        Returns dict with 'local_path' and optional 's3_key'.
        """
        local_path = self._artifact_local_path(name)
        try:
            import shutil
            shutil.copy2(src_path, local_path)
            logger.info(f"✅ Artifact saved to L1: {local_path}")
        except Exception as e:
            logger.error(f"❌ Failed to copy artifact to L1: {e}")
            raise

        s3_key = self._artifact_s3_key(name)
        if s3_key:
            try:
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                logger.info(f"✅ Artifact synced to L2: s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"❌ Failed to sync artifact to S3: {e}")
                s3_key = None

        return {"local_path": local_path, "s3_key": s3_key}

    def save_artifact_bytes(self, name: str, data: bytes) -> Dict[str, Optional[str]]:
        """Save raw bytes as an artifact to L1 and optionally L2."""
        local_path = self._artifact_local_path(name)
        try:
            with open(local_path, 'wb') as f:
                f.write(data)
            logger.info(f"✅ Artifact bytes saved to L1: {local_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write artifact to L1: {e}")
            raise

        s3_key = self._artifact_s3_key(name)
        if s3_key:
            try:
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                logger.info(f"✅ Artifact synced to L2: s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"❌ Failed to sync artifact to S3: {e}")
                s3_key = None

        return {"local_path": local_path, "s3_key": s3_key}

    def load_artifact_to_path(self, name: str, dst_path: Optional[str] = None) -> str:
        """Ensure artifact is present locally and return its path, optionally copying to dst_path."""
        local_path = self._artifact_local_path(name)
        if not os.path.exists(local_path) and self.s3_client:
            s3_key = self._artifact_s3_key(name)
            if s3_key:
                try:
                    self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                    logger.info(f"Downloaded artifact from L2: s3://{self.s3_bucket}/{s3_key} -> {local_path}")
                except Exception as e:
                    raise FileNotFoundError(f"Artifact not found in L1 or L2: {name}; error: {e}")

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Artifact not found: {name}")

        if dst_path:
            import shutil
            shutil.copy2(local_path, dst_path)
            return dst_path
        return local_path

    def load_artifact_bytes(self, name: str) -> bytes:
        """Load artifact as bytes, downloading from L2 if needed."""
        path = self.load_artifact_to_path(name)
        with open(path, 'rb') as f:
            return f.read()

    def list_artifacts(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List artifacts across L1 and L2 (filenames only for S3 unless downloaded)."""
        results: List[Dict[str, Any]] = []

        # L1
        if os.path.isdir(self.local_dir):
            for fname in os.listdir(self.local_dir):
                if prefix and not fname.startswith(prefix):
                    continue
                fpath = os.path.join(self.local_dir, fname)
                if os.path.isfile(fpath) and not fname.startswith('ckpt_') and not fname.endswith('.pt'):
                    results.append({
                        'name': fname,
                        'path': fpath,
                        'source': 'L1 (local)',
                        'timestamp': os.path.getmtime(fpath)
                    })

        # L2
        if self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"artifacts/{self.job_id}/"
                )
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if prefix and not os.path.basename(key).startswith(prefix):
                            continue
                        results.append({
                            'name': os.path.basename(key),
                            'path': f"s3://{self.s3_bucket}/{key}",
                            'source': 'L2 (S3)',
                            'timestamp': obj['LastModified'].timestamp()
                        })
            except Exception as e:
                logger.debug(f"Failed to list S3 artifacts: {e}")

        return sorted(results, key=lambda x: x['timestamp'], reverse=True)


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
