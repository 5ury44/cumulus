"""
Configuration system for Cumulus distributed checkpointing
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DistributedCheckpointingConfig:
    """Configuration for distributed checkpointing system."""
    
    # S3 Configuration
    s3_bucket: Optional[str] = None
    s3_region: str = "us-west-2"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # Local Cache Configuration
    local_cache_dir: str = "/tmp/cumulus/checkpoints"
    cache_size_limit_gb: float = 10.0
    keep_checkpoints: int = 5
    
    # Checkpointing Strategy
    checkpoint_every_steps: int = 100
    checkpoint_every_seconds: int = 300
    auto_cleanup: bool = True
    
    # Job Metadata
    enable_job_metadata: bool = True
    metadata_ttl_seconds: int = 86400  # 24 hours
    
    @classmethod
    def from_env(cls) -> 'DistributedCheckpointingConfig':
        """Create configuration from environment variables."""
        return cls(
            s3_bucket=os.getenv('CUMULUS_S3_BUCKET'),
            s3_region=os.getenv('CUMULUS_S3_REGION', 'us-west-2'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            local_cache_dir=os.getenv('CUMULUS_LOCAL_CACHE_DIR', '/tmp/cumulus/checkpoints'),
            cache_size_limit_gb=float(os.getenv('CUMULUS_CACHE_SIZE_LIMIT_GB', '10.0')),
            keep_checkpoints=int(os.getenv('CUMULUS_KEEP_CHECKPOINTS', '5')),
            checkpoint_every_steps=int(os.getenv('CUMULUS_CHECKPOINT_EVERY_STEPS', '100')),
            checkpoint_every_seconds=int(os.getenv('CUMULUS_CHECKPOINT_EVERY_SECONDS', '300')),
            auto_cleanup=os.getenv('CUMULUS_AUTO_CLEANUP', 'true').lower() == 'true',
            enable_job_metadata=os.getenv('CUMULUS_ENABLE_JOB_METADATA', 'true').lower() == 'true',
            metadata_ttl_seconds=int(os.getenv('CUMULUS_METADATA_TTL_SECONDS', '86400'))
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DistributedCheckpointingConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def is_s3_configured(self) -> bool:
        """Check if S3 is properly configured."""
        return (
            self.s3_bucket is not None and 
            self.aws_access_key_id is not None and 
            self.aws_secret_access_key is not None
        )
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any errors."""
        errors = {}
        
        if self.s3_bucket and not self.is_s3_configured():
            errors['s3_credentials'] = "S3 bucket specified but AWS credentials missing"
        
        if self.cache_size_limit_gb <= 0:
            errors['cache_size'] = "Cache size limit must be positive"
        
        if self.keep_checkpoints <= 0:
            errors['keep_checkpoints'] = "Keep checkpoints must be positive"
        
        if self.checkpoint_every_steps <= 0:
            errors['checkpoint_every_steps'] = "Checkpoint every steps must be positive"
        
        if self.checkpoint_every_seconds <= 0:
            errors['checkpoint_every_seconds'] = "Checkpoint every seconds must be positive"
        
        return errors


def get_config() -> DistributedCheckpointingConfig:
    """Get the current distributed checkpointing configuration."""
    return DistributedCheckpointingConfig.from_env()


def setup_config(s3_bucket: str = None,
                 s3_region: str = "us-west-2",
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 local_cache_dir: str = "/tmp/cumulus/checkpoints",
                 **kwargs) -> DistributedCheckpointingConfig:
    """
    Setup distributed checkpointing configuration.
    
    Args:
        s3_bucket: S3 bucket name for L2 cache
        s3_region: AWS region for S3 bucket
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        local_cache_dir: Local directory for L1 cache
        **kwargs: Additional configuration options
        
    Returns:
        Configured DistributedCheckpointingConfig instance
    """
    config = DistributedCheckpointingConfig(
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        local_cache_dir=local_cache_dir,
        **kwargs
    )
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")
    
    return config


def print_config_status():
    """Print the current configuration status."""
    config = get_config()
    
    print("🔧 Cumulus Distributed Checkpointing Configuration:")
    print(f"  📦 S3 Bucket: {config.s3_bucket or 'Not configured'}")
    print(f"  🌍 S3 Region: {config.s3_region}")
    print(f"  💾 Local Cache: {config.local_cache_dir}")
    print(f"  📏 Cache Size Limit: {config.cache_size_limit_gb} GB")
    print(f"  🔄 Keep Checkpoints: {config.keep_checkpoints}")
    print(f"  ⏱️  Checkpoint Every Steps: {config.checkpoint_every_steps}")
    print(f"  ⏰ Checkpoint Every Seconds: {config.checkpoint_every_seconds}")
    print(f"  🧹 Auto Cleanup: {config.auto_cleanup}")
    print(f"  📊 Job Metadata: {config.enable_job_metadata}")
    
    if config.is_s3_configured():
        print("  ✅ S3 Configuration: Valid")
    else:
        print("  ⚠️  S3 Configuration: Not configured (will use local-only checkpointing)")
    
    errors = config.validate()
    if errors:
        print("  ❌ Configuration Errors:")
        for key, error in errors.items():
            print(f"    - {key}: {error}")
    else:
        print("  ✅ Configuration: Valid")


# Environment variable documentation
ENV_VARS_DOC = """
Environment Variables for Cumulus Distributed Checkpointing:

Required for S3 (L2 Cache):
  CUMULUS_S3_BUCKET          - S3 bucket name for checkpoint storage
  AWS_ACCESS_KEY_ID          - AWS access key ID
  AWS_SECRET_ACCESS_KEY      - AWS secret access key

Optional Configuration:
  CUMULUS_S3_REGION          - AWS region (default: us-west-2)
  CUMULUS_LOCAL_CACHE_DIR    - Local cache directory (default: /tmp/cumulus/checkpoints)
  CUMULUS_CACHE_SIZE_LIMIT_GB - Max local cache size in GB (default: 10.0)
  CUMULUS_KEEP_CHECKPOINTS   - Number of checkpoints to keep locally (default: 5)
  CUMULUS_CHECKPOINT_EVERY_STEPS - Checkpoint every N steps (default: 100)
  CUMULUS_CHECKPOINT_EVERY_SECONDS - Checkpoint every N seconds (default: 300)
  CUMULUS_AUTO_CLEANUP       - Enable automatic cleanup (default: true)
  CUMULUS_ENABLE_JOB_METADATA - Enable job metadata tracking (default: true)
  CUMULUS_METADATA_TTL_SECONDS - Job metadata TTL in seconds (default: 86400)

Example setup:
  export CUMULUS_S3_BUCKET="my-checkpoint-bucket"
  export AWS_ACCESS_KEY_ID="your-access-key"
  export AWS_SECRET_ACCESS_KEY="your-secret-key"
  export CUMULUS_S3_REGION="us-west-2"
  export CUMULUS_LOCAL_CACHE_DIR="/tmp/cumulus/checkpoints"
"""
