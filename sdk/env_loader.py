"""
Environment loader for Cumulus distributed checkpointing
"""

import os
from typing import Dict, Any


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    if not os.path.exists(env_path):
        return env_vars
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                env_vars[key] = value
    
    return env_vars


def setup_distributed_checkpointing_env(env_path: str = ".env"):
    """
    Setup environment variables for distributed checkpointing.
    
    Args:
        env_path: Path to the .env file
    """
    env_vars = load_env_file(env_path)
    
    # Set environment variables
    for key, value in env_vars.items():
        if key.startswith('CUMULUS_') or key.startswith('AWS_'):
            os.environ[key] = value
    
    print(f"✅ Loaded {len(env_vars)} environment variables from {env_path}")
    
    # Print configuration status
    print_config_status()


def print_config_status():
    """Print the current distributed checkpointing configuration status."""
    print("\n🔧 Cumulus Distributed Checkpointing Configuration:")
    print(f"  📦 S3 Bucket: {os.getenv('CUMULUS_S3_BUCKET', 'Not configured')}")
    print(f"  🌍 S3 Region: {os.getenv('CUMULUS_S3_REGION', 'Not configured')}")
    print(f"  🔑 AWS Access Key: {'Configured' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not configured'}")
    print(f"  🔐 AWS Secret Key: {'Configured' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not configured'}")
    print(f"  💾 Local Cache: {os.getenv('CUMULUS_LOCAL_CACHE_DIR', '/tmp/cumulus/checkpoints')}")
    print(f"  📏 Cache Size Limit: {os.getenv('CUMULUS_CACHE_SIZE_LIMIT_GB', '10.0')} GB")
    print(f"  🔄 Keep Checkpoints: {os.getenv('CUMULUS_KEEP_CHECKPOINTS', '5')}")
    print(f"  ⏱️  Checkpoint Every Steps: {os.getenv('CUMULUS_CHECKPOINT_EVERY_STEPS', '100')}")
    print(f"  ⏰ Checkpoint Every Seconds: {os.getenv('CUMULUS_CHECKPOINT_EVERY_SECONDS', '300')}")
    print(f"  🧹 Auto Cleanup: {os.getenv('CUMULUS_AUTO_CLEANUP', 'true')}")
    print(f"  📊 Job Metadata: {os.getenv('CUMULUS_ENABLE_JOB_METADATA', 'true')}")
    
    # Check if S3 is properly configured
    s3_bucket = os.getenv('CUMULUS_S3_BUCKET')
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if s3_bucket and aws_key and aws_secret:
        print("  ✅ S3 Configuration: Valid")
    else:
        print("  ⚠️  S3 Configuration: Incomplete (will use local-only checkpointing)")


def create_env_file_from_template(template_path: str = "env.example", 
                                 output_path: str = ".env"):
    """
    Create a .env file from the template.
    
    Args:
        template_path: Path to the template file
        output_path: Path for the output .env file
    """
    if os.path.exists(output_path):
        print(f"⚠️  {output_path} already exists. Skipping creation.")
        return
    
    if not os.path.exists(template_path):
        print(f"❌ Template file {template_path} not found.")
        return
    
    # Copy template to .env
    with open(template_path, 'r') as src, open(output_path, 'w') as dst:
        dst.write(src.read())
    
    print(f"✅ Created {output_path} from {template_path}")
    print(f"📝 Please edit {output_path} with your actual values")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_env_file_from_template()
    else:
        setup_distributed_checkpointing_env()
