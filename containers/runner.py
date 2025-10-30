#!/usr/bin/env python3
"""
Cumulus Job Runner

This script executes user code and captures results in a standardized format.
It handles errors gracefully and returns results as JSON to stdout.
"""

import json
import sys
import traceback
import time
import os
import importlib.util
from pathlib import Path


def load_user_module():
    """Load the user's main.py module dynamically."""
    main_path = Path("/job/main.py")
    if not main_path.exists():
        raise FileNotFoundError("main.py not found in job directory")
    
    spec = importlib.util.spec_from_file_location("user_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load main.py")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_user_code():
    """Execute the user's code and return the result."""
    try:
        # Load user module
        user_module = load_user_module()
        
        # Look for main function
        if hasattr(user_module, 'main'):
            main_func = getattr(user_module, 'main')
            if callable(main_func):
                return main_func()
            else:
                raise AttributeError("main is not callable")
        else:
            raise AttributeError("No main() function found in main.py")
            
    except Exception as e:
        raise RuntimeError(f"User code execution failed: {str(e)}") from e


def get_system_info():
    """Get system information for debugging."""
    info = {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "job_id": os.environ.get("JOB_ID", "unknown"),
        "chronos_partition": os.environ.get("CHRONOS_PARTITION_ID", "none"),
        "gpu_device": os.environ.get("GPU_DEVICE", "none"),
    }
    
    # Try to get GPU info
    try:
        import subprocess
        result = subprocess.run(["clinfo", "-l"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["opencl_devices"] = result.stdout.strip()
    except:
        info["opencl_devices"] = "unavailable"
    
    return info


def main():
    """Main runner function."""
    start_time = time.time()
    
    try:
        print("🔄 Starting job execution...", file=sys.stderr)
        
        # Execute user code
        result = execute_user_code()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare success response
        response = {
            "status": "success",
            "result": result,
            "execution_time": execution_time,
            "timestamp": int(end_time),
            "system_info": get_system_info()
        }
        
        print("✅ Job completed successfully", file=sys.stderr)
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare error response
        response = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "execution_time": execution_time,
            "timestamp": int(end_time),
            "system_info": get_system_info()
        }
        
        print(f"❌ Job failed: {str(e)}", file=sys.stderr)
    
    # Output result as JSON to stdout
    print(json.dumps(response, indent=2))
    
    # Exit with appropriate code
    if response["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()