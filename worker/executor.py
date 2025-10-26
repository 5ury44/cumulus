"""
CodeExecutor - Handles execution of packaged Python code
"""

import os
import subprocess
import json
import tempfile
import asyncio
from typing import Any, Dict, Optional
import sys


class CodeExecutor:
    """
    Executes packaged Python code in isolated environments.
    """
    
    def __init__(self, python_executable: str = "python3"):
        self.python_executable = python_executable
        self.timeout = 3600  # 1 hour default timeout
    
    async def execute_code(self, job_dir: str, job_id: str, env: dict = None) -> Any:
        """
        Execute packaged code in the specified directory.
        
        Args:
            job_dir: Directory containing the packaged code
            job_id: Unique job identifier
            env: Environment variables to pass to the script
            
        Returns:
            Execution result
        """
        try:
            # Check if requirements.txt exists and install dependencies
            requirements_path = os.path.join(job_dir, "requirements.txt")
            if os.path.exists(requirements_path):
                await self._install_requirements(requirements_path)
            
            # Execute main.py
            main_script = os.path.join(job_dir, "main.py")
            if not os.path.exists(main_script):
                raise FileNotFoundError("main.py not found in job directory")
            
            # Run the script
            result = await self._run_script(main_script, job_dir, env)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {str(e)}")
    
    async def _install_requirements(self, requirements_path: str):
        """Install Python requirements."""
        try:
            # Read requirements
            with open(requirements_path, 'r') as f:
                requirements = f.read().strip()
            
            if not requirements:
                return
            
            # Install requirements
            cmd = [self.python_executable, "-m", "pip", "install", "-r", requirements_path]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Failed to install requirements: {stderr.decode()}")
                
        except Exception as e:
            raise RuntimeError(f"Requirements installation failed: {str(e)}")
    
    async def _run_script(self, script_path: str, working_dir: str, env: dict = None) -> Any:
        """Run a Python script and return its result."""
        try:
            # Set up environment
            if env is None:
                env = os.environ.copy()
            env['PYTHONPATH'] = working_dir
            
            # Run the script
            process = await asyncio.create_subprocess_exec(
                self.python_executable,
                script_path,
                cwd=working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Script execution timed out after {self.timeout} seconds")
            
            # Check for errors
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Script execution failed: {error_msg}")
            
            # Check for result file
            result_path = os.path.join(working_dir, "result.json")
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    return json.load(f)
            else:
                # Return stdout if no result file
                return stdout.decode() if stdout else None
                
        except Exception as e:
            raise RuntimeError(f"Script execution failed: {str(e)}")
    
    def set_timeout(self, timeout: int):
        """Set the execution timeout in seconds."""
        self.timeout = timeout
    
    async def execute_with_cumulus(self, 
                                 job_dir: str, 
                                 job_id: str,
                                 partition_id: str) -> Any:
        """
        Execute code within a Cumulus partition.
        
        Args:
            job_dir: Directory containing the packaged code
            job_id: Unique job identifier
            partition_id: Cumulus partition ID
            
        Returns:
            Execution result
        """
        try:
            # Set environment variables for Cumulus
            env = os.environ.copy()
            env['CUMULUS_PARTITION_ID'] = partition_id
            env['CUMULUS_JOB_ID'] = job_id
            env['CUMULUS_JOB_DIR'] = job_dir  # Add job directory for checkpointing
            
            # Set up S3 distributed checkpointing environment
            self._setup_s3_environment(env)
            
            # Execute with Cumulus context
            return await self.execute_code(job_dir, job_id, env)
            
        except Exception as e:
            raise RuntimeError(f"Cumulus execution failed: {str(e)}")
    
    def _setup_s3_environment(self, env: dict):
        """
        Setup S3 environment variables for distributed checkpointing.
        """
        # Default S3 configuration (can be overridden by environment)
        s3_config = {
            'CUMULUS_S3_BUCKET': 'cumulus-jobs',
            'CUMULUS_S3_REGION': 'us-east-1',
            'AWS_ACCESS_KEY_ID': '',
            'AWS_SECRET_ACCESS_KEY': '',
            'CUMULUS_LOCAL_CACHE_DIR': '/tmp/cumulus/checkpoints',
            'CUMULUS_CACHE_SIZE_LIMIT_GB': '10.0',
            'CUMULUS_KEEP_CHECKPOINTS': '5',
            'CUMULUS_CHECKPOINT_EVERY_STEPS': '100',
            'CUMULUS_CHECKPOINT_EVERY_SECONDS': '300',
            'CUMULUS_AUTO_CLEANUP': 'true',
            'CUMULUS_ENABLE_JOB_METADATA': 'true',
            'CUMULUS_METADATA_TTL_SECONDS': '86400'
        }
        
        # Set environment variables if not already set
        for key, value in s3_config.items():
            if key not in env:
                env[key] = value
        
        print(f"🔧 S3 Environment configured:")
        print(f"  📦 S3 Bucket: {env.get('CUMULUS_S3_BUCKET')}")
        print(f"  🌍 S3 Region: {env.get('CUMULUS_S3_REGION')}")
        print(f"  💾 Local Cache: {env.get('CUMULUS_LOCAL_CACHE_DIR')}")
        print(f"  🔑 AWS Access Key: {'Configured' if env.get('AWS_ACCESS_KEY_ID') else 'Not configured'}")


class IsolatedExecutor(CodeExecutor):
    """
    Executes code in isolated environments using subprocess isolation.
    """
    
    def __init__(self, python_executable: str = "python3"):
        super().__init__(python_executable)
        self.use_venv = True
    
    async def execute_code(self, job_dir: str, job_id: str) -> Any:
        """Execute code in an isolated virtual environment."""
        try:
            # Create isolated environment
            venv_dir = os.path.join(job_dir, "venv")
            await self._create_venv(venv_dir)
            
            # Install requirements in venv
            requirements_path = os.path.join(job_dir, "requirements.txt")
            if os.path.exists(requirements_path):
                await self._install_requirements_venv(requirements_path, venv_dir)
            
            # Execute in venv
            python_exe = os.path.join(venv_dir, "bin", "python")
            if not os.path.exists(python_exe):
                python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
            
            return await self._run_script_venv(python_exe, job_dir)
            
        except Exception as e:
            raise RuntimeError(f"Isolated execution failed: {str(e)}")
    
    async def _create_venv(self, venv_dir: str):
        """Create a virtual environment."""
        cmd = [self.python_executable, "-m", "venv", venv_dir]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to create virtual environment: {stderr.decode()}")
    
    async def _install_requirements_venv(self, requirements_path: str, venv_dir: str):
        """Install requirements in virtual environment."""
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        if not os.path.exists(pip_exe):
            pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        
        cmd = [pip_exe, "install", "-r", requirements_path]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to install requirements in venv: {stderr.decode()}")
    
    async def _run_script_venv(self, python_exe: str, working_dir: str) -> Any:
        """Run script in virtual environment."""
        main_script = os.path.join(working_dir, "main.py")
        
        process = await asyncio.create_subprocess_exec(
            python_exe,
            main_script,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Script execution timed out after {self.timeout} seconds")
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Script execution failed: {error_msg}")
        
        # Check for result file
        result_path = os.path.join(working_dir, "result.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                return json.load(f)
        else:
            return stdout.decode() if stdout else None
