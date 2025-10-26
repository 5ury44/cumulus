"""
ChronosClient - Main client for distributed execution
"""

import requests
import json
import zipfile
import tempfile
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from .code_packager import CodePackager


class CumulusClient:
    """
    Main client for sending code to remote GPU servers with Cumulus partitioning.
    """
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """
        Initialize the Cumulus client.
        
        Args:
            server_url: URL of the remote server (e.g., "http://server:8080")
            api_key: Optional API key for authentication
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def run(self, 
            func: Callable,
            gpu_memory: float = 0.5,
            duration: int = 3600,
            requirements: Optional[List[str]] = None,
            timeout: Optional[int] = None,
            args: Optional[List[Any]] = None,
            **kwargs) -> Any:
        """
        Run a function on the remote GPU server with Cumulus partitioning.
        
        Args:
            func: Function to execute remotely
            gpu_memory: Fraction of GPU memory to allocate (0.0-1.0)
            duration: Maximum execution time in seconds
            requirements: List of Python packages to install
            timeout: Client-side timeout in seconds
            args: List of positional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            Return value of the function
        """
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Package the code
        packager = CodePackager()
        zip_data = packager.package_function(
            func=func,
            requirements=requirements or [],
            job_id=job_id,
            args=args or [],
            **kwargs
        )
        
        # Submit job
        job_info = self._submit_job(
            job_id=job_id,
            zip_data=zip_data,
            gpu_memory=gpu_memory,
            duration=duration
        )
        
        # Wait for completion
        return self._wait_for_completion(job_id, timeout)
    
    def _submit_job(self, job_id: str, zip_data: bytes, gpu_memory: float, duration: int) -> Dict:
        """Submit a job to the remote server."""
        url = f"{self.server_url}/api/jobs"
        
        files = {
            'code': ('code.zip', zip_data, 'application/zip')
        }
        
        data = {
            'job_id': job_id,
            'gpu_memory': gpu_memory,
            'duration': duration
        }
        
        response = self.session.post(url, files=files, data=data)
        response.raise_for_status()
        
        return response.json()
    
    def _wait_for_completion(self, job_id: str, timeout: Optional[int] = None) -> Any:
        """Wait for job completion and return results."""
        url = f"{self.server_url}/api/jobs/{job_id}"
        
        start_time = time.time()
        
        while True:
            response = self.session.get(url)
            response.raise_for_status()
            
            job_status = response.json()
            status = job_status['status']
            
            if status == 'completed':
                # Download results
                return self._download_results(job_id)
            elif status == 'failed':
                error_msg = job_status.get('error', 'Unknown error')
                raise RuntimeError(f"Job failed: {error_msg}")
            elif status == 'timeout':
                raise TimeoutError("Job timed out")
            
            # Check client timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Client timeout exceeded")
            
            # Wait before next check
            time.sleep(1)
    
    def _download_results(self, job_id: str) -> Any:
        """Download and deserialize job results."""
        url = f"{self.server_url}/api/jobs/{job_id}/results"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        # The results are returned as JSON
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get the status of a job."""
        url = f"{self.server_url}/api/jobs/{job_id}"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def list_jobs(self) -> List[Dict]:
        """List all jobs."""
        url = f"{self.server_url}/api/jobs"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        url = f"{self.server_url}/api/jobs/{job_id}/cancel"
        
        response = self.session.post(url)
        response.raise_for_status()
        
        return response.json().get('cancelled', False)
    
    def get_server_info(self) -> Dict:
        """Get information about the remote server."""
        url = f"{self.server_url}/api/info"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def pause_job(self, job_id: str) -> Dict[str, Any]:
        """Pause a running job."""
        response = self.session.post(f"{self.server_url}/api/jobs/{job_id}/pause")
        response.raise_for_status()
        return response.json()

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        """Resume a paused job."""
        response = self.session.post(f"{self.server_url}/api/jobs/{job_id}/resume")
        response.raise_for_status()
        return response.json()

    def get_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """Get available checkpoints for a job."""
        response = self.session.get(f"{self.server_url}/api/jobs/{job_id}/checkpoints")
        response.raise_for_status()
        return response.json()


class CumulusJob:
    """
    Represents an asynchronous job that can be monitored and cancelled.
    """
    
    def __init__(self, client: CumulusClient, job_id: str):
        self.client = client
        self.job_id = job_id
    
    def status(self) -> str:
        """Get current job status."""
        return self.client.get_job_status(self.job_id)['status']
    
    def result(self, timeout: Optional[int] = None) -> Any:
        """Get job result, waiting if necessary."""
        return self.client._wait_for_completion(self.job_id, timeout)
    
    def cancel(self) -> bool:
        """Cancel the job."""
        return self.client.cancel_job(self.job_id)
    
    def pause(self) -> Dict[str, Any]:
        """Pause the job."""
        return self.client.pause_job(self.job_id)
    
    def resume(self) -> Dict[str, Any]:
        """Resume the job."""
        return self.client.resume_job(self.job_id)
    
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Get available checkpoints for this job."""
        return self.client.get_checkpoints(self.job_id)
    
    def __repr__(self) -> str:
        return f"CumulusJob(id={self.job_id}, status={self.status()})"
