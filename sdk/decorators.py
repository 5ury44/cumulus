"""
Decorators for the Cumulus SDK
"""

import functools
from typing import Callable, Any, Optional, List
from .client import CumulusClient, CumulusJob


def remote(client: CumulusClient, 
          gpu_memory: float = 0.5,
          duration: int = 3600,
          requirements: Optional[List[str]] = None,
          timeout: Optional[int] = None):
    """
    Decorator to make a function execute remotely on a GPU server.
    
    Args:
        client: CumulusClient instance
        gpu_memory: Fraction of GPU memory to allocate (0.0-1.0)
        duration: Maximum execution time in seconds
        requirements: List of Python packages to install
        timeout: Client-side timeout in seconds
    
    Example:
        @remote(client, gpu_memory=0.8, duration=7200)
        def train_model():
            import torch
            # Your training code here
            return model.state_dict()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return client.run(
                func=func,
                gpu_memory=gpu_memory,
                duration=duration,
                requirements=requirements,
                timeout=timeout,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


def gpu(client: CumulusClient, 
        memory: float = 0.5,
        duration: int = 3600,
        requirements: Optional[List[str]] = None):
    """
    Alias for @remote decorator with GPU-specific naming.
    
    Args:
        client: CumulusClient instance
        memory: Fraction of GPU memory to allocate (0.0-1.0)
        duration: Maximum execution time in seconds
        requirements: List of Python packages to install
    
    Example:
        @gpu(client, memory=0.8, duration=7200)
        def train_model():
            import torch
            # Your training code here
            return model.state_dict()
    """
    return remote(client, gpu_memory=memory, duration=duration, requirements=requirements)


def async_remote(client: CumulusClient,
                gpu_memory: float = 0.5,
                duration: int = 3600,
                requirements: Optional[List[str]] = None):
    """
    Decorator to make a function execute asynchronously on a GPU server.
    
    Args:
        client: CumulusClient instance
        gpu_memory: Fraction of GPU memory to allocate (0.0-1.0)
        duration: Maximum execution time in seconds
        requirements: List of Python packages to install
    
    Returns:
        CumulusJob object that can be monitored and cancelled
    
    Example:
        @async_remote(client, gpu_memory=0.8)
        def train_model():
            import torch
            # Your training code here
            return model.state_dict()
        
        # Usage
        job = train_model()
        print(f"Job status: {job.status()}")
        result = job.result()  # Wait for completion
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Submit job and return job object
            job_id = client._submit_job(
                job_id=str(uuid.uuid4()),
                zip_data=client._package_function(func, requirements or [], **kwargs),
                gpu_memory=gpu_memory,
                duration=duration
            )['job_id']
            
            return CumulusJob(client, job_id)
        return wrapper
    return decorator


class RemoteFunction:
    """
    A callable class that represents a remote function.
    """
    
    def __init__(self, 
                 client: CumulusClient,
                 func: Callable,
                 gpu_memory: float = 0.5,
                 duration: int = 3600,
                 requirements: Optional[List[str]] = None,
                 timeout: Optional[int] = None):
        self.client = client
        self.func = func
        self.gpu_memory = gpu_memory
        self.duration = duration
        self.requirements = requirements
        self.timeout = timeout
    
    def __call__(self, *args, **kwargs):
        """Execute the function remotely."""
        return self.client.run(
            func=self.func,
            gpu_memory=self.gpu_memory,
            duration=self.duration,
            requirements=self.requirements,
            timeout=self.timeout,
            *args,
            **kwargs
        )
    
    def async_call(self, *args, **kwargs) -> CumulusJob:
        """Execute the function asynchronously."""
        job_id = self.client._submit_job(
            job_id=str(uuid.uuid4()),
            zip_data=self.client._package_function(self.func, self.requirements or [], *args, **kwargs),
            gpu_memory=self.gpu_memory,
            duration=self.duration
        )['job_id']
        
        return CumulusJob(self.client, job_id)


def create_remote_function(client: CumulusClient,
                          func: Callable,
                          gpu_memory: float = 0.5,
                          duration: int = 3600,
                          requirements: Optional[List[str]] = None,
                          timeout: Optional[int] = None) -> RemoteFunction:
    """
    Create a remote function object.
    
    Args:
        client: CumulusClient instance
        func: Function to make remote
        gpu_memory: Fraction of GPU memory to allocate (0.0-1.0)
        duration: Maximum execution time in seconds
        requirements: List of Python packages to install
        timeout: Client-side timeout in seconds
    
    Returns:
        RemoteFunction object
    
    Example:
        def train_model():
            import torch
            # Your training code here
            return model.state_dict()
        
        remote_train = create_remote_function(
            client, train_model, gpu_memory=0.8, duration=7200
        )
        
        result = remote_train()  # Execute remotely
    """
    return RemoteFunction(
        client=client,
        func=func,
        gpu_memory=gpu_memory,
        duration=duration,
        requirements=requirements,
        timeout=timeout
    )
