"""
FastAPI server for remote code execution with Chronos integration
"""

import os
import tempfile
import zipfile
import json
import uuid
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .executor import CodeExecutor
from .cumulus_manager import CumulusManager


# Pydantic models
class JobSubmission(BaseModel):
    job_id: str
    gpu_memory: float
    duration: int


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, paused, completed, failed, timeout, cancelled
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    partition_id: Optional[str] = None


class ServerInfo(BaseModel):
    server_version: str
    chronos_available: bool
    gpu_devices: list
    active_jobs: int
    max_concurrent_jobs: int


# Global state
jobs: Dict[str, JobStatus] = {}
executor = CodeExecutor()
cumulus_manager = CumulusManager()
max_concurrent_jobs = int(os.getenv('MAX_CONCURRENT_JOBS', '5'))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Cumulus Worker",
        description="Remote code execution with Chronos GPU partitioning",
        version="1.0.0"
    )
    
    @app.get("/api/info")
    async def get_server_info():
        """Get server information and status."""
        try:
            gpu_devices = cumulus_manager.get_available_devices()
            active_jobs = len([job for job in jobs.values() if job.status in ['pending', 'running']])
            
            return ServerInfo(
                server_version="1.0.0",
                chronos_available=cumulus_manager.is_available(),
                gpu_devices=gpu_devices,
                active_jobs=active_jobs,
                max_concurrent_jobs=max_concurrent_jobs
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get server info: {str(e)}")
    
    @app.post("/api/jobs")
    async def submit_job(
        background_tasks: BackgroundTasks,
        code: UploadFile = File(...),
        job_id: str = Form(...),
        gpu_memory: float = Form(...),
        duration: int = Form(...)
    ):
        """Submit a new job for execution."""
        try:
            # Check if job already exists
            if job_id in jobs:
                raise HTTPException(status_code=400, detail="Job ID already exists")
            
            # Check concurrent job limit
            active_jobs = len([job for job in jobs.values() if job.status in ['pending', 'running']])
            if active_jobs >= max_concurrent_jobs:
                raise HTTPException(status_code=429, detail="Too many concurrent jobs")
            
            # Validate parameters
            if not 0.0 <= gpu_memory <= 1.0:
                raise HTTPException(status_code=400, detail="GPU memory must be between 0.0 and 1.0")
            
            if duration <= 0:
                raise HTTPException(status_code=400, detail="Duration must be positive")
            
            # Create job status
            job_status = JobStatus(
                job_id=job_id,
                status="pending",
                created_at=datetime.utcnow()
            )
            jobs[job_id] = job_status
            
            # Read uploaded code
            code_data = await code.read()
            
            # Schedule job execution
            background_tasks.add_task(
                execute_job,
                job_id=job_id,
                code_data=code_data,
                gpu_memory=gpu_memory,
                duration=duration
            )
            
            return {"job_id": job_id, "status": "pending", "message": "Job submitted successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")
    
    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Get the status of a job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return jobs[job_id]
    
    @app.get("/api/jobs/{job_id}/results")
    async def get_job_results(job_id: str):
        """Get the results of a completed job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        if job.status != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")
        
        try:
            # Load results from file
            results_path = f"/tmp/cumulus_jobs/{job_id}/result.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    return json.load(f)
            else:
                raise HTTPException(status_code=404, detail="Results not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")
    
    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        """Cancel a running job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        if job.status not in ["pending", "running"]:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        
        try:
            # Cancel Chronos partition if running
            if job.partition_id:
                cumulus_manager.release_partition(job.partition_id)
            
            # Update job status
            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            
            return {"cancelled": True, "message": "Job cancelled successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")
    
    @app.get("/api/jobs")
    async def list_jobs():
        """List all jobs."""
        return list(jobs.values())
    
    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str):
        """Delete a job and its data."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        try:
            # Clean up job directory
            job_dir = f"/tmp/cumulus_jobs/{job_id}"
            if os.path.exists(job_dir):
                import shutil
                shutil.rmtree(job_dir)
            
            # Remove from jobs dict
            del jobs[job_id]
            
            return {"deleted": True, "message": "Job deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")
    
    @app.post("/api/jobs/{job_id}/pause")
    async def pause_job(job_id: str):
        """Pause a running job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        if job.status != "running":
            raise HTTPException(status_code=400, detail="Job is not running")
        
        try:
            # Write pause signal to control file
            job_dir = f"/tmp/cumulus_jobs/{job_id}"
            control_path = os.path.join(job_dir, "control.json")
            
            with open(control_path, 'w') as f:
                json.dump({"pause": True}, f)
            
            # Update job status
            job.status = "paused"
            
            return {"paused": True, "message": "Job paused successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to pause job: {str(e)}")

    @app.post("/api/jobs/{job_id}/resume")
    async def resume_job(job_id: str):
        """Resume a paused job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        if job.status != "paused":
            raise HTTPException(status_code=400, detail="Job is not paused")
        
        try:
            # Clear pause signal
            job_dir = f"/tmp/cumulus_jobs/{job_id}"
            control_path = os.path.join(job_dir, "control.json")
            
            with open(control_path, 'w') as f:
                json.dump({"pause": False}, f)
            
            # Update job status
            job.status = "running"
            
            return {"resumed": True, "message": "Job resumed successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to resume job: {str(e)}")

    @app.get("/api/jobs/{job_id}/checkpoints")
    async def get_checkpoints(job_id: str):
        """Get available checkpoints for a job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        try:
            job_dir = f"/tmp/cumulus_jobs/{job_id}"
            checkpoints = []
            
            if os.path.exists(job_dir):
                for fname in os.listdir(job_dir):
                    if fname.endswith('.pt'):
                        fpath = os.path.join(job_dir, fname)
                        try:
                            import torch
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list checkpoints: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    return app


async def execute_job(job_id: str, code_data: bytes, gpu_memory: float, duration: int):
    """Execute a job with Chronos GPU partitioning."""
    job = jobs[job_id]
    partition_id = None
    
    try:
        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        
        # Create job directory
        job_dir = f"/tmp/cumulus_jobs/{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        
        # Extract code
        with zipfile.ZipFile(io.BytesIO(code_data), 'r') as zip_file:
            zip_file.extractall(job_dir)
        
        # Create Chronos partition
        partition_id = cumulus_manager.create_partition(
            device=0,  # Use first available GPU
            memory_fraction=gpu_memory,
            duration=duration
        )
        job.partition_id = partition_id
        
        # Execute code
        result = await executor.execute_code(job_dir, job_id)
        
        # Save result
        result_path = os.path.join(job_dir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        
    except Exception as e:
        # Update job status
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        
        # Save error
        error_path = os.path.join(job_dir, "error.json")
        with open(error_path, 'w') as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
        
    finally:
        # Release Chronos partition
        if partition_id:
            try:
                cumulus_manager.release_partition(partition_id)
            except Exception as e:
                print(f"Warning: Failed to release partition {partition_id}: {e}")


# Import required modules
import io
import traceback


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
