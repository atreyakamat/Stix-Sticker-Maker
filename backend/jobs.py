"""
Job Manager
In-memory job queue for tracking processing status
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime
import threading


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Job:
    id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    paths: Dict[str, str] = field(default_factory=dict)
    # Paths will contain: original, mask, transparent, with_border (optional)


class JobManager:
    """Thread-safe job manager"""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
    
    def create_job(self, job_id: str, filename: str) -> Job:
        """Create a new job"""
        with self._lock:
            job = Job(id=job_id, filename=filename)
            self._jobs[job_id] = job
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_status(self, job_id: str, status: JobStatus, progress: int = None):
        """Update job status"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = status
                if progress is not None:
                    self._jobs[job_id].progress = progress
                if status == JobStatus.COMPLETE:
                    self._jobs[job_id].completed_at = datetime.now()
    
    def set_error(self, job_id: str, error: str):
        """Set job error"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.ERROR
                self._jobs[job_id].error = error
    
    def set_paths(self, job_id: str, paths: Dict[str, str]):
        """Set output paths"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].paths = paths
    
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs"""
        with self._lock:
            return list(self._jobs.values())
    
    def get_completed_jobs(self) -> List[Job]:
        """Get all completed jobs"""
        with self._lock:
            return [j for j in self._jobs.values() if j.status == JobStatus.COMPLETE]
    
    def delete_job(self, job_id: str):
        """Delete a job"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]


# Global job manager instance
job_manager = JobManager()
