"""
API Routes
Endpoints for upload, processing, border generation, and export
"""

import uuid
import asyncio
import zipfile
import io
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image

from jobs import job_manager, JobStatus
from pipeline import process_sticker, save_results
from mask_utils import composite_with_border, hex_to_rgb

router = APIRouter()

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"


# Request/Response Models
class JobResponse(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    error: Optional[str] = None
    paths: dict = {}


class BorderRequest(BaseModel):
    job_id: str
    thickness: int = 10
    color: str = "#FFFFFF"


class BatchExportRequest(BaseModel):
    job_ids: List[str]
    include_border: bool = True


# Helper Functions
def process_image_task(job_id: str, image_path: Path):
    """Background task to process an image"""
    try:
        job_manager.update_status(job_id, JobStatus.PROCESSING, 10)
        
        # Load image
        image = Image.open(image_path)
        job_manager.update_status(job_id, JobStatus.PROCESSING, 20)
        
        # Process through pipeline
        results = process_sticker(image, use_edge_detection=True)
        job_manager.update_status(job_id, JobStatus.PROCESSING, 80)
        
        # Save results
        paths = save_results(results, OUTPUT_DIR, job_id)
        job_manager.update_status(job_id, JobStatus.PROCESSING, 95)
        
        # Convert paths to relative URLs
        url_paths = {
            key: f"/output/{Path(path).name}"
            for key, path in paths.items()
        }
        
        job_manager.set_paths(job_id, url_paths)
        job_manager.update_status(job_id, JobStatus.COMPLETE, 100)
        
    except Exception as e:
        job_manager.set_error(job_id, str(e))


# Routes
@router.post("/upload", response_model=List[JobResponse])
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload one or more images for processing
    Returns list of job IDs for tracking
    """
    jobs = []
    
    for file in files:
        # Validate file type (flexible check)
        filename = file.filename.lower()
        valid_ext = ('.png', '.jpg', '.jpeg', '.webp', '.jfif', '.heic')
        
        is_image = (file.content_type and file.content_type.startswith('image/')) or \
                   filename.endswith(valid_ext)
                   
        if not is_image:
            print(f"⚠️ Skipping non-image file: {file.filename}")
            continue
        
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Save upload
        upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        content = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(content)
        
        # Create job
        job = job_manager.create_job(job_id, file.filename)
        
        # Queue processing
        background_tasks.add_task(process_image_task, job_id, upload_path)
        
        jobs.append(JobResponse(
            id=job.id,
            filename=job.filename,
            status=job.status.value,
            progress=job.progress
        ))
    
    return jobs


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        id=job.id,
        filename=job.filename,
        status=job.status.value,
        progress=job.progress,
        error=job.error,
        paths=job.paths
    )


@router.get("/jobs", response_model=List[JobResponse])
async def get_all_jobs():
    """Get all jobs"""
    jobs = job_manager.get_all_jobs()
    return [
        JobResponse(
            id=job.id,
            filename=job.filename,
            status=job.status.value,
            progress=job.progress,
            error=job.error,
            paths=job.paths
        )
        for job in jobs
    ]


@router.post("/border")
async def generate_border(request: BorderRequest):
    """
    Generate a border for a processed sticker
    Uses mask-based expansion for Canva-quality results
    """
    job = job_manager.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Job not complete")
    
    try:
        # Load transparent image and mask
        transparent_path = OUTPUT_DIR / f"{request.job_id}_transparent.png"
        mask_path = OUTPUT_DIR / f"{request.job_id}_mask.png"
        
        sticker = Image.open(transparent_path)
        mask = Image.open(mask_path)
        import numpy as np
        mask_array = np.array(mask)
        
        # Generate border
        color = hex_to_rgb(request.color)
        result = composite_with_border(
            sticker,
            mask_array,
            request.thickness,
            color
        )
        
        # Save result
        border_path = OUTPUT_DIR / f"{request.job_id}_with_border.png"
        result.save(border_path, "PNG")
        
        # Update job paths
        url_path = f"/output/{request.job_id}_with_border.png"
        job.paths["with_border"] = url_path
        
        return {
            "success": True,
            "path": url_path,
            "thickness": request.thickness,
            "color": request.color
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{job_id}")
async def download_image(job_id: str, with_border: bool = False):
    """Download a processed image"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Job not complete")
    
    # Choose file
    if with_border and "with_border" in job.paths:
        filename = f"{job_id}_with_border.png"
    else:
        filename = f"{job_id}_transparent.png"
    
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=f"stix_{job.filename.rsplit('.', 1)[0]}.png",
        media_type="image/png"
    )


@router.post("/export/batch")
async def export_batch(request: BatchExportRequest):
    """Export multiple processed images as a ZIP file"""
    # Validate jobs
    valid_jobs = []
    for job_id in request.job_ids:
        job = job_manager.get_job(job_id)
        if job and job.status == JobStatus.COMPLETE:
            valid_jobs.append(job)
    
    if not valid_jobs:
        raise HTTPException(status_code=400, detail="No valid completed jobs")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for job in valid_jobs:
            # Choose file
            if request.include_border and "with_border" in job.paths:
                filename = f"{job.id}_with_border.png"
            else:
                filename = f"{job.id}_transparent.png"
            
            file_path = OUTPUT_DIR / filename
            if file_path.exists():
                # Use original filename in ZIP
                output_name = f"stix_{job.filename.rsplit('.', 1)[0]}.png"
                zip_file.write(file_path, output_name)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=stix_batch.zip"}
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    for key in ["original", "mask", "transparent", "with_border"]:
        file_path = OUTPUT_DIR / f"{job_id}_{key}.png"
        if file_path.exists():
            file_path.unlink()
    
    # Delete upload
    for f in UPLOAD_DIR.glob(f"{job_id}_*"):
        f.unlink()
    
    # Remove from job manager
    job_manager.delete_job(job_id)
    
    return {"success": True, "deleted": job_id}
