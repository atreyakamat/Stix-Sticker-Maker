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
from config import UPLOAD_DIR, OUTPUT_DIR


# Request/Response Models
class JobResponse(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    error: Optional[str] = None
    paths: dict = {}
    # Validation metadata
    edge_confidence: Optional[str] = None
    warnings: List[str] = []
    needs_review: bool = False


class BorderRequest(BaseModel):
    job_id: str
    thickness: int = 10
    color: str = "#FFFFFF"


class BatchExportRequest(BaseModel):
    job_ids: List[str]
    include_border: bool = True


# Helper Functions
def process_image_task(job_id: str, image_path: Path):
    """Background task to process an image using EDGE-FIRST pipeline"""
    try:
        job_manager.update_status(job_id, JobStatus.PROCESSING, 10)
        
        # Load image
        image = Image.open(image_path)
        job_manager.update_status(job_id, JobStatus.PROCESSING, 20)
        
        # Process through EDGE-FIRST pipeline
        result = process_sticker(image)
        job_manager.update_status(job_id, JobStatus.PROCESSING, 80)
        
        # Save results
        paths = save_results(result, OUTPUT_DIR, job_id)
        job_manager.update_status(job_id, JobStatus.PROCESSING, 95)
        
        # Convert paths to relative URLs
        url_paths = {
            key: f"/output/{Path(path).name}"
            for key, path in paths.items()
        }
        
        # Store validation metadata in job
        job = job_manager.get_job(job_id)
        if job:
            job.edge_confidence = result.edge_confidence.value
            job.warnings = result.warnings
            job.needs_review = (
                result.edge_confidence.value in ['low', 'failed'] or
                not result.contour_closed or
                result.ai_escaped_contour
            )
        
        job_manager.set_paths(job_id, url_paths)
        job_manager.update_status(job_id, JobStatus.COMPLETE, 100)
        
        # Log validation info
        print(f"✅ Processed {job_id}: edge_confidence={result.edge_confidence.value}")
        if result.warnings:
            for w in result.warnings:
                print(f"   ⚠️ {w}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        paths=job.paths,
        edge_confidence=getattr(job, 'edge_confidence', None),
        warnings=getattr(job, 'warnings', []),
        needs_review=getattr(job, 'needs_review', False)
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
            paths=job.paths,
            edge_confidence=getattr(job, 'edge_confidence', None),
            warnings=getattr(job, 'warnings', []),
            needs_review=getattr(job, 'needs_review', False)
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
    for key in ["original", "mask", "transparent", "with_border", "alpha", "edited_mask"]:
        file_path = OUTPUT_DIR / f"{job_id}_{key}.png"
        if file_path.exists():
            file_path.unlink()
    
    # Delete upload
    for f in UPLOAD_DIR.glob(f"{job_id}_*"):
        f.unlink()
    
    # Remove from job manager
    job_manager.delete_job(job_id)
    
    return {"success": True, "deleted": job_id}


@router.post("/reanalyze/{job_id}")
async def reanalyze_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Re-run the AI processing pipeline on an existing job.
    Useful when the initial extraction wasn't perfect.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Find the original upload file
    upload_files = list(UPLOAD_DIR.glob(f"{job_id}_*"))
    if not upload_files:
        # Try to use the saved original from output
        original_path = OUTPUT_DIR / f"{job_id}_original.png"
        if not original_path.exists():
            raise HTTPException(status_code=404, detail="Original image not found")
        upload_path = original_path
    else:
        upload_path = upload_files[0]
    
    # Delete old processed files (keep original)
    for key in ["mask", "transparent", "with_border", "alpha", "edited_mask", "preview"]:
        file_path = OUTPUT_DIR / f"{job_id}_{key}.png"
        if file_path.exists():
            file_path.unlink()
    
    # Reset job status
    job_manager.update_status(job_id, JobStatus.PENDING, 0)
    job.paths = {}
    job.warnings = []
    job.edge_confidence = None
    job.needs_review = False
    
    # Queue reprocessing
    background_tasks.add_task(process_image_task, job_id, upload_path)
    
    return {
        "success": True,
        "message": "Reprocessing started",
        "job_id": job_id
    }


# =============================================================================
# MANUAL MASK EDITING ENDPOINTS
# =============================================================================

class MaskEditRequest(BaseModel):
    job_id: str
    mask_data: str  # Base64 encoded PNG mask data


@router.post("/mask/save")
async def save_edited_mask(request: MaskEditRequest):
    """
    Save an edited mask from the frontend canvas.
    The mask is the source of truth - this saves the user's manual corrections.
    """
    import base64
    import numpy as np
    from mask_utils import apply_mask
    
    job = job_manager.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # Decode base64 mask data
        # Remove data URL prefix if present
        mask_data = request.mask_data
        if ',' in mask_data:
            mask_data = mask_data.split(',')[1]
        
        mask_bytes = base64.b64decode(mask_data)
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert('L')
        mask_array = np.array(mask_image)
        
        # Save edited mask
        edited_mask_path = OUTPUT_DIR / f"{request.job_id}_edited_mask.png"
        mask_image.save(edited_mask_path, "PNG")
        
        # Re-apply mask to original to create new transparent output
        original_path = OUTPUT_DIR / f"{request.job_id}_original.png"
        if original_path.exists():
            original = Image.open(original_path)
            transparent = apply_mask(original, mask_array)
            
            # Save new transparent
            transparent_path = OUTPUT_DIR / f"{request.job_id}_transparent.png"
            transparent.save(transparent_path, "PNG")
        
        # Update job paths
        job.paths["edited_mask"] = f"/output/{request.job_id}_edited_mask.png"
        job.paths["transparent"] = f"/output/{request.job_id}_transparent.png"
        
        return {
            "success": True,
            "message": "Mask saved and sticker regenerated",
            "paths": job.paths
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save mask: {str(e)}")


@router.get("/mask/{job_id}")
async def get_mask(job_id: str):
    """Get the current mask for a job (edited if available, otherwise original)"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Prefer edited mask if it exists
    edited_mask_path = OUTPUT_DIR / f"{job_id}_edited_mask.png"
    if edited_mask_path.exists():
        return FileResponse(edited_mask_path, media_type="image/png")
    
    # Fall back to original mask
    mask_path = OUTPUT_DIR / f"{job_id}_mask.png"
    if mask_path.exists():
        return FileResponse(mask_path, media_type="image/png")
    
    raise HTTPException(status_code=404, detail="Mask not found")


# =============================================================================
# RE-BACKGROUND SYSTEM ENDPOINTS
# =============================================================================

class RebackgroundRequest(BaseModel):
    job_id: str
    background_type: str  # "solid", "gradient", "checker", "noise"
    color1: str = "#ffffff"
    color2: str = "#000000"  # For gradients
    gradient_direction: str = "horizontal"  # "horizontal", "vertical", "diagonal"


@router.post("/rebackground")
async def generate_rebackground(request: RebackgroundRequest):
    """
    Generate a sticker preview with a custom background.
    This is for verification/QA - reveals edge defects on different backgrounds.
    Does NOT alter the mask or sticker.
    """
    import numpy as np
    
    job = job_manager.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # Load transparent sticker
        transparent_path = OUTPUT_DIR / f"{request.job_id}_transparent.png"
        if not transparent_path.exists():
            raise HTTPException(status_code=404, detail="Transparent sticker not found")
        
        sticker = Image.open(transparent_path).convert('RGBA')
        width, height = sticker.size
        
        # Generate background based on type
        if request.background_type == "solid":
            bg = Image.new('RGBA', (width, height), hex_to_rgb(request.color1) + (255,))
            
        elif request.background_type == "checker":
            # Checkerboard pattern for transparency verification
            bg = Image.new('RGBA', (width, height), (255, 255, 255, 255))
            bg_array = np.array(bg)
            checker_size = 20
            for y in range(height):
                for x in range(width):
                    if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                        bg_array[y, x] = [200, 200, 200, 255]
            bg = Image.fromarray(bg_array)
            
        elif request.background_type == "gradient":
            bg = Image.new('RGBA', (width, height))
            bg_array = np.zeros((height, width, 4), dtype=np.uint8)
            
            c1 = hex_to_rgb(request.color1)
            c2 = hex_to_rgb(request.color2)
            
            for i in range(height if request.gradient_direction == "vertical" else width):
                if request.gradient_direction == "vertical":
                    t = i / height
                else:
                    t = i / width
                    
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
                
                if request.gradient_direction == "vertical":
                    bg_array[i, :] = [r, g, b, 255]
                else:
                    bg_array[:, i] = [r, g, b, 255]
                    
            bg = Image.fromarray(bg_array)
            
        elif request.background_type == "noise":
            # Random noise for stress testing edges
            noise = np.random.randint(150, 255, (height, width, 3), dtype=np.uint8)
            alpha = np.full((height, width, 1), 255, dtype=np.uint8)
            bg_array = np.concatenate([noise, alpha], axis=2)
            bg = Image.fromarray(bg_array)
            
        elif request.background_type == "contrast_white":
            bg = Image.new('RGBA', (width, height), (255, 255, 255, 255))
            
        elif request.background_type == "contrast_black":
            bg = Image.new('RGBA', (width, height), (0, 0, 0, 255))
            
        else:
            # Default to white
            bg = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        
        # Composite sticker over background
        result = Image.alpha_composite(bg, sticker)
        
        # Save preview
        preview_path = OUTPUT_DIR / f"{request.job_id}_preview.png"
        result.save(preview_path, "PNG")
        
        # Update job paths
        job.paths["preview"] = f"/output/{request.job_id}_preview.png"
        
        return {
            "success": True,
            "path": f"/output/{request.job_id}_preview.png",
            "background_type": request.background_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")


@router.get("/rebackground/presets")
async def get_background_presets():
    """Get available background presets for the UI"""
    return {
        "presets": [
            {"id": "transparent", "name": "Transparent", "type": "checker"},
            {"id": "white", "name": "White", "type": "solid", "color1": "#ffffff"},
            {"id": "black", "name": "Black", "type": "solid", "color1": "#000000"},
            {"id": "gray", "name": "Gray", "type": "solid", "color1": "#808080"},
            {"id": "gradient_bw", "name": "B/W Gradient", "type": "gradient", "color1": "#000000", "color2": "#ffffff"},
            {"id": "gradient_sunset", "name": "Sunset", "type": "gradient", "color1": "#ff6b6b", "color2": "#feca57"},
            {"id": "gradient_ocean", "name": "Ocean", "type": "gradient", "color1": "#0052d4", "color2": "#65c7f7"},
            {"id": "noise", "name": "Noise Test", "type": "noise"},
        ]
    }
