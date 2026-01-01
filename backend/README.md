# Stix Backend - Sticker Processing Engine

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /api/upload` - Upload single or multiple images
- `GET /api/jobs/{job_id}` - Get job status
- `POST /api/border` - Generate border for processed sticker
- `GET /api/download/{job_id}` - Download processed image
- `POST /api/export/batch` - Export multiple images as ZIP
