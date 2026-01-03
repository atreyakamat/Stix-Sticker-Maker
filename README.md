# Stix - Sticker Background Removal & Generator Platform

A production-ready, Canva-quality sticker creation platform with AI-powered background removal, mask-based editing, border generation, and batch processing.

![Stix](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![React](https://img.shields.io/badge/React-18+-61DAFB)

## ✨ Features

- **🎯 AI Background Removal** - BiRefNet-powered segmentation for highest quality results
- **✂️ Edge Detection** - Handles white-on-white and low-contrast stickers
- **🎨 Canva-Style Borders** - Mask-based border generation with custom colors
- **📦 Batch Processing** - Upload and process multiple stickers at once
- **⚡ Real-time Preview** - Instant feedback on border adjustments
- **📥 Flexible Export** - Individual PNGs or batch ZIP download

## 🏗️ Architecture

```
Stix-Sticker-Maker/
├── backend/              # Python FastAPI server
│   ├── main.py          # Application entry point
│   ├── model_loader.py  # BiRefNet model management
│   ├── pipeline.py      # 5-stage processing pipeline
│   ├── mask_utils.py    # Mask manipulation utilities
│   ├── jobs.py          # Batch job manager
│   └── routes.py        # API endpoints
│
└── frontend/            # React Vite application
    └── src/
        ├── App.jsx      # Main application
        ├── api.js       # API client
        └── components/
            ├── UploadZone.jsx
            ├── Gallery.jsx
            └── Editor.jsx
```

## 🚀 Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start server (model will download on first run)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Then open http://localhost:5173 in your browser.

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload images for processing |
| `/api/jobs/{id}` | GET | Get job status |
| `/api/jobs` | GET | List all jobs |
| `/api/border` | POST | Generate border with color/thickness |
| `/api/download/{id}` | GET | Download processed image |
| `/api/export/batch` | POST | Export multiple as ZIP |

## 🔧 Processing Pipeline

1. **Preprocess** - Fix orientation, resize, normalize colors
2. **Edge Detection** - Canny + contours for sticker boundary
3. **AI Segmentation** - BiRefNet for high-quality mask generation
4. **Mask Refinement** - Morphological cleanup and smoothing
5. **Output** - Transparent PNG with optional border

## 🎨 Border Generation

Borders are generated using mask expansion, not stroke drawing:

1. Original mask is expanded outward by thickness pixels
2. Expanded mask minus original = border region
3. Border region is filled with user-selected color
4. Layers composited: border → sticker

This ensures consistent, print-ready borders regardless of shape complexity.

## ⚙️ Configuration

### Backend
- Output directory: `backend/output/`
- Upload directory: `backend/uploads/`
- Max image size: 2048px (configurable in pipeline.py)

### Frontend
- API URL: `http://localhost:8000` (configurable in api.js)

## 📋 Requirements

### Backend
- Python 3.11+
- 4GB+ RAM (for BiRefNet model)
- GPU optional but recommended

### Frontend
- Node.js 20.19+ or 22.12+
- Modern browser with Canvas support

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - feel free to use for commercial projects.
