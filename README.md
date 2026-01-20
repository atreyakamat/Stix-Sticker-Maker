# 🎨 Stix - AI Sticker Maker

A professional-grade sticker background removal and editing platform. Extract stickers from any image with AI precision, manually refine edges, and export production-ready transparent PNGs.

![Stix](https://img.shields.io/badge/Version-1.0-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![React](https://img.shields.io/badge/React-18+-purple)

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Edge-First AI** | BiRefNet model with geometric edge detection for white-on-white stickers |
| **Manual Editing** | Brush tools to erase/restore mask areas with full undo/redo |
| **Re-Background** | Test sticker edges on different backgrounds to spot defects |
| **Canva-Style Borders** | Add customizable outlines with smooth, rounded edges |
| **Batch Processing** | Upload and process multiple images at once |
| **Ephemeral Storage** | Auto-cleanup keeps your system clean (1-hour retention) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- ~500MB disk space (for AI model)

### 1. Start the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000
```

> ⏳ First run downloads the AI model (~400MB). Watch the terminal for "✅ Model loaded successfully!"

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Open the App

Navigate to **http://localhost:5173** in your browser.

---

## 🖼️ How to Use

### Step 1: Upload
Drag and drop sticker images (PNG, JPG, WEBP) onto the upload zone.

### Step 2: Wait for Processing
The AI will automatically:
1. Detect sticker edges (works even on white backgrounds)
2. Generate a precise mask
3. Create a transparent PNG

### Step 3: Refine (Optional)
Click on any processed sticker to open the editor where you can:
- **Erase** (🧹) - Remove background areas the AI missed
- **Restore** (✨) - Bring back parts that were accidentally removed
- **Test Backgrounds** - Preview on white, black, gradients to spot edge issues

### Step 4: Export
- Add a border if desired
- Download individual PNGs or batch export as ZIP

---

## 📁 Project Structure

```
Stix-Sticker-Maker/
├── backend/                 # Python FastAPI server
│   ├── main.py             # App entry point
│   ├── pipeline.py         # 6-stage AI processing
│   ├── routes.py           # API endpoints
│   ├── mask_utils.py       # Mask manipulation functions
│   ├── model_loader.py     # BiRefNet model loading
│   ├── jobs.py             # Job queue management
│   └── config.py           # Storage & cleanup settings
│
├── frontend/               # React + Vite UI
│   └── src/
│       ├── App.jsx         # Main application
│       ├── api.js          # Backend API client
│       ├── index.css       # Design system
│       └── components/
│           ├── UploadZone.jsx
│           ├── Gallery.jsx
│           └── Editor.jsx
│
├── docker-compose.yml      # Docker deployment
└── DEPLOYMENT.md           # Server deployment guide
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload images for processing |
| `/api/jobs` | GET | List all processing jobs |
| `/api/jobs/{id}` | GET | Get job status and paths |
| `/api/border` | POST | Generate border for a sticker |
| `/api/mask/save` | POST | Save edited mask |
| `/api/rebackground` | POST | Generate background preview |
| `/api/download/{id}` | GET | Download processed sticker |

---

## 🐳 Docker Deployment

For production deployment on a homelab or server:

```bash
docker-compose up -d --build
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for full instructions.

---

## 🛠️ Configuration

Edit `backend/config.py` to customize:

```python
FILE_RETENTION_SECONDS = 3600   # How long to keep files (default: 1 hour)
CLEANUP_INTERVAL_SECONDS = 600  # Cleanup check interval (default: 10 min)
```

---

## 📝 License

MIT License - Use freely for personal or commercial projects.
