# 🚀 Stix Deployment Guide

This guide describes how to deploy the Stix Sticker Maker platform on your homelab or server using Docker. This setup is production-ready, self-healing (restarts automatically), and persists all your data.

## Prerequisites

- **Docker** and **Docker Compose** installed on your server.
  - Windows/Mac: Install Docker Desktop.
  - Linux: `sudo apt install docker.io docker-compose`

## Quick Start

1. **Transfer the files**: Move this entire project folder to your server.
2. **Navigate to the directory**:
   ```bash
   cd Stix-Sticker-Maker
   ```
3. **Launch**:
   ```bash
   docker-compose up -d --build
   ```

That's it! 
- The **Frontend** will be available at `http://localhost` (or your server's IP).
- The **Backend** API will be at `http://localhost:8000`.

---

## 📂 Data & Persistence

The `docker-compose.yml` is configured to persist your data so you don't lose anything when restarting containers.

- **Images/Stickers**: Mapped to `./backend/uploads` and `./backend/output` on your host machine.
- **AI Models**: The BiRefNet model (~400MB) is cached in a Docker volume named `stix_model_cache`. It will only download once.

## 🛠 Management Commands

**View Logs (to check progress or errors):**
```bash
docker-compose logs -f
```
*Tip: On first launch, watch the backend logs. It will take a minute to download the AI model.*

**Stop the Server:**
```bash
docker-compose down
```

**Update the App (after changing code):**
```bash
docker-compose up -d --build
```
This forces a rebuild of the Docker images with your latest code changes.

## 🔧 Architecture Overview

*   **Service: `frontend`**
    *   Builds the React App into highly optimized static files.
    *   Uses **Nginx** to serve the site and proxy API requests.
    *   Listens on Port `80`.
*   **Service: `backend`**
    *   Runs Python/FastAPI.
    *   Installs system-level dependencies for OpenCV (`libgl1`).
    *   Handles all AI processing and file storage.

## ⚠️ Troubleshooting

**"Upload failed" or 500 Error**
Check if the AI model is still loading. Run `docker-compose logs -f backend`. You should see "✅ Model loaded successfully!" before using the app.

**Port Conflicts**
If Port 80 is already used on your server, edit `docker-compose.yml`:
```yaml
frontend:
  ports:
    - "8080:80"  # Change outer port to 8080 (or any other)
```
Then access at `http://localhost:8080`.
