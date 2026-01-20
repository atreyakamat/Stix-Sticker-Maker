# 🚀 Stix Deployment Guide

This guide describes how to deploy the Stix Sticker Maker platform on your homelab or server using Docker. This setup is production-ready, self-healing (restarts automatically), and uses ephemeral storage (files auto-delete after 1 hour).

## ✨ Features

- **Edge-First AI Processing**: BiRefNet with geometric edge detection for white-on-white stickers
- **Manual Mask Editing**: Erase/Restore brushes for pixel-perfect corrections
- **Re-Background System**: Test stickers on different backgrounds to reveal edge defects
- **Undo/Redo**: Full history for non-destructive editing
- **Canva-Style Borders**: Adjustable thickness and color
- **Batch Processing**: Upload and process multiple stickers at once
- **Auto-Cleanup**: Ephemeral storage - files auto-delete after 1 hour

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

## 📂 Data & Storage

This platform uses **ephemeral storage** by design:

- **Uploads & Outputs**: Stored in system temp directory, automatically deleted after 1 hour
- **AI Models**: Cached in a Docker volume (`stix_model_cache`) to avoid re-downloading

## 🛠 Management Commands

**View Logs:**
```bash
docker-compose logs -f
```

**Stop the Server:**
```bash
docker-compose down
```

**Update the App:**
```bash
docker-compose up -d --build
```

## ⚠️ Troubleshooting

**"Upload failed" or 500 Error**
Check if the AI model is still loading: `docker-compose logs -f backend`

**Port Conflicts**
Edit `docker-compose.yml` to change ports:
```yaml
frontend:
  ports:
    - "8080:80"  # Change to any available port
```

