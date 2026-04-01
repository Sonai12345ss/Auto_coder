import os
import sys
import io
import uuid
import base64
import zipfile
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root to path so we can import agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.planner import generate_blueprint
from agent.builder import build_project
from agent.packager import run_packager

app = FastAPI(title="Auto Coder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectRequest(BaseModel):
    description: str

# In-memory storage
project_store = {}

# ─────────────────────────────────────────────
# BUILD STATUS STORE
# Tracks live progress for each build
# ─────────────────────────────────────────────
build_status_store = {}
# Schema per build_id:
# {
#   "stage": "planning" | "building" | "testing" | "done" | "error",
#   "message": "human readable status",
#   "files": [{"path": "...", "status": "building"|"done"|"failed"}],
#   "total_files": N,
#   "built_count": N,
#   "failed_count": N,
#   "started_at": "ISO timestamp",
#   "project_name": "...",
#   "result": {...} | None   # set when done
# }

def new_build_status(description):
    return {
        "stage": "planning",
        "message": "🧠 Planning your project...",
        "description": description,
        "files": [],
        "total_files": 0,
        "built_count": 0,
        "failed_count": 0,
        "started_at": datetime.utcnow().isoformat(),
        "project_name": None,
        "result": None,
        "error": None,
    }

@app.get("/")
def root():
    return {"status": "Auto Coder API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status/{build_id}")
async def get_status(build_id: str):
    """Poll this endpoint to get live build progress."""
    if build_id not in build_status_store:
        raise HTTPException(status_code=404, detail="Build not found")
    return build_status_store[build_id]

@app.post("/build")
async def build(request: ProjectRequest):
    """
    Starts a build and returns a build_id immediately.
    Frontend polls /status/{build_id} for live progress.
    """
    if not request.description or len(request.description) < 10:
        raise HTTPException(status_code=400, detail="Description too short")

    build_id = str(uuid.uuid4())[:8]
    build_status_store[build_id] = new_build_status(request.description)

    print(f"\n🔨 New build [{build_id}]: {request.description}")

    # Run build in background so we can return build_id immediately
    asyncio.create_task(_run_build(build_id, request.description))

    return {"build_id": build_id, "status": "started"}


async def _run_build(build_id: str, description: str):
    """Background task: runs the full build pipeline and updates status store."""
    status = build_status_store[build_id]

    try:
        # ── STAGE 1: Planning ──
        status["stage"] = "planning"
        status["message"] = "🧠 Planning your project architecture..."

        loop = asyncio.get_event_loop()
        blueprint = await loop.run_in_executor(None, generate_blueprint, description)

        if not blueprint:
            status["stage"] = "error"
            status["error"] = "Failed to generate project blueprint"
            status["message"] = "❌ Planning failed"
            return

        project_name = blueprint["project_name"]
        total = len(blueprint["files"])
        status["project_name"] = project_name
        status["total_files"] = total
        status["message"] = f"📋 Plan ready — building {total} files..."
        status["stage"] = "building"

        # Pre-populate file list so frontend can show them immediately
        status["files"] = [
            {"path": f["path"], "status": "pending"}
            for f in blueprint["files"]
        ]

        # ── STAGE 2: Building (with per-file callback) ──
        def on_file_start(file_path):
            for f in status["files"]:
                if f["path"] == file_path:
                    f["status"] = "building"
            status["message"] = f"⚙️  Building {file_path}..."

        def on_file_done(file_path, success):
            for f in status["files"]:
                if f["path"] == file_path:
                    f["status"] = "done" if success else "failed"
            if success:
                status["built_count"] += 1
            else:
                status["failed_count"] += 1
            done = status["built_count"] + status["failed_count"]
            status["message"] = f"⚙️  Built {done}/{status['total_files']} files..."

        project_path, built_files, failed_files = await loop.run_in_executor(
            None,
            lambda: build_project(blueprint, on_file_start=on_file_start, on_file_done=on_file_done)
        )

        # ── STAGE 3: Testing ──
        status["stage"] = "testing"
        status["message"] = "🧪 Running tests and fixing errors..."

        # ── STAGE 4: Packaging ──
        status["stage"] = "packaging"
        status["message"] = "🐳 Generating Docker + deploy config..."

        # Run packager — adds Dockerfile, docker-compose.yml, Makefile, DEPLOY.md
        existing_files = await loop.run_in_executor(
            None,
            lambda: run_packager(project_path, blueprint, existing_files)
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    file_full_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_full_path, project_path)
                    zipf.write(file_full_path, arcname)
        zip_buffer.seek(0)

        files_content = {}
        for root, dirs, filenames in os.walk(project_path):
            for filename in filenames:
                file_full_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_full_path, project_path)
                try:
                    with open(file_full_path, "r", encoding="utf-8") as f:
                        files_content[arcname] = f.read()
                except:
                    files_content[arcname] = "# Binary or unreadable file"

        zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode()
        project_store[project_name] = {
            "zip_b64": zip_b64,
            "files": files_content,
            "blueprint": blueprint
        }

        # ── STAGE 5: Done ──
        status["stage"] = "done"
        status["message"] = f"✅ Done! {len(built_files)} files built successfully."
        status["result"] = {
            "success": True,
            "project_name": project_name,
            "files_built": len(built_files),
            "files_failed": len(failed_files),
            "failed_files": failed_files,
            "download_url": f"/download/{project_name}",
            "zip_b64": zip_b64,
            "blueprint": blueprint,
        }
        print(f"✅ Build [{build_id}] complete: {project_name}")

    except Exception as e:
        status["stage"] = "error"
        status["error"] = str(e)
        status["message"] = f"❌ Build failed: {str(e)[:100]}"
        print(f"❌ Build [{build_id}] failed: {e}")

@app.get("/files/{project_name}")
async def get_files(project_name: str):
    """Returns all generated file contents for preview."""
    # Try memory first
    if project_name in project_store:
        return {
            "project_name": project_name,
            "files": project_store[project_name]["files"]
        }

    # Fallback to disk
    project_path = f"sandbox/projects/{project_name}"
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")

    files = {}
    for root, dirs, filenames in os.walk(project_path):
        for filename in filenames:
            file_full_path = os.path.join(root, filename)
            arcname = os.path.relpath(file_full_path, project_path)
            try:
                with open(file_full_path, "r", encoding="utf-8") as f:
                    files[arcname] = f.read()
            except:
                files[arcname] = "# Binary or unreadable file"

    return {"project_name": project_name, "files": files}

@app.get("/download/{project_name}")
async def download(project_name: str):
    """Returns the ZIP file for download."""
    # Try memory first
    if project_name in project_store:
        from fastapi.responses import Response
        zip_bytes = base64.b64decode(project_store[project_name]["zip_b64"])
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={project_name}.zip"}
        )

    # Fallback to disk
    zip_path = f"sandbox/projects/{project_name}.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Project not found")

    return FileResponse(
        path=zip_path,
        filename=f"{project_name}.zip",
        media_type="application/zip"
    )