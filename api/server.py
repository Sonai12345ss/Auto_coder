import os
import sys
import json
import shutil
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root to path so we can import agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.planner import generate_blueprint
from agent.builder import build_project

app = FastAPI(title="Auto Coder API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectRequest(BaseModel):
    description: str

@app.get("/")
def root():
    return {"status": "Auto Coder API is running"}

@app.post("/build")
async def build(request: ProjectRequest):
    """Takes a project description and builds the full project."""

    if not request.description or len(request.description) < 10:
        raise HTTPException(status_code=400, detail="Description too short")

    print(f"\n🔨 New build request: {request.description}")

    # Step 1: Generate blueprint
    blueprint = generate_blueprint(request.description)
    if not blueprint:
        raise HTTPException(status_code=500, detail="Failed to generate project blueprint")

    # Step 2: Build project files
    project_path, built_files, failed_files = build_project(blueprint)

    # Step 3: Create ZIP
    zip_path = f"{project_path}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_path):
            for file in files:
                file_full_path = os.path.join(root, file)
                arcname = os.path.relpath(file_full_path, project_path)
                zipf.write(file_full_path, arcname)

    print(f"📦 ZIP created: {zip_path}")

    return {
        "success": True,
        "project_name": blueprint["project_name"],
        "files_built": len(built_files),
        "files_failed": len(failed_files),
        "failed_files": failed_files,
        "download_url": f"/download/{blueprint['project_name']}",
        "blueprint": blueprint
    }

@app.get("/files/{project_name}")
async def get_files(project_name: str):
    """Returns all generated file contents for preview."""
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