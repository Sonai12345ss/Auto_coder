import os
import sys
import io
import base64
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

# In-memory storage for built projects (survives within same session)
project_store = {}

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

    # Step 3: Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_path):
            for file in files:
                file_full_path = os.path.join(root, file)
                arcname = os.path.relpath(file_full_path, project_path)
                zipf.write(file_full_path, arcname)
    zip_buffer.seek(0)

    # Step 4: Read all file contents into memory
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

    # Step 5: Store in memory
    project_name = blueprint["project_name"]
    project_store[project_name] = {
        "zip_b64": base64.b64encode(zip_buffer.getvalue()).decode(),
        "files": files_content,
        "blueprint": blueprint
    }

    print(f"📦 Project stored in memory: {project_name}")

    return {
        "success": True,
        "project_name": project_name,
        "files_built": len(built_files),
        "files_failed": len(failed_files),
        "failed_files": failed_files,
        "download_url": f"/download/{project_name}",
        "blueprint": blueprint,
        "zip_b64": project_store[project_name]["zip_b64"]
    }

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