import os
import io
import json
import time
import requests
from datetime import datetime

# ─────────────────────────────────────────────
#  STORAGE LAYER
#  Persists builds to Supabase Storage.
#  Survives Render restarts, enables build history,
#  shareable download links.
# ─────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
BUCKET = "builds"

def _headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream",
    }

def _json_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

def is_configured():
    """Check if Supabase is configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)

# ─────────────────────────────────────────────
#  ZIP STORAGE
# ─────────────────────────────────────────────

def upload_zip(project_name: str, zip_bytes: bytes) -> str | None:
    """
    Upload ZIP to Supabase Storage.
    Returns public URL or None on failure.
    """
    if not is_configured():
        print("⚠️  Supabase not configured — skipping ZIP upload")
        return None

    file_path = f"zips/{project_name}.zip"
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{file_path}"

    try:
        # Use upsert header so it works for both new and existing files
        headers = _headers()
        headers["x-upsert"] = "true"
        res = requests.post(
            url,
            headers=headers,
            data=zip_bytes,
            timeout=30
        )
        if res.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{file_path}"
            print(f"  ✅ ZIP uploaded: {public_url}")
            return public_url
        print(f"  ⚠️  ZIP upload failed: {res.status_code} {res.text[:100]}")
        return None
    except Exception as e:
        print(f"  ⚠️  ZIP upload error: {e}")
        return None

def get_zip_url(project_name: str) -> str | None:
    """Get public URL for a project ZIP."""
    if not is_configured():
        return None
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/zips/{project_name}.zip"

# ─────────────────────────────────────────────
#  BUILD METADATA (stored as JSON in Supabase)
# ─────────────────────────────────────────────

def save_build_metadata(build_id: str, metadata: dict) -> bool:
    """
    Save build metadata to Supabase Storage as JSON.
    Used for build history.
    """
    if not is_configured():
        return False

    file_path = f"metadata/{build_id}.json"
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{file_path}"
    json_bytes = json.dumps(metadata, indent=2).encode()

    try:
        # Try POST first, then PUT if exists
        res = requests.post(url, headers=_headers(), data=json_bytes, timeout=10)
        if res.status_code == 409:
            res = requests.put(url, headers=_headers(), data=json_bytes, timeout=10)
        return res.status_code in (200, 201)
    except Exception as e:
        print(f"  ⚠️  Metadata save error: {e}")
        return False

def get_build_metadata(build_id: str) -> dict | None:
    """Retrieve build metadata from Supabase."""
    if not is_configured():
        return None

    file_path = f"metadata/{build_id}.json"
    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{file_path}"

    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception as e:
        print(f"  ⚠️  Metadata fetch error: {e}")
        return None

def list_builds(limit: int = 20) -> list:
    """
    List recent builds from Supabase Storage.
    Returns list of metadata dicts.
    """
    if not is_configured():
        return []

    url = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
    try:
        res = requests.post(
            url,
            headers=_json_headers(),
            json={"prefix": "metadata/", "limit": limit, "sortBy": {"column": "created_at", "order": "desc"}},
            timeout=10
        )
        if res.status_code != 200:
            return []

        files = res.json()
        builds = []
        for f in files:
            name = f.get("name", "")
            if name.endswith(".json"):
                build_id = name.replace("metadata/", "").replace(".json", "")
                meta = get_build_metadata(build_id)
                if meta:
                    builds.append(meta)

        return builds
    except Exception as e:
        print(f"  ⚠️  List builds error: {e}")
        return []

# ─────────────────────────────────────────────
#  HIGH LEVEL: Save a complete build
# ─────────────────────────────────────────────

def save_build(build_id: str, project_name: str, description: str,
               zip_bytes: bytes, files_count: int, failed_count: int) -> dict:
    """
    Save a complete build to Supabase:
    1. Upload the ZIP
    2. Save metadata JSON

    Returns storage info dict.
    """
    print(f"\n💾 Saving build to Supabase...")

    zip_url = upload_zip(project_name, zip_bytes)

    metadata = {
        "build_id": build_id,
        "project_name": project_name,
        "description": description,
        "files_built": files_count,
        "files_failed": failed_count,
        "zip_url": zip_url,
        "created_at": datetime.utcnow().isoformat(),
        "status": "success"
    }

    saved = save_build_metadata(build_id, metadata)
    if saved:
        print(f"  ✅ Build saved — ID: {build_id}")
    else:
        print(f"  ⚠️  Metadata save failed")

    return {
        "zip_url": zip_url,
        "build_id": build_id,
        "persistent": zip_url is not None,
    }