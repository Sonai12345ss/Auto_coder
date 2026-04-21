import os
import re
import json
import time
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# PROVIDER SETUP
# 20 providers — Gemini Pro first (best at JSON structure),
# then Flash, then Groq/OpenRouter as fallbacks.
# ═══════════════════════════════════════════════════════════════

groq1      = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2      = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3      = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",  api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1",  api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))


def call_llm(messages):
    providers = [
        ("Gemini-1 / gemini-2.5-pro",   lambda: gemini1.chat.completions.create(model="gemini-2.5-pro",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-2 / gemini-2.5-pro",   lambda: gemini2.chat.completions.create(model="gemini-2.5-pro",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-3 / gemini-2.5-pro",   lambda: gemini3.chat.completions.create(model="gemini-2.5-pro",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-1 / gemini-2.5-flash", lambda: gemini1.chat.completions.create(model="gemini-2.5-flash",  messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-2 / gemini-2.5-flash", lambda: gemini2.chat.completions.create(model="gemini-2.5-flash",  messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-3 / gemini-2.5-flash", lambda: gemini3.chat.completions.create(model="gemini-2.5-flash",  messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-1 / llama-3.3-70b",      lambda: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-2 / llama-3.3-70b",      lambda: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-3 / llama-3.3-70b",      lambda: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-1 / llama-3.1-8b",       lambda: groq1.chat.completions.create(model="llama-3.1-8b-instant",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-2 / llama-3.1-8b",       lambda: groq2.chat.completions.create(model="llama-3.1-8b-instant",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-3 / llama-3.1-8b",       lambda: groq3.chat.completions.create(model="llama-3.1-8b-instant",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-1 / gemini-2.0-flash",  lambda: gemini1.chat.completions.create(model="gemini-2.0-flash",  messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-2 / gemini-2.0-flash",  lambda: gemini2.chat.completions.create(model="gemini-2.0-flash",  messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-3 / gemini-2.0-flash",  lambda: gemini3.chat.completions.create(model="gemini-2.0-flash",  messages=messages, temperature=0.2, max_tokens=4096)),
        ("OpenRouter / llama-3.3-70b",  lambda: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=messages, temperature=0.2, max_tokens=4096)),
        ("OpenRouter / gemma-3-27b",    lambda: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free",              messages=messages, temperature=0.2, max_tokens=4096)),
        ("OpenRouter / gemma-3-12b",    lambda: openrouter.chat.completions.create(model="google/gemma-3-12b-it:free",              messages=messages, temperature=0.2, max_tokens=4096)),
        ("Doubleword / Qwen3.5-35B",    lambda: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",               messages=messages, temperature=0.2, max_tokens=4096)),
        ("Doubleword / Qwen3.5-397B",   lambda: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8",             messages=messages, temperature=0.2, max_tokens=4096)),
    ]
    last_error = None
    for attempt, (name, fn) in enumerate(providers):
        try:
            print(f"  🤖 Planner using {name}...")
            return fn()
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "402", "503",
                                       "quota", "temporarily", "overloaded", "upstream"]):
                wait = min(2 ** (attempt % 4), 16)
                print(f"  ⚠️  {name} rate limited, waiting {wait}s...")
                last_error = e
                time.sleep(wait)
                continue
            elif any(x in err for x in ["decommission", "deprecated", "no longer supported",
                                         "invalid model", "404"]):
                print(f"  ⚠️  {name} model unavailable, skipping...")
                last_error = e
                continue
            last_error = e
            print(f"  ⚠️  {name} failed: {str(e)[:80]}, trying next...")
            continue
    raise Exception(f"All planner providers failed. Last error: {last_error}")


# ═══════════════════════════════════════════════════════════════
# GHOST COMPONENT FILTERING
# ═══════════════════════════════════════════════════════════════

# BUG-4 FIX: Words that appear in route descriptions but are NOT real page
# components. The auto-add logic regex-extracts PascalCase words from the
# App.js description — these must be excluded before they generate ghost files.
#
# EXCLUDED is used by the auto-add logic (words that match PascalCase in
# descriptions but should never become component files).
EXCLUDED_COMPONENT_NAMES = {
    # React Router internals
    "React", "Route", "Routes", "BrowserRouter", "Navigate", "Outlet",
    "Link", "NavLink", "Switch", "RouterProvider", "Redirect",
    # Structural / layout
    "App", "Router", "Routing", "Component", "Fragment",
    "Provider", "Context", "Suspense", "Layout", "Footer", "Navbar",
    # Auth guard DUPLICATES — PrivateRoute.js is the one real auth guard,
    # these are incorrect alternatives the LLM sometimes generates instead.
    # PrivateRoute is intentionally absent from this set — it is a real
    # canonical template file that must always be generated (in REQUIRED_FILES).
    "Protected", "AuthRoute", "GuardedRoute",
    # ── BUG-4 additions ──────────────────────────────────────────────────────
    # These words appear in descriptions like "authentication required",
    # "auth guard", "loading state", "error boundary", etc.
    # Without this exclusion, the auto-add loop generates Required.js,
    # Auth.js, Guard.js, Loading.js — ghost files that break cross-file checks.
    "Required", "Auth", "Guard", "Private",
    "Loading", "Error", "Modal", "Alert", "Toast",
    "Base", "Common", "Utils", "Helper", "Shared", "Wrapper",
    "Index", "NotFound", "Redirect", "Fallback",
}

# GHOST_FILES: hardcoded paths that should never exist regardless of project type.
# These are files the LLM sometimes generates for routing/auth concepts that are
# already handled by App.js (routing) or PrivateRoute.js (auth guarding).
GHOST_FILE_PATHS = {
    # Routing wrapper files
    "frontend/src/components/Routing.js",
    "frontend/src/components/Router.js",
    "frontend/src/components/Routes.js",
    "frontend/src/Routing.js",
    "frontend/src/Router.js",
    # Auth guard duplicates (PrivateRoute.js is the canonical guard)
    "frontend/src/components/Protected.js",
    "frontend/src/components/AuthRoute.js",
    "frontend/src/components/GuardedRoute.js",
    # ── BUG-4 additions: ghost utility/guard component files ────────────────
    "frontend/src/components/Required.js",
    "frontend/src/components/Auth.js",
    "frontend/src/components/Guard.js",
    "frontend/src/components/Private.js",
    "frontend/src/components/Loading.js",
    "frontend/src/components/Error.js",
    "frontend/src/components/Modal.js",
    "frontend/src/components/Wrapper.js",
    "frontend/src/components/Base.js",
    "frontend/src/components/Common.js",
    "frontend/src/components/NotFound.js",
    "frontend/src/components/Fallback.js",
    "frontend/src/components/Index.js",
}


def filter_ghost_components(files_list):
    """
    Remove ghost component files from the blueprint files list.

    Two-pass filter:
    1. Remove any file whose path is in GHOST_FILE_PATHS (hardcoded known ghosts)
    2. Remove any components/ file whose base name is in EXCLUDED_COMPONENT_NAMES
       AND is thin (< 50 lines when eventually built) — but we can't check that
       at planning time, so we check the base name only.

    Called on blueprint["files"] before returning from generate_blueprint().
    """
    filtered = []
    removed = []

    for file_info in files_list:
        path = file_info.get("path", "")

        # Pass 1: hardcoded ghost paths
        if path in GHOST_FILE_PATHS:
            removed.append(path)
            continue

        # Pass 2: component base name in exclusion set
        if path.startswith("frontend/src/components/") and path.endswith(".js"):
            component_name = os.path.basename(path).replace(".js", "")
            if component_name in EXCLUDED_COMPONENT_NAMES:
                removed.append(path)
                continue

        filtered.append(file_info)

    if removed:
        print(f"  🧹 Filtered {len(removed)} ghost file(s): {[os.path.basename(p) for p in removed]}")

    return filtered


# ═══════════════════════════════════════════════════════════════
# PLANNER PROMPT
# ═══════════════════════════════════════════════════════════════

PLANNER_PROMPT = """
You are a senior software architect. Generate a complete project blueprint as JSON.

ABSOLUTE RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no backticks.
2. Every file imported by another file MUST be in the files array.
3. App.js imports components — every component it imports MUST have its own file entry.
4. NEVER import a component in App.js that is not in the files list.
5. NEVER generate utility/guard component files — see forbidden list below.

FORBIDDEN FILES — never generate these, they do not exist:
❌ Routing.js, Router.js, Routes.js (routing lives inside App.js)
❌ Required.js, Auth.js, Guard.js, Private.js (auth guard = PrivateRoute.js only)
❌ Loading.js, Error.js, Modal.js, Wrapper.js, NotFound.js (not real page components)
❌ Any file whose purpose is "guard", "wrapper", "utility", "helper", or "auth check"

BACKEND FILES — always include all of these:
- backend/__init__.py — ONLY: db = SQLAlchemy() and jwt = JWTManager()
- backend/config.py — DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY, DEBUG from env
- backend/models.py — SQLAlchemy models: User (with set_password/check_password), plus project models
- backend/routes.py — ALL endpoints: /api/register, /api/login, /api/user, plus resource endpoints
- backend/app.py — Flask factory: create_app(), init extensions, register blueprint, /health route

FRONTEND FILES — always include all of these:
- frontend/package.json — react, react-dom, react-scripts, axios, react-router-dom
- frontend/public/index.html — React HTML template with Tailwind CDN script tag
- frontend/src/index.js — ReactDOM.createRoot (React 18 API only)
- frontend/src/index.css — complete stylesheet: variables, navbar, cards, buttons, forms, spinner
- frontend/src/App.js — BrowserRouter + Routes + Route for every page. PURE ROUTER: no hooks, no API calls
- frontend/src/api.js — axios instance, request interceptor (JWT), response interceptor (401→redirect), one export per endpoint
- frontend/src/components/PrivateRoute.js — ALWAYS include: checks localStorage token, renders Outlet or redirects
- frontend/src/components/Navbar.js — sticky nav with auth-aware links
- frontend/src/components/Home.js — landing page with hero section
- frontend/src/components/Login.js — login form: username + password fields only
- frontend/src/components/Register.js — register form: username, email, password
- .env.example — DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY, DEBUG, REACT_APP_API_URL

RESOURCE COMPONENTS — for each main data entity in the project:
- [Resource]List.js — paginated list with loading spinner, error state, empty state
- [Resource]Form.js — create/edit form with validation and error display
- [Resource]Detail.js — single item detail view (only if the project needs it)

BACKEND RULES:
- routes.py MUST include /api/register (POST), /api/login (POST), /api/user (GET)
- login returns: {"token": create_access_token(identity=user.id), "user": user.to_dict()}
- All list endpoints: ?page=1&per_page=20, paginate(), return {"items":[...], "total":n, "page":n, "pages":n}
- All write endpoints: @jwt_required() + validate required fields → 400 if missing
- Registration: new_user.set_password(data['password']) — NEVER store plain text
- Login: user.check_password(data['password']) — NEVER compare with ==

CRITICAL — DEPENDENCY ORDER (wave placement):
backend/models.py MUST have "depends_on": [] (empty — Wave 1 placement).
If models.py depends on config.py, it builds in Wave 2 alongside 9+ components
and fails due to provider rate limit exhaustion. Empty depends_on = Wave 1.

OUTPUT FORMAT:
{
  "project_name": "snake_case_name",
  "description": "one line description",
  "stack": {"frontend": "React", "backend": "Flask", "database": "PostgreSQL"},
  "files": [
    {"path": "backend/__init__.py", "description": "...", "depends_on": []},
    {"path": "backend/config.py", "description": "...", "depends_on": []},
    {"path": "backend/models.py", "description": "...", "depends_on": []},
    {"path": "backend/routes.py", "description": "...", "depends_on": ["backend/models.py"]},
    {"path": "backend/app.py", "description": "...", "depends_on": ["backend/config.py", "backend/models.py", "backend/routes.py"]},
    {"path": "frontend/package.json", "description": "...", "depends_on": []},
    {"path": "frontend/public/index.html", "description": "...", "depends_on": []},
    {"path": "frontend/src/index.css", "description": "...", "depends_on": []},
    {"path": "frontend/src/api.js", "description": "...", "depends_on": []},
    {"path": "frontend/src/components/PrivateRoute.js", "description": "...", "depends_on": []},
    {"path": "frontend/src/components/Navbar.js", "description": "...", "depends_on": ["frontend/src/api.js"]},
    {"path": "frontend/src/components/Home.js", "description": "...", "depends_on": []},
    {"path": "frontend/src/components/Login.js", "description": "...", "depends_on": ["frontend/src/api.js"]},
    {"path": "frontend/src/components/Register.js", "description": "...", "depends_on": ["frontend/src/api.js"]},
    ... (add resource components here based on the project)
    {"path": "frontend/src/App.js", "description": "...", "depends_on": ["frontend/src/components/...]},
    {"path": "frontend/src/index.js", "description": "...", "depends_on": ["frontend/src/App.js"]},
    {"path": ".env.example", "description": "...", "depends_on": []}
  ],
  "database_schema": {
    "tables": [
      {"name": "users", "columns": [
        {"name": "id", "type": "Integer PK"},
        {"name": "username", "type": "String(100) unique"},
        {"name": "email", "type": "String(255) unique"},
        {"name": "password", "type": "String(255) hashed"},
        {"name": "created_at", "type": "DateTime"}
      ]}
    ]
  },
  "api_endpoints": [
    {"method": "POST", "path": "/api/register", "description": "Register new user", "auth_required": false},
    {"method": "POST", "path": "/api/login", "description": "Login and get JWT token", "auth_required": false},
    {"method": "GET", "path": "/api/user", "description": "Get current user info", "auth_required": true}
  ],
  "setup_instructions": [
    "cp .env.example .env && fill in values",
    "pip install -r requirements.txt",
    "flask db init && flask db migrate && flask db upgrade",
    "python -m backend.app",
    "cd frontend && npm install && npm start"
  ]
}
"""

# Files that MUST exist in every blueprint regardless of what the LLM generates
REQUIRED_FILES = [
    "backend/__init__.py",
    "backend/config.py",
    "backend/models.py",
    "backend/routes.py",
    "backend/app.py",
    "frontend/package.json",
    "frontend/public/index.html",
    "frontend/src/App.js",
    "frontend/src/index.js",
    "frontend/src/index.css",
    "frontend/src/api.js",
    "frontend/src/components/Navbar.js",
    "frontend/src/components/Login.js",
    "frontend/src/components/Register.js",
    "frontend/src/components/Home.js",
    "frontend/src/components/PrivateRoute.js",
    ".env.example",
]


def _enforce_required_files(blueprint):
    """Add any missing required files to the blueprint."""
    existing_paths = {f["path"] for f in blueprint["files"]}
    for required in REQUIRED_FILES:
        if required not in existing_paths:
            print(f"  ⚠️  Adding missing required file: {required}")
            blueprint["files"].append({
                "path": required,
                "description": f"Required file: {required}",
                "depends_on": [],
            })


def _enforce_models_wave1(blueprint):
    """
    Force backend/models.py to depends_on: [] so it builds in Wave 1.
    If models.py is in Wave 2 (alongside 9+ parallel components), provider
    rate limits cause all 3 build retries to fail with syntax errors.
    """
    for f in blueprint["files"]:
        if f["path"] == "backend/models.py":
            if f.get("depends_on"):
                f["depends_on"] = []
                print("  ✅ Enforced: backend/models.py → Wave 1 (depends_on: [])")
            break


def _auto_add_missing_components(blueprint):
    """
    Scan App.js description for PascalCase component names that are referenced
    but not yet in the files list. Auto-add them so the builder doesn't hit
    missing_component errors.

    BUG-4 FIX: Only adds names NOT in EXCLUDED_COMPONENT_NAMES, preventing
    ghost files like Required.js, Auth.js, Loading.js from being generated.
    """
    app_entry = next((f for f in blueprint["files"] if f["path"] == "frontend/src/App.js"), None)
    if not app_entry:
        return

    existing_paths = {f["path"] for f in blueprint["files"]}
    desc = app_entry.get("description", "")

    # Also scan the component descriptions for referenced names
    for f in blueprint["files"]:
        if "components/" in f.get("path", ""):
            desc += " " + f.get("description", "")

    component_names = re.findall(r'\b([A-Z][a-zA-Z]+)\b', desc)

    for name in component_names:
        if name in EXCLUDED_COMPONENT_NAMES:
            continue
        component_path = f"frontend/src/components/{name}.js"
        if component_path not in existing_paths:
            print(f"  ⚠️  Auto-adding referenced component: {component_path}")
            blueprint["files"].append({
                "path": component_path,
                "description": f"{name} component for the application",
                "depends_on": ["frontend/src/api.js"],
            })
            existing_paths.add(component_path)


def _ensure_init_py(blueprint):
    """backend/__init__.py must always be present and first."""
    existing_paths = {f["path"] for f in blueprint["files"]}
    if "backend/__init__.py" not in existing_paths:
        blueprint["files"].insert(0, {
            "path": "backend/__init__.py",
            "description": "Makes backend a Python package. Initializes db = SQLAlchemy() and jwt = JWTManager().",
            "depends_on": [],
        })


def _sort_by_wave(blueprint):
    """
    Sort files by dependency depth so the wave builder processes them correctly.
    Files with 0 deps → Wave 1 (built first, providers fresh).
    Files with 1 dep → Wave 2.
    Files with 2+ deps → Wave 3+ (built last).
    Stable sort preserves relative order within the same wave.
    """
    blueprint["files"] = sorted(
        blueprint["files"],
        key=lambda f: len(f.get("depends_on", [])),
    )


def generate_blueprint(project_description):
    print("\n🧠 Planner Agent thinking...")

    response = call_llm([
        {"role": "system", "content": PLANNER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Create a complete blueprint for: {project_description}\n\n"
                "REMINDERS:\n"
                "- Every component App.js imports MUST be in the files list\n"
                "- backend/models.py MUST have depends_on: [] (Wave 1)\n"
                "- NEVER generate Required.js, Auth.js, Guard.js, Loading.js, Error.js\n"
                "- login returns {\"token\": ..., \"user\": ...} — key is 'token' not 'access_token'"
            ),
        },
    ])

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model wrapped in them despite instructions
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        blueprint = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"❌ Planner returned invalid JSON: {e}")
        print(f"   Raw output (first 500 chars):\n{raw[:500]}")
        return None

    # ── Post-processing pipeline ────────────────────────────────
    # Order matters: enforce → filter → auto-add → sort

    # 1. Ensure required files exist
    _enforce_required_files(blueprint)

    # 2. Force models.py to Wave 1
    _enforce_models_wave1(blueprint)

    # 3. Remove ghost/utility component files (BUG-4 fix)
    blueprint["files"] = filter_ghost_components(blueprint["files"])

    # 4. Auto-add any components referenced in descriptions but missing from files
    _auto_add_missing_components(blueprint)

    # 5. Ensure __init__.py exists after filtering
    _ensure_init_py(blueprint)

    # 6. Sort by wave (dependency depth)
    _sort_by_wave(blueprint)

    # ── Summary ────────────────────────────────────────────────
    component_names = [
        f["path"].split("/")[-1]
        for f in blueprint["files"]
        if "components/" in f["path"]
    ]
    table_names = [
        t["name"] if isinstance(t, dict) else t
        for t in blueprint.get("database_schema", {}).get("tables", [])
    ]

    print(f"\n  ✅ Blueprint: {blueprint.get('project_name', '?')}")
    print(f"  📁 Files: {len(blueprint['files'])}")
    print(f"  🗄️  Tables: {table_names}")
    print(f"  🔗 Endpoints: {len(blueprint.get('api_endpoints', []))}")
    print(f"  📦 Components: {component_names}")

    return blueprint


if __name__ == "__main__":
    blueprint = generate_blueprint(
        "A real-time chat app where users can create rooms and send messages"
    )
    if blueprint:
        os.makedirs("sandbox", exist_ok=True)
        with open("sandbox/blueprint.json", "w") as f:
            json.dump(blueprint, f, indent=2)
        print("\n📄 Blueprint saved to sandbox/blueprint.json")