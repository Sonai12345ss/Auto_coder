import os
import re
import json
import time
import threading
from datetime import datetime, timedelta
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# PLANNER HEALTH TRACKER
#
# Root cause from build log: the builder hammers all Gemini
# providers across its waves, putting them on 120s cooldown.
# When the planner runs (same process, ~2 min later), Gemini
# providers are still blocked — but the planner has no health
# state, so it blindly retries all 6 Gemini providers and waits
# 1+2+4+8+1+2 = 18 seconds before finally reaching Groq.
#
# Fix: a shared health tracker that the planner uses to skip
# already-blocked providers. The builder's ProviderHealth is a
# separate instance — this one is for the planner only, but it
# reads the same cooldown logic.
# ═══════════════════════════════════════════════════════════════

class _PlannerHealth:
    def __init__(self):
        self._blocked_until = {}
        self._lock = threading.Lock()

    def block(self, name, seconds=120):
        with self._lock:
            self._blocked_until[name] = datetime.now() + timedelta(seconds=seconds)

    def is_available(self, name):
        with self._lock:
            return datetime.now() >= self._blocked_until.get(name, datetime.min)

    def reset(self):
        with self._lock:
            self._blocked_until.clear()

_health = _PlannerHealth()


# ═══════════════════════════════════════════════════════════════
# PROVIDER SETUP
#
# ORDER RATIONALE (from build log analysis):
# The builder runs before the planner returns, exhausting all
# Gemini providers across its waves. By the time generate_blueprint
# is called, all 6 Gemini Pro/Flash slots are on 120s cooldown.
# Placing Groq first means the planner succeeds on attempt 1
# without wasting ~18 seconds on blocked Gemini providers.
# Gemini is kept as fallback for when Groq is also rate limited.
# ═══════════════════════════════════════════════════════════════

groq1      = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2      = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3      = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",  api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1",  api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

# max_tokens=8192: e-commerce blueprints with 24+ files need ~3000 tokens of
# JSON output. At 4096, the model hits the limit mid-JSON (char ~9700) and
# produces unterminated strings. 8192 gives headroom for 30-file blueprints.
_MAX_TOKENS = 8192

PROVIDERS = [
    # ── Groq first: always available when Gemini is exhausted post-build ──
    ("Groq-1 / llama-3.3-70b",      lambda m: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-2 / llama-3.3-70b",      lambda m: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-3 / llama-3.3-70b",      lambda m: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    # ── Gemini Flash: good JSON quality, usually available after Groq ──────
    ("Gemini-1 / gemini-2.5-flash",  lambda m: gemini1.chat.completions.create(model="gemini-2.5-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-2 / gemini-2.5-flash",  lambda m: gemini2.chat.completions.create(model="gemini-2.5-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-3 / gemini-2.5-flash",  lambda m: gemini3.chat.completions.create(model="gemini-2.5-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    # ── Gemini Pro: best JSON quality but most likely to be rate limited ───
    ("Gemini-1 / gemini-2.5-pro",    lambda m: gemini1.chat.completions.create(model="gemini-2.5-pro",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-2 / gemini-2.5-pro",    lambda m: gemini2.chat.completions.create(model="gemini-2.5-pro",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-3 / gemini-2.5-pro",    lambda m: gemini3.chat.completions.create(model="gemini-2.5-pro",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    # ── Smaller/fast models: for fallback only ─────────────────────────────
    ("Groq-1 / llama-3.1-8b",        lambda m: groq1.chat.completions.create(model="llama-3.1-8b-instant",   messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-2 / llama-3.1-8b",        lambda m: groq2.chat.completions.create(model="llama-3.1-8b-instant",   messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-3 / llama-3.1-8b",        lambda m: groq3.chat.completions.create(model="llama-3.1-8b-instant",   messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-1 / gemini-2.0-flash",  lambda m: gemini1.chat.completions.create(model="gemini-2.0-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-2 / gemini-2.0-flash",  lambda m: gemini2.chat.completions.create(model="gemini-2.0-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-3 / gemini-2.0-flash",  lambda m: gemini3.chat.completions.create(model="gemini-2.0-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("OpenRouter / llama-3.3-70b",   lambda m: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("OpenRouter / gemma-3-27b",     lambda m: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free",              messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("OpenRouter / gemma-3-12b",     lambda m: openrouter.chat.completions.create(model="google/gemma-3-12b-it:free",              messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Doubleword / Qwen3.5-35B",     lambda m: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",               messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Doubleword / Qwen3.5-397B",    lambda m: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8",             messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
]


def call_llm(messages):
    """
    Call providers in order, skipping any that are health-blocked.
    On rate limit: block the provider and immediately try the next one
    (no sleep — the next provider in the list is already different).
    Sleep only when all available providers are exhausted.
    """
    available = [p for p in PROVIDERS if _health.is_available(p[0])]
    if not available:
        print("  ⚠️  All planner providers in cooldown — resetting health and retrying...")
        _health.reset()
        available = PROVIDERS

    last_error = None
    for attempt, (name, fn) in enumerate(available):
        try:
            print(f"  🤖 Planner using {name}...")
            return fn(messages)
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "402", "503",
                                       "quota", "temporarily", "overloaded", "upstream"]):
                cooldown = 120 if "gemini" in name.lower() else 60
                _health.block(name, seconds=cooldown)
                print(f"  ⚠️  {name} rate limited (blocked {cooldown}s), trying next...")
                last_error = e
                continue
            elif any(x in err for x in ["decommission", "deprecated", "no longer supported",
                                         "invalid model", "404"]):
                _health.block(name, seconds=600)
                print(f"  ⚠️  {name} model unavailable, skipping...")
                last_error = e
                continue
            last_error = e
            print(f"  ⚠️  {name} failed: {str(e)[:80]}, trying next...")
            continue

    raise Exception(f"All planner providers failed. Last error: {last_error}")


# ═══════════════════════════════════════════════════════════════
# JSON TRUNCATION REPAIR
#
# When a model hits its token limit mid-JSON (typically at char
# ~9000-9700 for a 24-file blueprint), the output is syntactically
# valid up to the truncation point but missing closing brackets.
# This function closes any open structures so json.loads can parse
# the partial result. _enforce_required_files() then fills in any
# missing required files from the partial blueprint.
# ═══════════════════════════════════════════════════════════════

def _try_repair_json(raw):
    """
    Attempt to repair truncated JSON by closing unclosed structures.
    Returns (repaired_string, was_repaired: bool).
    Only works for truncation — won't fix structural JSON errors.
    """
    s = raw.strip()
    stack = []
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if stack:
                stack.pop()

    if not stack:
        return s, False  # Already closed (valid or unfixable structural error)

    # Close any mid-string truncation
    if in_string:
        s += '"'

    # Remove trailing comma before closing
    s = s.rstrip()
    if s.endswith(','):
        s = s[:-1]

    # Close open structures in reverse order
    closing = {'{': '}', '[': ']'}
    for opener in reversed(stack):
        s += closing[opener]

    return s, True


# ═══════════════════════════════════════════════════════════════
# GHOST COMPONENT FILTERING
#
# Two sources of ghost components:
# 1. LLM generates them directly (covered by GHOST_FILE_PATHS)
# 2. _auto_add_missing_components() adds them from PascalCase
#    words in descriptions (covered by EXCLUDED_COMPONENT_NAMES)
# ═══════════════════════════════════════════════════════════════

EXCLUDED_COMPONENT_NAMES = {
    # React Router / React internals
    "React", "Route", "Routes", "BrowserRouter", "Navigate", "Outlet",
    "Link", "NavLink", "Switch", "RouterProvider", "Redirect",
    # Structural / layout (always handled elsewhere)
    "App", "Router", "Routing", "Component", "Fragment",
    "Provider", "Context", "Suspense", "Layout", "Footer", "Navbar",
    # Auth guard duplicates (PrivateRoute is the canonical one)
    # Note: PrivateRoute itself is intentionally absent — it's a real template
    "Protected", "AuthRoute", "GuardedRoute",
    # BUG-4: guard/utility words from descriptions
    "Required", "Auth", "Guard", "Private",
    "Loading", "Error", "Modal", "Alert", "Toast",
    "Base", "Common", "Utils", "Helper", "Shared", "Wrapper",
    "Index", "NotFound", "Fallback",
    # From build log: Groq-generated non-standard names that are duplicates
    # or generic placeholders rather than real page components:
    # "Main" → too generic; real component should be MainContent, MainPage, etc.
    # "Registration" → near-duplicate of "Register" (already in REQUIRED_FILES)
    "Main", "Registration",
}

GHOST_FILE_PATHS = {
    # Routing wrapper files (routing lives in App.js)
    "frontend/src/components/Routing.js",
    "frontend/src/components/Router.js",
    "frontend/src/components/Routes.js",
    "frontend/src/Routing.js",
    "frontend/src/Router.js",
    # Auth guard duplicates
    "frontend/src/components/Protected.js",
    "frontend/src/components/AuthRoute.js",
    "frontend/src/components/GuardedRoute.js",
    # BUG-4 utility/guard names
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
    # From build log: Groq-generated generics
    "frontend/src/components/Main.js",
    "frontend/src/components/Registration.js",
}


def filter_ghost_components(files_list):
    """
    Two-pass filter:
    Pass 1: Remove files whose full path is in GHOST_FILE_PATHS.
    Pass 2: Remove components/ files whose base name is in EXCLUDED_COMPONENT_NAMES.
    PrivateRoute.js is intentionally preserved (not in either set).
    """
    filtered = []
    removed = []

    for file_info in files_list:
        path = file_info.get("path", "")

        if path in GHOST_FILE_PATHS:
            removed.append(path)
            continue

        if path.startswith("frontend/src/components/") and path.endswith(".js"):
            name = os.path.basename(path).replace(".js", "")
            if name in EXCLUDED_COMPONENT_NAMES:
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
5. NEVER generate utility/guard/generic component files — see forbidden list below.

FORBIDDEN FILES — never generate these:
❌ Routing.js, Router.js, Routes.js (routing lives inside App.js)
❌ Required.js, Auth.js, Guard.js, Private.js (auth guard = PrivateRoute.js only)
❌ Loading.js, Error.js, Modal.js, Wrapper.js, NotFound.js (not real page components)
❌ Main.js (too generic — name the component properly e.g. Dashboard.js, ProductPage.js)
❌ Registration.js (use Register.js — it's already in the required files)
❌ Any file whose purpose is "guard", "wrapper", "utility", "helper", or "auth check"

NAMING RULES FOR COMPONENTS:
- Use descriptive resource-specific names: ProductList.js, OrderForm.js, CartDetail.js
- NEVER use generic single-word names: Main.js, Data.js, Page.js, View.js, Item.js
- Register.js already exists — don't generate Registration.js or Signup.js

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
    ... (add resource-specific components here, e.g. ProductList.js, ProductForm.js)
    {"path": "frontend/src/App.js", "description": "...", "depends_on": ["frontend/src/components/Home.js", "...all components..."]},
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


# ─────────────────────────────────────────────
# POST-PROCESSING PIPELINE STEPS
# ─────────────────────────────────────────────

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
    In Wave 1 all providers are fresh. In Wave 2 (alongside 9+ parallel
    components) provider rate limits cause all 3 builder retries to fail.
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
    missing_component errors during tester validation.

    Only adds names NOT in EXCLUDED_COMPONENT_NAMES, preventing ghost files
    like Registration.js, Main.js, Auth.js from being generated.
    """
    app_entry = next((f for f in blueprint["files"] if f["path"] == "frontend/src/App.js"), None)
    if not app_entry:
        return

    existing_paths = {f["path"] for f in blueprint["files"]}

    # Scan App.js description + all component descriptions for referenced names
    desc = app_entry.get("description", "")
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
    """backend/__init__.py must always exist (sometimes filtered out or missing)."""
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
    0 deps → Wave 1 (providers fresh). 1 dep → Wave 2. 2+ deps → Wave 3+.
    Stable sort preserves relative order within the same wave.
    """
    blueprint["files"] = sorted(
        blueprint["files"],
        key=lambda f: len(f.get("depends_on", [])),
    )


def _process_blueprint(blueprint):
    """Run the full post-processing pipeline on a parsed blueprint."""
    _enforce_required_files(blueprint)
    _enforce_models_wave1(blueprint)
    blueprint["files"] = filter_ghost_components(blueprint["files"])
    _auto_add_missing_components(blueprint)
    _ensure_init_py(blueprint)
    _sort_by_wave(blueprint)
    return blueprint


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def generate_blueprint(project_description, max_attempts=3):
    """
    Generate a project blueprint with up to max_attempts retries.

    On JSON parse failure: attempts _try_repair_json() first (handles
    token-limit truncation). If repair also fails, retries with a fresh
    provider call. A partial repaired blueprint is still useful because
    _enforce_required_files() fills in any missing required files.
    """
    print("\n🧠 Planner Agent thinking...")

    user_message = (
        f"Create a complete blueprint for: {project_description}\n\n"
        "REMINDERS:\n"
        "- Every component App.js imports MUST be in the files list\n"
        "- backend/models.py MUST have depends_on: [] (Wave 1)\n"
        "- NEVER generate Registration.js (use Register.js), Main.js, Auth.js, Loading.js\n"
        "- Use descriptive names: ProductList.js not Product.js, OrderForm.js not Order.js\n"
        "- login returns {\"token\": ..., \"user\": ...} — key is 'token' not 'access_token'"
    )

    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"  🔄 Planner retry attempt {attempt}/{max_attempts}...")

        try:
            response = call_llm(messages)
        except Exception as e:
            print(f"  ❌ All providers exhausted: {e}")
            return None

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model added them despite instructions
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # ── Parse with repair fallback ──────────────────────────────────────
        blueprint = None
        try:
            blueprint = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Attempt {attempt}: invalid JSON ({e}) — trying repair...")
            repaired, was_repaired = _try_repair_json(raw)
            if was_repaired:
                try:
                    blueprint = json.loads(repaired)
                    print(f"  ✅ JSON repair succeeded ({len(raw)} → {len(repaired)} chars)")
                except json.JSONDecodeError as e2:
                    print(f"  ❌ Repair failed ({e2}) — raw (first 300): {raw[:300]}")
            else:
                print(f"  ❌ Not repairable ({e}) — raw (first 300): {raw[:300]}")

        if blueprint is None:
            if attempt < max_attempts:
                time.sleep(2)
                continue
            print("  ❌ Planner failed after all attempts")
            return None
        # ────────────────────────────────────────────────────────────────────

        # Post-processing pipeline
        _process_blueprint(blueprint)

        # Summary
        component_names = [f["path"].split("/")[-1] for f in blueprint["files"] if "components/" in f["path"]]
        table_names = [t["name"] if isinstance(t, dict) else t for t in blueprint.get("database_schema", {}).get("tables", [])]

        print(f"\n  ✅ Blueprint: {blueprint.get('project_name', '?')}")
        print(f"  📁 Files: {len(blueprint['files'])}")
        print(f"  🗄️  Tables: {table_names}")
        print(f"  🔗 Endpoints: {len(blueprint.get('api_endpoints', []))}")
        print(f"  📦 Components: {component_names}")
        return blueprint

    print("  ❌ Planner failed after all attempts")
    return None


if __name__ == "__main__":
    blueprint = generate_blueprint(
        "A real-time chat app where users can create rooms and send messages"
    )
    if blueprint:
        os.makedirs("sandbox", exist_ok=True)
        with open("sandbox/blueprint.json", "w") as f:
            json.dump(blueprint, f, indent=2)
        print("\n📄 Blueprint saved to sandbox/blueprint.json")