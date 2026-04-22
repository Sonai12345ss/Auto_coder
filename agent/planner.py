import os
import re
import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# PLANNER HEALTH TRACKER
#
# Tracks rate-limited providers so they are skipped on the next
# call rather than retried blindly. The builder exhausts Gemini
# across its waves (120s cooldown); the planner seeing those same
# providers blocked wastes 18s of useless retries without this.
#
# FIX 2 (improved reset): When ALL providers are blocked, wait
# for the shortest remaining cooldown before resetting, rather
# than resetting instantly (which just re-fails immediately).
# ═══════════════════════════════════════════════════════════════

class _PlannerHealth:
    def __init__(self):
        self._blocked_until = {}
        self._lock = threading.Lock()

    def block(self, name, seconds):
        with self._lock:
            self._blocked_until[name] = datetime.now() + timedelta(seconds=seconds)

    def is_available(self, name):
        with self._lock:
            return datetime.now() >= self._blocked_until.get(name, datetime.min)

    def soonest_available_in(self):
        """Return seconds until the soonest blocked provider becomes available."""
        with self._lock:
            now = datetime.now()
            waits = [(bt - now).total_seconds() for bt in self._blocked_until.values() if bt > now]
            return min(waits) if waits else 0

    def reset_non_gemini(self):
        """Reset only Groq/OpenRouter/Doubleword — they recover faster than Gemini."""
        with self._lock:
            self._blocked_until = {
                k: v for k, v in self._blocked_until.items()
                if "gemini" in k.lower()
            }

    def reset(self):
        with self._lock:
            self._blocked_until.clear()

_health = _PlannerHealth()


# ═══════════════════════════════════════════════════════════════
# PROVIDER SETUP
#
# FIX 3 (timeout=45.0): Without a timeout, a provider that hangs
# stalls the entire build on Render's free tier indefinitely.
# 45s is generous enough for large JSON responses (planner needs
# to return 3000+ token blueprints) while still being a reasonable
# hard cutoff.
#
# ORDER: Groq first (fresh when Gemini is exhausted post-build),
# then Gemini Flash (good JSON quality), Gemini Pro last (best
# quality but most rate-limited), then smaller fallbacks.
# ═══════════════════════════════════════════════════════════════

_TIMEOUT = 45.0   # seconds — FIX 3: prevents infinite hangs
_MAX_TOKENS = 8192  # e-commerce blueprints need ~3000 tokens; 4096 caused truncation

groq1      = Groq(api_key=os.environ.get("GROQ_API_KEY", ""),      timeout=_TIMEOUT)
groq2      = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""),    timeout=_TIMEOUT)
groq3      = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""),    timeout=_TIMEOUT)
gemini1    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""),   timeout=_TIMEOUT)
gemini2    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""), timeout=_TIMEOUT)
gemini3    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""), timeout=_TIMEOUT)
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",  api_key=os.environ.get("OPENROUTER_API_KEY", ""), timeout=_TIMEOUT)
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1",  api_key=os.environ.get("DOUBLEWORD_API_KEY", ""), timeout=_TIMEOUT)

PROVIDERS = [
    # Groq first — available when Gemini is exhausted from builder waves
    ("Groq-1 / llama-3.3-70b",      lambda m: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-2 / llama-3.3-70b",      lambda m: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-3 / llama-3.3-70b",      lambda m: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    # Gemini Flash: good JSON quality, faster quota recovery than Pro
    ("Gemini-1 / gemini-2.5-flash",  lambda m: gemini1.chat.completions.create(model="gemini-2.5-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-2 / gemini-2.5-flash",  lambda m: gemini2.chat.completions.create(model="gemini-2.5-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-3 / gemini-2.5-flash",  lambda m: gemini3.chat.completions.create(model="gemini-2.5-flash",  messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    # Gemini Pro: best JSON quality but most rate-limited
    ("Gemini-1 / gemini-2.5-pro",    lambda m: gemini1.chat.completions.create(model="gemini-2.5-pro",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-2 / gemini-2.5-pro",    lambda m: gemini2.chat.completions.create(model="gemini-2.5-pro",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Gemini-3 / gemini-2.5-pro",    lambda m: gemini3.chat.completions.create(model="gemini-2.5-pro",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    # Smaller/fast fallbacks
    ("Groq-1 / llama-3.1-8b",        lambda m: groq1.chat.completions.create(model="llama-3.1-8b-instant",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-2 / llama-3.1-8b",        lambda m: groq2.chat.completions.create(model="llama-3.1-8b-instant",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
    ("Groq-3 / llama-3.1-8b",        lambda m: groq3.chat.completions.create(model="llama-3.1-8b-instant",    messages=m, temperature=0.2, max_tokens=_MAX_TOKENS)),
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
    Call providers in order, skipping health-blocked ones.
    On rate limit: block that provider and immediately try the next.
    
    FIX 2 (improved reset): When ALL providers are blocked, wait for the
    soonest non-Gemini recovery (Groq recovers in 60s, not 120s) rather
    than resetting instantly and failing again immediately.
    """
    available = [p for p in PROVIDERS if _health.is_available(p[0])]
    if not available:
        wait = _health.soonest_available_in()
        if wait > 0:
            actual_wait = min(wait, 30)  # cap at 30s — Groq recovers in 60s total
            print(f"  ⚠️  All planner providers in cooldown — waiting {actual_wait:.0f}s for recovery...")
            time.sleep(actual_wait)
        _health.reset_non_gemini()  # reset only Groq/OR/DW; keep Gemini blocked
        available = [p for p in PROVIDERS if _health.is_available(p[0])]
        if not available:
            # Groq still blocked — full reset as last resort
            _health.reset()
            available = PROVIDERS

    last_error = None
    for name, fn in available:
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
            elif any(x in err for x in ["timeout", "timed out", "read timeout"]):
                # FIX 3: timeout hit — block briefly and move to next provider
                _health.block(name, seconds=30)
                print(f"  ⚠️  {name} timed out after {_TIMEOUT}s, trying next...")
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
# ═══════════════════════════════════════════════════════════════

def _try_repair_json(raw):
    """
    Close unclosed JSON structures from token-limit truncation.
    Returns (repaired_string, was_repaired: bool).
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
        return s, False

    if in_string:
        s += '"'
    s = s.rstrip()
    if s.endswith(','):
        s = s[:-1]
    closing = {'{': '}', '[': ']'}
    for opener in reversed(stack):
        s += closing[opener]

    return s, True


# ═══════════════════════════════════════════════════════════════
# FIX 4 — BLUEPRINT SCHEMA VALIDATION
#
# json.loads can succeed even when the blueprint is structurally
# wrong — missing keys, wrong types, empty files list.
# A lightweight validation pass catches these before post-processing
# tries to iterate over None or missing sections.
# ═══════════════════════════════════════════════════════════════

def _validate_blueprint(blueprint):
    """
    Lightweight structural validation — no external dependencies.
    Returns (is_valid: bool, errors: list[str]).
    
    Checks:
    - Required top-level keys exist and have correct types
    - files list has at least 5 entries (< 5 = almost certainly truncated)
    - Each file entry has path (str), description (str), depends_on (list)
    - No duplicate file paths
    - api_endpoints is a non-empty list
    """
    errors = []

    # Top-level keys
    required_keys = {
        "project_name": str,
        "description": str,
        "stack": dict,
        "files": list,
        "database_schema": dict,
        "api_endpoints": list,
    }
    for key, expected_type in required_keys.items():
        if key not in blueprint:
            errors.append(f"Missing top-level key: '{key}'")
        elif not isinstance(blueprint[key], expected_type):
            errors.append(f"'{key}' must be {expected_type.__name__}, got {type(blueprint[key]).__name__}")

    if errors:
        return False, errors

    # files list minimum size
    files = blueprint["files"]
    if len(files) < 5:
        errors.append(f"files list has only {len(files)} entries — blueprint is likely truncated (minimum 5)")

    # Each file entry structure
    seen_paths = set()
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            errors.append(f"files[{i}] is not an object")
            continue
        if "path" not in f or not isinstance(f["path"], str) or not f["path"]:
            errors.append(f"files[{i}] missing or invalid 'path'")
        elif f["path"] in seen_paths:
            errors.append(f"Duplicate file path: '{f['path']}'")
        else:
            seen_paths.add(f["path"])
        if "depends_on" not in f or not isinstance(f["depends_on"], list):
            errors.append(f"files[{i}] ('{f.get('path', '?')}') missing or invalid 'depends_on'")

    # api_endpoints non-empty
    if len(blueprint["api_endpoints"]) == 0:
        errors.append("api_endpoints is empty — must include at least /api/register, /api/login, /api/user")

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════
# GHOST COMPONENT FILTERING
# ═══════════════════════════════════════════════════════════════

# FIX 1 (expanded exclusion set): The auto-add regex catches all PascalCase
# words in descriptions — including data model names like User, Token, Database.
# Expanding this set prevents ghost component files from being created.
EXCLUDED_COMPONENT_NAMES = {
    # React Router / React internals
    "React", "Route", "Routes", "BrowserRouter", "Navigate", "Outlet",
    "Link", "NavLink", "Switch", "RouterProvider", "Redirect",
    # Structural / layout
    "App", "Router", "Routing", "Component", "Fragment",
    "Provider", "Context", "Suspense", "Layout", "Footer", "Navbar",
    # Auth guard duplicates (PrivateRoute is canonical — intentionally absent)
    "Protected", "AuthRoute", "GuardedRoute",
    # Guard / utility words from descriptions
    "Required", "Auth", "Guard", "Private",
    "Loading", "Error", "Modal", "Alert", "Toast",
    "Base", "Common", "Utils", "Helper", "Shared", "Wrapper",
    "Index", "NotFound", "Fallback",
    # From build logs: Groq-generated generic names
    "Main", "Registration",
    # FIX 1: data-model / technical names that appear in descriptions but
    # are NOT React components — e.g. "handles User authentication" → User.js
    "User", "Token", "Database", "Api", "Jwt", "Http", "Url", "Id", "Uuid",
    "String", "Integer", "Boolean", "Object", "Array", "Dict",
    "Config", "Schema", "Model", "Service", "Handler",
    "Manager", "Controller", "Middleware", "Decorator",
    "Request", "Response", "Session", "Cookie", "Header",
    "Password", "Email", "Username", "Role", "Permission",
    "Status", "Type", "Category", "Tag", "Label", "Badge",
}

GHOST_FILE_PATHS = {
    "frontend/src/components/Routing.js",
    "frontend/src/components/Router.js",
    "frontend/src/components/Routes.js",
    "frontend/src/Routing.js",
    "frontend/src/Router.js",
    "frontend/src/components/Protected.js",
    "frontend/src/components/AuthRoute.js",
    "frontend/src/components/GuardedRoute.js",
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
    "frontend/src/components/Main.js",
    "frontend/src/components/Registration.js",
    # FIX 1: data-model ghost paths
    "frontend/src/components/User.js",
    "frontend/src/components/Token.js",
    "frontend/src/components/Database.js",
    "frontend/src/components/Model.js",
    "frontend/src/components/Schema.js",
    "frontend/src/components/Config.js",
    "frontend/src/components/Api.js",
    "frontend/src/components/Service.js",
}


def filter_ghost_components(files_list):
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
# POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────

def _enforce_required_files(blueprint):
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
    """Force backend/models.py to depends_on: [] — builds in Wave 1 when providers are fresh."""
    for f in blueprint["files"]:
        if f["path"] == "backend/models.py":
            if f.get("depends_on"):
                f["depends_on"] = []
                print("  ✅ Enforced: backend/models.py → Wave 1 (depends_on: [])")
            break


def _auto_add_missing_components(blueprint):
    """
    Scan App.js description for PascalCase names not yet in files list.
    FIX 1: EXCLUDED_COMPONENT_NAMES is now much larger to prevent data-model
    names (User, Token, Database) from becoming ghost component files.
    """
    app_entry = next((f for f in blueprint["files"] if f["path"] == "frontend/src/App.js"), None)
    if not app_entry:
        return

    existing_paths = {f["path"] for f in blueprint["files"]}
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
    existing_paths = {f["path"] for f in blueprint["files"]}
    if "backend/__init__.py" not in existing_paths:
        blueprint["files"].insert(0, {
            "path": "backend/__init__.py",
            "description": "Makes backend a Python package. Initializes db = SQLAlchemy() and jwt = JWTManager().",
            "depends_on": [],
        })


def _topological_sort(blueprint):
    """
    FIX 5: Kahn's algorithm topological sort — replaces the naive len(depends_on) sort.

    Old sort: key=len(depends_on) — WRONG for A→B→C chains where all have 1 dep.
    New sort: proper topological order matching what builder.py's get_waves() expects.

    Example: __init__.py → models.py → routes.py → app.py
    Old: [__init__(0), models(0), routes(1), app(3)] — correct by coincidence
    Real win: A(1)→B(1)→C(1) chain — old sort fails, new sort gives A, B, C.

    Falls back to dep-count sort if a dependency cycle is detected.
    """
    files = blueprint["files"]
    path_to_file = {f["path"]: f for f in files}
    all_paths = set(path_to_file.keys())

    # Build in-degree and dependent adjacency
    in_degree = {p: 0 for p in all_paths}
    dependents = {p: [] for p in all_paths}

    for f in files:
        for dep in f.get("depends_on", []):
            if dep in all_paths:  # Only count deps that exist in the blueprint
                in_degree[f["path"]] += 1
                dependents[dep].append(f["path"])

    # Process nodes with 0 in-degree (Kahn's BFS)
    queue = deque(sorted(p for p in all_paths if in_degree[p] == 0))
    sorted_paths = []

    while queue:
        path = queue.popleft()
        sorted_paths.append(path)
        for dependent in sorted(dependents[path]):  # sorted for determinism
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_paths) != len(files):
        # Cycle detected — fall back to dep-count sort (always safe)
        print("  ⚠️  Dependency cycle detected in blueprint — using dep-count fallback")
        blueprint["files"] = sorted(files, key=lambda f: len(f.get("depends_on", [])))
        return

    blueprint["files"] = [path_to_file[p] for p in sorted_paths]


def _process_blueprint(blueprint):
    """Run the full post-processing pipeline on a parsed blueprint."""
    _enforce_required_files(blueprint)
    _enforce_models_wave1(blueprint)
    blueprint["files"] = filter_ghost_components(blueprint["files"])
    _auto_add_missing_components(blueprint)
    _ensure_init_py(blueprint)
    _topological_sort(blueprint)  # FIX 5: was _sort_by_wave (naive len sort)
    return blueprint


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def generate_blueprint(project_description, max_attempts=3):
    """
    Generate a project blueprint with up to max_attempts retries.

    Flow per attempt:
    1. call_llm() — health-aware provider selection
    2. Strip markdown fences
    3. json.loads() — try parse
    4. If parse fails → _try_repair_json() — close truncated brackets
    5. _validate_blueprint() — FIX 4: check structural correctness
    6. If invalid → retry with next attempt
    7. _process_blueprint() — enforce, filter, sort
    8. Return complete blueprint
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

        # Strip markdown fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # Step 1: Parse JSON (with truncation repair fallback)
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

        # Step 2: FIX 4 — Validate structure before post-processing
        is_valid, validation_errors = _validate_blueprint(blueprint)
        if not is_valid:
            print(f"  ⚠️  Attempt {attempt}: blueprint validation failed:")
            for err in validation_errors:
                print(f"    - {err}")
            if attempt < max_attempts:
                time.sleep(2)
                continue
            # On last attempt: still try to salvage with post-processing
            print("  ⚠️  Proceeding with invalid blueprint — post-processing will patch gaps")

        # Step 3: Post-processing pipeline
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