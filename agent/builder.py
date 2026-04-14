import os
import time
import json
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import write_file, read_file, execute_python_code

MEMORY_ENABLED = False
def query_experience(desc): return ""
def add_experience(desc, code, error=None): pass

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# GLOBAL RATE LIMITER
# ═══════════════════════════════════════════════════════════════
_RATE_LIMIT_LOCK = threading.Lock()
_LAST_REQUEST_TIME = 0
_MIN_INTERVAL = 2.0

def _throttle():
    global _LAST_REQUEST_TIME
    with _RATE_LIMIT_LOCK:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _LAST_REQUEST_TIME)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_TIME = time.time()

# ═══════════════════════════════════════════════════════════════
# PROVIDER HEALTH TRACKER
# ═══════════════════════════════════════════════════════════════
class ProviderHealth:
    def __init__(self):
        self.failures = defaultdict(int)
        self.blocked_until = defaultdict(lambda: datetime.min)
        self._lock = threading.Lock()

    def block(self, name, seconds=90):
        with self._lock:
            self.failures[name] += 1
            self.blocked_until[name] = datetime.now() + timedelta(seconds=seconds)

    def ok(self, name):
        with self._lock:
            self.failures[name] = 0
            self.blocked_until[name] = datetime.min

    def is_available(self, name):
        with self._lock:
            return datetime.now() >= self.blocked_until[name]

    def reset_all(self):
        with self._lock:
            self.failures.clear()
            self.blocked_until.clear()

_health = ProviderHealth()

groq1   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2   = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3   = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter  = OpenAI(base_url="https://openrouter.ai/api/v1",      api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword  = OpenAI(base_url="https://api.doubleword.ai/v1",       api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.3-70b",       "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",       "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",       "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.1-8b",        "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.1-8b",        "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.1-8b",        "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "OpenRouter / llama-3.3-70b",   "call": lambda msgs, mt: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "OpenRouter / gemma-3-27b",     "call": lambda msgs, mt: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free",  messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-pro",    "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-pro",       messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-pro",    "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-pro",       messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-pro",    "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-pro",       messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-35B",     "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",   messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",    "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.15, max_tokens=mt)},
]

UI_PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-pro",       messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-pro",       messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-pro",       messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.3-70b",      "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",      "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",      "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",   "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.2, max_tokens=mt)},
]

# ─────────────────────────────────────────────
# CHANGE 1: TOKEN OPTIMIZER — increased sizes
# components: 2500→3500 | routes: 2000→2500 | api.js: 1500→2000
# ─────────────────────────────────────────────
def get_optimal_tokens(file_path):
    if file_path.endswith(".env.example"):
        return 600
    elif file_path.endswith("config.py"):
        return 800
    elif file_path.endswith("index.css"):
        return 600
    elif file_path.endswith("package.json"):
        return 800
    elif file_path.endswith("index.html"):
        return 1000
    elif "components/" in file_path and file_path.endswith((".js", ".jsx")):
        return 3500  # ↑ was 2500 — more room for full Tailwind UI patterns
    elif "routes.py" in file_path:
        return 2500  # ↑ was 2000 — full CRUD + pagination without truncation
    elif "App.js" in file_path:
        return 1800
    elif "api.js" in file_path:
        return 2000  # ↑ was 1500 — all endpoints + interceptors fit cleanly
    elif "models.py" in file_path:
        return 1800
    elif file_path.endswith(".py"):
        return 1500
    elif file_path.endswith("index.js"):
        return 600
    else:
        return 1200


def call_llm(messages, max_tokens=4096, task_type="general"):
    provider_list = UI_PROVIDERS if task_type == "ui" else PROVIDERS
    last_error = None

    available = [p for p in provider_list if _health.is_available(p['name'])]
    if not available:
        print("  ⚠️  All providers in cooldown — resetting and retrying...")
        _health.reset_all()
        available = provider_list

    for provider in available:
        try:
            print(f"  🤖 Using {provider['name']}...")
            _throttle()
            response = provider["call"](messages, max_tokens)
            _health.ok(provider['name'])
            return response

        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "quota", "503", "402", "temporarily", "overloaded", "upstream"]):
                cooldown = 120 if "gemini" in provider['name'].lower() else 60
                _health.block(provider['name'], seconds=cooldown)
                print(f"  ⚠️  {provider['name']} rate limited (cooldown {cooldown}s), skipping...")
                last_error = e
                continue
            elif any(x in err for x in ["decommission", "deprecated", "no longer supported", "invalid model"]):
                _health.block(provider['name'], seconds=600)
                print(f"  ⚠️  {provider['name']} model unavailable, skipping...")
                last_error = e
                continue
            else:
                print(f"  ⚠️  {provider['name']} error: {str(e)[:80]}, trying next...")
                last_error = e
                continue

    raise Exception(f"All providers failed. Last error: {last_error}")


# ═══════════════════════════════════════════════════════════════
# CHANGE 2: SPLIT PROMPTS
# BACKEND_PROMPT: Flask/Python only — no UI rules, ~3200 tokens
# FRONTEND_PROMPT: React/Tailwind only — no Flask rules, ~3500 tokens
# Previously a single BUILDER_PROMPT at ~6400 tokens input each call
# ═══════════════════════════════════════════════════════════════

BACKEND_PROMPT = """
You are a senior Flask/Python engineer. Write production-grade backend code.

ABSOLUTE RULES:
1. Output ONLY raw code. No explanation, no markdown, no backticks.
2. Every file must be 100% complete — no placeholders, no "TODO", no "pass".
3. All imports must be correct and match the actual file structure.

⛔ HARD STOPS — INVALID if these appear:
❌ from backend.api import anything — backend/api.py does not exist. Call db/models directly.
❌ db = SQLAlchemy() in app.py or models.py — ONLY backend/__init__.py defines db and jwt.
❌ Duplicate db.Index() names — every index name must be globally unique across all models.
❌ db.Index() inside a class body — ALWAYS place OUTSIDE and AFTER class definitions.

For backend/__init__.py:
- ONLY: db = SQLAlchemy() and jwt = JWTManager() — nothing else.

For backend/config.py:
- Load all from os.environ.get() with fallbacks.
- Include: SECRET_KEY, DATABASE_URL (fallback sqlite), DEBUG, JWT_SECRET_KEY.
- JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24). SQLALCHEMY_TRACK_MODIFICATIONS = False.

For backend/models.py:
- from backend import db  (NEVER redefine db here)
- db.relationship() for EVERY foreign key.
- password hashing via werkzeug.
- created_at on every model, to_dict() with .isoformat() for datetimes.
- db.Index() OUTSIDE and AFTER class definitions. Names MUST be unique: ix_post_user_id not ix_user_id.
- Every model: id (PK), created_at, to_dict(), __repr__().
- NEVER store plain text passwords.

For backend/routes.py:
- from backend import db (NEVER from backend.api import anything)
- from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
- Every POST/PUT validates required fields → 400 with clear message if missing.
- Login: accepts username + password, returns {"token": create_access_token(identity=user.id), "user": user.to_dict()}
- All GET list endpoints: ?page=1&per_page=20 with .paginate(), return {"items":[...], "total":n, "page":n, "pages":n}
- Wrap all db writes in try/except with db.session.rollback() on error.
- @jwt_required() on all write endpoints.

For backend/app.py:
- from backend import db, jwt (NEVER redefine db = SQLAlchemy() here)
- Factory: def create_app()
- Extensions: db.init_app(app), jwt.init_app(app), CORS(app), Migrate(app, db)
- Register blueprint from backend.routes.
- /health route → {"status": "ok"}
- from flask import Flask, jsonify
"""

FRONTEND_PROMPT = """
You are a senior React engineer. Write production-grade frontend code using Tailwind CSS.

ABSOLUTE RULES:
1. Output ONLY raw code. No explanation, no markdown, no backticks.
2. Every file must be 100% complete — no placeholders, no "TODO".
3. NEVER truncate. Write every function, every JSX block, every handler to completion.

⛔ HARD STOPS — INVALID if these appear:
❌ fetch('/api/...') — NEVER raw fetch. ALWAYS use named imports from '../api'.
❌ import './ComponentName.css' — NEVER per-component CSS. Tailwind only.
❌ ReactDOM.render() — NEVER React 17. Use ReactDOM.createRoot() only.
❌ <BrowserRouter> in index.js — BrowserRouter lives in App.js only.
❌ style={{ ... }} — NEVER inline styles. Tailwind className only.
❌ TODO / placeholder — NEVER stubs.
❌ import LoadingSpinner / import ErrorAlert — NEVER import components not in blueprint.

CORRECT PATTERNS:
✅ API calls: import { getRooms } from '../api'; then call in useEffect.
✅ React 18: const root = ReactDOM.createRoot(document.getElementById('root')); root.render(...)
✅ Errors: error.response?.data?.message || error.message || 'Something went wrong'
✅ useEffect: ALWAYS with dependency array [].

CRITICAL — AUTH FIELD FORMAT:
- Login form MUST use 'username' and 'password' fields ONLY.
- DO NOT use 'email' as the login credential field.
- Backend login endpoint expects: { username, password }
- api.js loginUser function must send: { username, password } NOT { email, password }
- Register form uses: { username, email, password } (email is only for registration, not login)

For frontend/src/index.js:
- React 18 createRoot. No BrowserRouter here. Only imports: react, react-dom/client, ./index.css, ./App.

For frontend/src/App.js:
- React Router v6: BrowserRouter, Routes, Route.
- Export default function App. Routes for every page.
- ONLY import components explicitly in blueprint files array.
- Protected routes: <Route element={<PrivateRoute />}> wrapping children.

For frontend/src/api.js:
- axios with baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000'
- Request interceptor: inject Authorization Bearer token from localStorage.
- Response interceptor: on 401, clear localStorage and redirect to /login.
- Named exports only — one async function per API endpoint.
- MUST include: interceptors block + ALL endpoint functions. Do not stop early.

For components/:
- Every useEffect MUST have [].
- Navbar: check localStorage.getItem('token') BEFORE calling getUser(). Skip if no token.
- Forms: controlled inputs + onChange + onSubmit with preventDefault().
- After POST/PUT/DELETE: refresh data list automatically.
- ONLY import from: react, react-router-dom, ../api.

COMPLETE COMPONENT STRUCTURE (follow this exactly for every component):
1. import statements (react, react-router-dom, ../api)
2. const ComponentName = () => {
3.   useState declarations
4.   useEffect with fetch + dependency array []
5.   handler functions (handleSubmit, handleDelete, etc.)
6.   return ( ... complete JSX ... )
7. }
8. export default ComponentName;
(Steps 1-8 ALL required. File is invalid if any step is missing.)

UI DESIGN TOKENS:
- Page wrapper: min-h-screen bg-gray-50 py-8 px-4
- Card: bg-white rounded-2xl shadow-md hover:shadow-xl transition p-6
- Button primary: bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-6 py-3 rounded-xl transition
- Button secondary: bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold px-6 py-3 rounded-xl transition
- Input: w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition
- Loading: <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent"></div>
- Error: bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm
- Empty state: bg-white rounded-2xl shadow p-16 text-center with SVG icon

NAVBAR (dark sticky):
  <nav className="bg-gray-900 sticky top-0 z-50 shadow-lg">
    <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-16">
      <Link to="/" className="text-white font-bold text-xl">AppName</Link>
      {token ? (logout button) : (login + signup links)}
    </div>
  </nav>

HOME (gradient hero + feature cards):
  Hero: bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500, text-5xl font-bold
  Features: grid grid-cols-1 md:grid-cols-3 gap-8, bg-white rounded-2xl shadow-md p-8

LOGIN/REGISTER (centered card):
  min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50, max-w-md, rounded-2xl shadow-xl p-8

LIST pages: max-w-7xl, grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6
FORM pages: max-w-2xl, bg-white rounded-2xl shadow-md p-8

✅ REQUIRED in every component: gradient hero on Home, hover effects, loading spinner, empty state with SVG, error styled red-50.
"""


def build_file(file_info, blueprint, project_path, existing_files={}):
    """Builds a single file. Uses BACKEND_PROMPT or FRONTEND_PROMPT based on file type."""

    file_path = file_info["path"]
    file_description = file_info["description"]
    depends_on = file_info.get("depends_on", [])

    print(f"\n📝 Building: {file_path}")

    # ── Template injection — guaranteed-correct skeletons ──
    SKELETON_TEMPLATES = {
        # Fix: .env.example is always the same — never waste LLM retries on it
        ".env.example": (
            "DATABASE_URL=postgresql://user:password@localhost/dbname\n"
            "SECRET_KEY=your-secret-key-here\n"
            "JWT_SECRET_KEY=your-jwt-secret-here\n"
            "DEBUG=True\n"
            "FLASK_ENV=development\n"
            "REACT_APP_API_URL=http://localhost:5000\n"
        ),
        "backend/__init__.py": (
            "from flask_sqlalchemy import SQLAlchemy\n"
            "from flask_jwt_extended import JWTManager\n\n"
            "db = SQLAlchemy()\n"
            "jwt = JWTManager()\n"
        ),
        "frontend/src/index.js": (
            "import React from 'react';\n"
            "import ReactDOM from 'react-dom/client';\n"
            "import './index.css';\n"
            "import App from './App';\n\n"
            "const root = ReactDOM.createRoot(document.getElementById('root'));\n"
            "root.render(\n"
            "  <React.StrictMode>\n"
            "    <App />\n"
            "  </React.StrictMode>\n"
            ");\n"
        ),
        "frontend/src/components/PrivateRoute.js": (
            "import React from 'react';\n"
            "import { Navigate, Outlet } from 'react-router-dom';\n\n"
            "const PrivateRoute = () => {\n"
            "  const token = localStorage.getItem('token');\n"
            "  return token ? <Outlet /> : <Navigate to=\"/login\" replace />;\n"
            "};\n\n"
            "export default PrivateRoute;\n"
        ),
    }

    if file_path in SKELETON_TEMPLATES:
        code = SKELETON_TEMPLATES[file_path]
        full_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
        print(f"  ✅ {file_path} written from template (guaranteed correct)")
        return code

    # ── File-aware context ──
    file_aware_context = ""
    if "frontend/src/api.js" in existing_files:
        import re as _re
        api_exports = _re.findall(
            r"^export\s+(?:const|async function|function)\s+(\w+)",
            existing_files["frontend/src/api.js"],
            _re.MULTILINE
        )
        if api_exports:
            file_aware_context += f"\nAPI functions available in '../api': {', '.join(api_exports)}"
            file_aware_context += "\nIMPORTANT: Import and use ONLY these exact function names — do NOT invent new ones."

    built_components = [
        os.path.basename(p).replace(".js", "")
        for p in existing_files
        if p.startswith("frontend/src/components/") and p.endswith(".js")
    ]
    if built_components:
        file_aware_context += f"\nComponents already built: {', '.join(built_components)}"
        file_aware_context += "\nIMPORTANT: Only import components from this list."

    dependency_context = ""
    for dep in depends_on:
        if dep in existing_files:
            dependency_context += f"\n\n--- {dep} ---\n{existing_files[dep]}"

    past_experience = query_experience(file_description)
    memory_context = ""
    if past_experience and past_experience[0]:
        memory_context = "\nPAST SIMILAR CODE (use as reference):\n" + "\n".join(past_experience[0])

    user_prompt = f"""
Project: {blueprint['description']}
Stack: {blueprint['stack']}
Database tables: {json.dumps(blueprint['database_schema']['tables'], indent=2)}
API endpoints: {json.dumps(blueprint['api_endpoints'], indent=2)}

File to write: {file_path}
Purpose: {file_description}
{file_aware_context}

Dependencies already written:
{dependency_context if dependency_context else "None"}

{memory_context}

Write the COMPLETE, PRODUCTION-READY code for {file_path} now.
No placeholders, no TODOs, no stubs. Real working code only.
"""

    # CHANGE 2: Route to the correct prompt based on file type
    is_backend = file_path.startswith("backend/") and file_path.endswith(".py")
    is_frontend = file_path.startswith("frontend/") and file_path.endswith((".js", ".jsx", ".css", ".html"))

    if is_backend:
        system_prompt = BACKEND_PROMPT
        task_type = "general"
    else:
        system_prompt = FRONTEND_PROMPT
        task_type = "ui"

    max_tokens = get_optimal_tokens(file_path)

    if is_frontend:
        print(f"  🎨 Using UI pipeline with {max_tokens} tokens")

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    last_error = ""
    final_code = ""
    failure_patterns = []

    for attempt in range(1, 4):
        if attempt > 1:
            print(f"  🔄 Retry attempt {attempt}...")
            if failure_patterns:
                failure_hint = "\n\nPREVIOUS ATTEMPT FAILURES (do NOT repeat these):\n" + "\n".join(
                    f"- {p}" for p in failure_patterns
                )
                history[-1]["content"] = user_prompt + failure_hint

        try:
            response = call_llm(history, max_tokens=max_tokens, task_type=task_type)
        except Exception as e:
            print(f"  ❌ All providers failed: {str(e)[:120]}")
            return final_code

        if not response or not response.choices:
            failure_patterns.append("model returned empty response — output complete code")
            continue
        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            failure_patterns.append("model returned blank content — output complete working code")
            continue
        code = raw.strip()

        code = (code.replace("```python", "").replace("```javascript", "")
                .replace("```jsx", "").replace("```css", "")
                .replace("```json", "").replace("```html", "")
                .replace("```", "").strip())

        if len(code) < 50:
            failure_patterns.append(f"output was too short ({len(code)} chars) — write the COMPLETE file")
            continue

        # ──────────────────────────────────────────────
        # CHANGE 3: TRUNCATION DETECTION for .js files
        # Before writing, validate the file isn't cut off
        # ──────────────────────────────────────────────
        if file_path.endswith(".js") and "components/" in file_path:
            truncation_issues = []

            if "export default" not in code:
                truncation_issues.append("missing 'export default' — component is truncated")
            if "return (" not in code and "return(" not in code:
                truncation_issues.append("missing 'return (' — JSX block is truncated")

            if truncation_issues:
                component_name = os.path.basename(file_path).replace(".js", "")
                print(f"  ❌ Truncation detected: {', '.join(truncation_issues)}")
                failure_patterns.append(
                    f"TRUNCATED OUTPUT — the file was cut off before completion. "
                    f"You MUST write all 9 steps: "
                    f"1) imports 2) const {component_name} = () => {{ 3) useState 4) useEffect with [] "
                    f"5) handlers 6) return ( 7) complete JSX 8) closing }} 9) export default {component_name}; "
                    f"Do NOT stop before step 9."
                )
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"TRUNCATED — {' | '.join(truncation_issues)}. Write the COMPLETE file including export default and return ()."})
                continue

        # CHANGE 4: api.js specific truncation check
        if file_path == "frontend/src/api.js":
            api_issues = []

            if "interceptors" not in code:
                api_issues.append("missing axios interceptors block")
            if "export" not in code:
                api_issues.append("missing exported functions")

            # Check that functions exist for each blueprint endpoint
            import re as _re
            export_count = len(_re.findall(r"^export\s+const\s+\w+", code, _re.MULTILINE))
            endpoint_count = len(blueprint.get("api_endpoints", []))
            if export_count < max(2, endpoint_count - 2):  # allow up to 2 missing
                api_issues.append(f"only {export_count} exported functions but {endpoint_count} endpoints defined — likely truncated")

            if api_issues:
                print(f"  ❌ api.js truncation detected: {', '.join(api_issues)}")
                failure_patterns.append(
                    f"api.js is INCOMPLETE: {', '.join(api_issues)}. "
                    f"api.js MUST contain: 1) axios instance with baseURL "
                    f"2) request interceptor (JWT injection) "
                    f"3) response interceptor (401 → clear localStorage + redirect) "
                    f"4) one exported async function per API endpoint ({endpoint_count} total). "
                    f"Write all sections completely."
                )
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"api.js is incomplete: {' | '.join(api_issues)}. Write the FULL api.js with interceptors AND all {endpoint_count} endpoint functions."})
                continue

        # Write file to project
        full_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)

        if file_path.endswith(".py"):
            result = execute_python_code(f"import ast\nast.parse(open('{full_path}').read())\nprint('syntax ok')")
            if "syntax ok" in result["stdout"]:
                print(f"  ✅ {file_path} built successfully")
                final_code = code
                add_experience(file_description, code, error=last_error)
                return code
            else:
                last_error = result["stderr"]
                print(f"  ❌ Syntax error: {last_error[:100]}")
                failure_patterns.append(f"syntax error: {last_error[:150]}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"Syntax error:\n{last_error}\n\nFix and output the complete corrected file."})
        else:
            print(f"  ✅ {file_path} built successfully")
            final_code = code
            return code

    print(f"  ⚠️  Could not fix {file_path} after 3 attempts")
    return final_code


def build_project(blueprint, output_dir="sandbox/projects", on_file_start=None, on_file_done=None):
    """Builds an entire project from a blueprint — sequential for stability."""

    project_name = blueprint["project_name"]
    project_path = os.path.join(output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"\n🚀 Building project: {project_name}")
    print(f"📁 Output: {project_path}")

    existing_files = {}
    failed_files = []
    files = blueprint["files"]

    def get_waves(files):
        completed = set()
        waves = []
        remaining = list(files)
        while remaining:
            wave = []
            still_remaining = []
            for f in remaining:
                deps = f.get("depends_on", [])
                if all(d in completed for d in deps):
                    wave.append(f)
                else:
                    still_remaining.append(f)
            if not wave:
                wave = still_remaining
                still_remaining = []
            for f in wave:
                completed.add(f["path"])
            waves.append(wave)
            remaining = still_remaining
        return waves

    waves = get_waves(files)
    print(f"⚡ Building in {len(waves)} wave(s) — SEQUENTIAL mode (stable on free tier)")

    for wave_idx, wave in enumerate(waves):
        print(f"\n🌊 Wave {wave_idx + 1}/{len(waves)}: {len(wave)} file(s)")

        for file_info in wave:
            if on_file_start:
                on_file_start(file_info["path"])
            code = build_file(
                file_info=file_info,
                blueprint=blueprint,
                project_path=project_path,
                existing_files=existing_files
            )
            if code:
                existing_files[file_info["path"]] = code
                if on_file_done:
                    on_file_done(file_info["path"], success=True)
            else:
                failed_files.append(file_info["path"])
                if on_file_done:
                    on_file_done(file_info["path"], success=False)

        if wave_idx < len(waves) - 1:
            time.sleep(3)

    # ── TEST + DEBUG LOOP ──
    print(f"\n{'='*50}")
    print("🧪 RUNNING TESTER + DEBUGGER...")
    print(f"{'='*50}")

    try:
        from agent.tester import run_tests, format_errors_for_log
        from agent.debugger import run_debug_loop

        existing_files, final_test_result, attempts = run_debug_loop(
            files=existing_files,
            tester_fn=run_tests,
            max_retries=3
        )

        for file_path, code in existing_files.items():
            full_path = os.path.join(project_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(code)

        print(f"\n🔬 Test result after {attempts} debug attempt(s): {final_test_result['summary']}")

    except Exception as e:
        print(f"\n⚠️  Tester/Debugger failed: {e} — continuing with unvalidated build")

    with open(os.path.join(project_path, "requirements.txt"), "w") as f:
        f.write("flask\nflask-cors\nflask-sqlalchemy\nflask-jwt-extended\nflask-migrate\nsqlalchemy\npsycopg2-binary\npython-dotenv\nwerkzeug\n")
    print("\n📄 requirements.txt written")

    readme = f"""# {project_name.replace('_', ' ').title()}

{blueprint['description']}

## Stack
- Frontend: {blueprint['stack']['frontend']}
- Backend: {blueprint['stack']['backend']}
- Database: {blueprint['stack']['database']}

## Prerequisites
- Python 3.9+, Node.js 16+, PostgreSQL

## Setup

### Backend
```bash
pip install -r requirements.txt
cp .env.example .env
flask db init && flask db migrate && flask db upgrade
python -m backend.app
```

### Frontend
```bash
cd frontend && npm install && npm start
```

## Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL connection string | postgresql://user:pass@localhost/dbname |
| SECRET_KEY | Flask secret key | your-secret-key |
| JWT_SECRET_KEY | JWT signing key | your-jwt-secret |
| DEBUG | Debug mode | True |
| FLASK_ENV | Flask environment | development |
| REACT_APP_API_URL | Backend URL for React | http://localhost:5000 |

## API Endpoints
"""
    for endpoint in blueprint.get("api_endpoints", []):
        if isinstance(endpoint, dict):
            auth = "🔒" if endpoint.get("auth_required") else "🔓"
            readme += f"| {endpoint.get('method','GET')} | {endpoint.get('path','/')} | {endpoint.get('description','')} | {auth} |\n"

    with open(os.path.join(project_path, "README.md"), "w") as f:
        f.write(readme)
    print("📄 README.md written")

    print(f"\n{'='*50}")
    print(f"✅ Project built: {len(existing_files)}/{len(files)} files")
    if failed_files:
        print(f"⚠️  Failed files: {failed_files}")
    print(f"📁 Location: {project_path}")

    return project_path, existing_files, failed_files


if __name__ == "__main__":
    from agent.planner import generate_blueprint
    blueprint = generate_blueprint(
        "A simple e-commerce store where users can browse products, add to cart, and place orders"
    )
    if blueprint:
        project_path, built, failed = build_project(blueprint)
        print(f"\nBuilt files: {list(built.keys())}")