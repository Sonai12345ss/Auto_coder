import os
import time
import json
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import write_file, read_file, execute_python_code

MEMORY_ENABLED = False
def query_experience(desc): return ""
def add_experience(desc, code, error=None): pass

load_dotenv()

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

groq1      = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2      = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3      = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",       api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1",        api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

# Interleaved provider order — prevents same-quota exhaustion
PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.3-70b",       "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",       "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",       "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "OpenRouter / llama-3.3-70b",   "call": lambda msgs, mt: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "OpenRouter / gemma-3-27b",     "call": lambda msgs, mt: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free",  messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.1-8b",        "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.1-8b",        "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.1-8b",        "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-35B",     "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",   messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",    "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.15, max_tokens=mt)},
]

UI_PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.2, max_tokens=mt)},
]

# ═══════════════════════════════════════════════════════════════
# CRITICAL FILE SYSTEM
# ═══════════════════════════════════════════════════════════════
CRITICAL_FILES = {
    "frontend/src/App.js",
    "frontend/src/index.js",
    "backend/app.py",
    "backend/routes.py",
    "backend/models.py",
    "backend/__init__.py",
}

# If key fails, values are marked invalid and skipped
DEPENDENCY_CASCADE = {
    "frontend/src/App.js":  ["frontend/src/index.js"],
    "backend/__init__.py":  ["backend/models.py", "backend/routes.py", "backend/app.py"],
    "backend/models.py":    ["backend/routes.py", "backend/app.py"],
    "backend/config.py":    ["backend/models.py", "backend/app.py"],
}

# Minimal but guaranteed-valid fallback templates for critical files
FALLBACK_TEMPLATES = {
    "frontend/src/App.js": """import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={
            <div className="flex items-center justify-center min-h-screen">
              <div className="text-center">
                <h1 className="text-4xl font-bold text-gray-900 mb-4">Welcome</h1>
                <p className="text-gray-500">App is loading. Please refresh.</p>
              </div>
            </div>
          } />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
""",
    "frontend/src/index.js": """import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""",
    "backend/__init__.py": """from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
jwt = JWTManager()
""",
    "backend/app.py": """from flask import Flask, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from backend import db, jwt
from backend.routes import routes

def create_app():
    app = Flask(__name__)
    app.config.from_object('backend.config.Config')
    db.init_app(app)
    jwt.init_app(app)
    Migrate(app, db)
    CORS(app)
    app.register_blueprint(routes)

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
""",
}

# ─────────────────────────────────────────────
# TOKEN ALLOCATION
# ─────────────────────────────────────────────
def get_optimal_tokens(file_path):
    if file_path.endswith("config.py") or file_path.endswith(".env.example"):
        return 1000
    elif file_path.endswith("package.json") or file_path.endswith("index.html"):
        return 1200
    elif file_path.endswith("index.css"):
        return 800
    elif "components/" in file_path and file_path.endswith((".js", ".jsx")):
        return 2500
    elif "routes.py" in file_path:
        return 2200
    elif "App.js" in file_path or "api.js" in file_path:
        return 2000
    elif "models.py" in file_path:
        return 1800
    elif file_path.endswith(".py"):
        return 1500
    else:
        return 1500

# ─────────────────────────────────────────────
# SAFE RESPONSE EXTRACTION
# ─────────────────────────────────────────────
def extract_code(response):
    try:
        if not response or not response.choices:
            return None
        msg = response.choices[0].message
        if not msg or not msg.content:
            return None
        content = msg.content.strip()
        return content if content else None
    except Exception as e:
        print(f"  ⚠️  Response extraction error: {e}")
        return None

# ─────────────────────────────────────────────
# CODE QUALITY VALIDATION
# ─────────────────────────────────────────────
def is_bad_code(code, file_path):
    if not code or len(code.strip()) < 50:
        return True, "too short or empty"
    for s in ["# TODO", "# FIXME", "pass  # placeholder", "raise NotImplementedError"]:
        if s in code:
            return True, f"contains stub: {s}"
    return False, "ok"

# ─────────────────────────────────────────────
# JS STRUCTURAL VALIDATION
# ─────────────────────────────────────────────
def validate_js_structure(code, file_path):
    if not code or len(code.strip()) < 30:
        return False, "empty or too short"
    if not ("export" in code or "function" in code or "const " in code):
        return False, "no export, function, or const found"
    if "App.js" in file_path:
        if "export default" not in code:
            return False, "App.js missing 'export default'"
        if "function App" not in code and "const App" not in code and "App =" not in code:
            return False, "App.js missing App component definition"
        if "BrowserRouter" not in code and "Routes" not in code:
            return False, "App.js missing routing"
    if "index.js" in file_path and "components/" not in file_path:
        if "createRoot" not in code:
            return False, "index.js missing createRoot"
    if "api.js" in file_path and "axios" not in code:
        return False, "api.js missing axios"
    if "PrivateRoute.js" in file_path and "localStorage" not in code:
        return False, "PrivateRoute.js missing localStorage check"
    return True, "ok"

# ─────────────────────────────────────────────
# SMART RETRY PROMPTS — different strategy per attempt
# ─────────────────────────────────────────────
def get_retry_prompt(file_path, attempt, base_prompt, last_error=""):
    """Progressively stricter prompts for each retry attempt."""
    if attempt == 0:
        return base_prompt

    if attempt == 1:
        # Attempt 2: explicit structure requirements per file type
        additions = ""
        if "App.js" in file_path:
            additions = """
STRICT REQUIREMENTS (retry):
- MUST export a default function named App
- MUST include BrowserRouter, Routes, Route from react-router-dom
- MUST return valid JSX
- MUST end with: export default App;
- Do NOT leave any section empty"""
        elif "routes.py" in file_path:
            additions = """
STRICT REQUIREMENTS (retry):
- MUST define Blueprint named 'routes'
- MUST include /api/register, /api/login, /api/user routes
- MUST import create_access_token from flask_jwt_extended
- Every route MUST return jsonify(...)"""
        elif "models.py" in file_path:
            additions = """
STRICT REQUIREMENTS (retry):
- MUST import db from backend
- MUST define all models as db.Model subclasses
- Every model MUST have id, created_at, to_dict(), __repr__
- db.Index() calls MUST be OUTSIDE class definitions"""
        elif "index.js" in file_path and "components/" not in file_path:
            additions = """
STRICT REQUIREMENTS (retry):
- MUST use ReactDOM.createRoot (React 18)
- MUST NOT wrap App in BrowserRouter
- MUST NOT use ReactDOM.render()"""
        return base_prompt + additions

    elif attempt >= 2:
        # Attempt 3+: simplest possible correct version
        return base_prompt + f"""
FINAL ATTEMPT — Write the SIMPLEST possible working version.
Previous failure: {last_error}
- Prioritize correctness over completeness
- Every function must work even if simplified
- No placeholders, no TODO, no empty functions
- Must be immediately runnable"""

    return base_prompt

# ─────────────────────────────────────────────
# MAIN LLM CALLER — two-pass fallback
# ─────────────────────────────────────────────
def call_llm(messages, max_tokens=4096, task_type="general"):
    full_list = (UI_PROVIDERS + PROVIDERS) if task_type == "ui" else PROVIDERS
    last_error = None

    for pass_num in range(2):
        for provider in full_list:
            name = provider["name"]
            if pass_num == 0 and not _health.is_available(name):
                continue
            try:
                print(f"  🤖 Using {name}...")
                response = provider["call"](messages, max_tokens)
                _health.ok(name)
                return response
            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["rate_limit", "rate-limit", "429", "quota",
                                           "503", "402", "temporarily", "overloaded", "upstream"]):
                    cooldown = 120 if "gemini" in name.lower() else 60
                    _health.block(name, seconds=cooldown)
                    print(f"  ⚠️  {name} rate limited, skipping...")
                    last_error = e
                elif any(x in err for x in ["decommission", "deprecated", "invalid model",
                                             "model not found", "404", "400"]):
                    _health.block(name, seconds=3600)
                    print(f"  ⚠️  {name} unavailable, skipping...")
                    last_error = e
                else:
                    print(f"  ⚠️  {name}: {str(e)[:80]}, trying next...")
                    last_error = e

        if pass_num == 0:
            print("  ⏳ All preferred providers busy — second pass...")
            _health.reset_all()

    raise Exception(f"All providers exhausted. Last error: {last_error}")


BUILDER_PROMPT = """
You are a senior full stack engineer with 10+ years of experience. You write production-grade code that is secure, maintainable, and complete. Your job is to write a single file as part of a larger project.

ABSOLUTE RULES:
1. Output ONLY raw code. Zero explanation, zero markdown, zero backticks.
2. Every file must be 100% complete — no placeholders, no "TODO", no "pass", no "..." ellipsis.
3. Never write stub functions. Every function must have real, working logic.
4. Stay consistent with the blueprint: use exact model names, field names, and endpoint paths provided.
5. All imports must be correct and match the actual file structure.

═══════════════════════════════════════════
BACKEND RULES (Flask/Python)
═══════════════════════════════════════════

For backend/__init__.py:
- ONLY these two lines of content (plus imports):
  db = SQLAlchemy()
  jwt = JWTManager()
- Imports: from flask_sqlalchemy import SQLAlchemy and from flask_jwt_extended import JWTManager

For backend/config.py:
- Load all config from os.environ.get()
- Include: SECRET_KEY, DATABASE_URL (fallback sqlite), DEBUG, JWT_SECRET_KEY
- JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
- SQLALCHEMY_TRACK_MODIFICATIONS = False

For backend/models.py:
- from backend import db
- db.relationship() for EVERY foreign key
- werkzeug password hashing, check_password method
- created_at on every model, to_dict() with .isoformat()
- db.Index() OUTSIDE and AFTER class definitions
- __repr__ on every model

For backend/routes.py:
- MUST import: from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
- Generate ALL endpoints from blueprint
- Login: data.get('username'), data.get('password')
- Login response: {"token": create_access_token(identity=user.id), "user": user.to_dict()}
- Pagination: ?page=1&per_page=20 → {"items":[...], "total":n, "page":n, "pages":n}
- db.session.rollback() on all write errors
- @jwt_required() on all write endpoints

For backend/app.py:
- from backend import db, jwt
- Factory: def create_app()
- db.init_app, jwt.init_app, Migrate, CORS
- /health route → {"status": "ok"}
- from flask import Flask, jsonify

═══════════════════════════════════════════
FRONTEND RULES (React)
═══════════════════════════════════════════

For frontend/src/App.js:
- React Router v6: BrowserRouter, Routes, Route
- export default function App — REQUIRED
- Routes for every page in blueprint
- PrivateRoute wrapping protected routes with Outlet pattern
- ONLY import components listed in blueprint

For frontend/src/index.js:
- React 18: ReactDOM.createRoot(document.getElementById('root'))
- NEVER BrowserRouter here, NEVER ReactDOM.render()

For frontend/src/api.js:
- axios baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000'
- JWT request interceptor
- 401 → clear localStorage, redirect to /login
- Named exports only

For PrivateRoute.js:
  import React from 'react';
  import { Navigate, Outlet } from 'react-router-dom';
  const PrivateRoute = () => {
    const token = localStorage.getItem('token');
    return token ? <Outlet /> : <Navigate to="/login" replace />;
  };
  export default PrivateRoute;

STYLING: Tailwind CSS only. No inline styles. No component CSS imports.

UI PATTERNS:
- NAVBAR: bg-gray-900 sticky top-0 z-50, max-w-7xl mx-auto
- HOME: gradient hero bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500, text-5xl, feature cards
- AUTH: min-h-screen bg-gradient-to-br from-indigo-50, max-w-md, rounded-2xl shadow-xl
- LISTS: max-w-7xl, grid grid-cols-3 gap-6, rounded-2xl cards with hover:shadow-xl
- FORMS: max-w-2xl, bg-white rounded-2xl shadow-md p-8

HOOKS:
- Every useEffect MUST have []
- Navbar: check localStorage.getItem('token') before calling getUser()
- Errors: error.response?.data?.message || error.message || 'Something went wrong'
- Loading: animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent
- Empty states with SVG icons, gradient hero on Home, hover effects everywhere

✅ Required: gradient hero, hover effects, loading states, empty states with SVG, avatar initials
❌ Forbidden: bullet lists, /api/... links in UI, flat pages, missing states
"""


def _write_file(project_path, file_path, code):
    """Write code to disk, creating directories as needed."""
    full_path = os.path.join(project_path, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(code)


def build_file(file_info, blueprint, project_path, existing_files={}):
    """
    Builds a single file.
    Critical files get 5 attempts + smart retry prompts + fallback template.
    Regular files get 3 attempts.
    """
    file_path = file_info["path"]
    file_description = file_info["description"]
    depends_on = file_info.get("depends_on", [])
    is_critical = file_path in CRITICAL_FILES

    print(f"\n📝 Building: {file_path}{' 🔴 [CRITICAL]' if is_critical else ''}")

    # Build dependency context
    dependency_context = ""
    for dep in depends_on:
        if dep in existing_files:
            dependency_context += f"\n\n--- {dep} ---\n{existing_files[dep]}"

    base_user_prompt = f"""
Project: {blueprint['description']}
Stack: {blueprint['stack']}
Database tables: {json.dumps(blueprint['database_schema']['tables'], indent=2)}
API endpoints: {json.dumps(blueprint['api_endpoints'], indent=2)}

File to write: {file_path}
Purpose: {file_description}

Dependencies already written:
{dependency_context if dependency_context else "None"}

Write the COMPLETE, PRODUCTION-READY code for {file_path} now.
No placeholders, no TODOs, no stubs. Real working code only.
"""

    is_frontend = file_path.startswith("frontend/") and file_path.endswith((".js", ".jsx", ".css", ".html"))
    task_type = "ui" if is_frontend else "general"
    max_tokens = get_optimal_tokens(file_path)

    if is_frontend:
        print(f"  🎨 UI pipeline — {max_tokens} tokens")

    # Critical files: 5 attempts. Regular files: 3 attempts.
    max_attempts = 5 if is_critical else 3
    last_error = ""
    final_code = ""

    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"  🔄 Retry {attempt + 1}/{max_attempts}{' 🔴' if is_critical else ''}...")

        # Smart retry: escalating prompt strategy
        user_prompt = get_retry_prompt(file_path, attempt, base_user_prompt, last_error)
        history = [
            {"role": "system", "content": BUILDER_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM
        try:
            response = call_llm(history, max_tokens=max_tokens, task_type=task_type)
        except Exception as e:
            print(f"  ❌ All providers failed: {str(e)[:120]}")
            last_error = str(e)
            break  # Fall through to fallback template logic below

        # Safe extraction
        code = extract_code(response)
        if code is None:
            print(f"  ❌ Empty/invalid response")
            last_error = "empty response"
            continue

        # Clean backticks
        code = (code
                .replace("```python", "").replace("```javascript", "")
                .replace("```jsx", "").replace("```css", "")
                .replace("```json", "").replace("```html", "")
                .replace("```", "").strip())

        # Quality check
        bad, reason = is_bad_code(code, file_path)
        if bad:
            print(f"  ❌ Quality check failed: {reason}")
            last_error = reason
            continue

        # Validate Python syntax
        if file_path.endswith(".py"):
            _write_file(project_path, file_path, code)
            full_path = os.path.join(project_path, file_path)
            result = execute_python_code(f"import ast\nast.parse(open('{full_path}').read())\nprint('syntax ok')")
            if "syntax ok" in result["stdout"]:
                print(f"  ✅ {file_path} built successfully")
                return code
            else:
                last_error = result["stderr"]
                print(f"  ❌ Syntax error: {last_error[:120]}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"Syntax error:\n{last_error}\nFix and return complete file."})
                final_code = code  # keep as best-effort
                continue

        # Validate JS structure
        elif file_path.endswith((".js", ".jsx")):
            valid, reason = validate_js_structure(code, file_path)
            if not valid:
                print(f"  ❌ JS validation failed: {reason}")
                last_error = reason
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"Validation failed: {reason}. Return the complete corrected file."})
                final_code = code  # keep as best-effort
                continue
            _write_file(project_path, file_path, code)
            print(f"  ✅ {file_path} built successfully")
            return code

        # Other files (CSS, HTML, JSON) — accept as-is
        else:
            _write_file(project_path, file_path, code)
            print(f"  ✅ {file_path} built successfully")
            return code

    # ── All attempts exhausted ──
    print(f"  ⚠️  Could not fix {file_path} after {max_attempts} attempts")

    # CRITICAL FILE PROTECTION: use fallback template
    if is_critical and file_path in FALLBACK_TEMPLATES:
        print(f"  🆘 CRITICAL FILE PROTECTION — using fallback template for {file_path}")
        fallback = FALLBACK_TEMPLATES[file_path]
        _write_file(project_path, file_path, fallback)
        print(f"  ✅ {file_path} written with fallback (functional but minimal)")
        return fallback

    # Non-critical: return best-effort code if we have any
    if final_code:
        _write_file(project_path, file_path, final_code)
    return final_code


def build_project(blueprint, output_dir="sandbox/projects", on_file_start=None, on_file_done=None):
    """Builds entire project sequentially with critical file protection and cascade detection."""

    project_name = blueprint["project_name"]
    project_path = os.path.join(output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"\n🚀 Building project: {project_name}")
    print(f"📁 Output: {project_path}")

    existing_files = {}
    failed_files = []
    cascade_invalidated = set()  # Files to skip due to dependency failure
    files = blueprint["files"]

    def get_waves(files):
        completed = set()
        waves = []
        remaining = list(files)
        while remaining:
            wave, still_remaining = [], []
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
    print(f"⚡ Building {len(files)} files in {len(waves)} wave(s) — sequential + critical protection")

    for wave_idx, wave in enumerate(waves):
        print(f"\n🌊 Wave {wave_idx + 1}/{len(waves)}: {len(wave)} file(s)")

        for file_info in wave:
            fp = file_info["path"]

            # Skip files invalidated by cascade
            if fp in cascade_invalidated:
                print(f"\n  ⏭️  Skipping {fp} — dependency cascade (parent file failed)")
                failed_files.append(fp)
                if on_file_done:
                    on_file_done(fp, success=False)
                continue

            if on_file_start:
                on_file_start(fp)

            code = build_file(
                file_info=file_info,
                blueprint=blueprint,
                project_path=project_path,
                existing_files=existing_files
            )

            if code:
                existing_files[fp] = code
                if on_file_done:
                    on_file_done(fp, success=True)
            else:
                failed_files.append(fp)
                if on_file_done:
                    on_file_done(fp, success=False)

                if fp in CRITICAL_FILES:
                    print(f"\n  🔴 CRITICAL FILE FAILED: {fp} (fallback was used if available)")

                # Cascade: mark dependent files as invalid
                if fp in DEPENDENCY_CASCADE:
                    cascaded = DEPENDENCY_CASCADE[fp]
                    cascade_invalidated.update(cascaded)
                    print(f"  📉 Cascade: marking {len(cascaded)} dependent file(s) as invalid")

        if wave_idx < len(waves) - 1:
            time.sleep(2)

    # ── PHASE 2: TEST + DEBUG ──
    print(f"\n{'='*50}")
    print("🧪 RUNNING TESTER + DEBUGGER...")
    print(f"{'='*50}")

    try:
        from agent.tester import run_tests
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

    # Write requirements.txt
    with open(os.path.join(project_path, "requirements.txt"), "w") as f:
        f.write("flask\nflask-cors\nflask-sqlalchemy\nflask-jwt-extended\nflask-migrate\nsqlalchemy\npsycopg2-binary\npython-dotenv\nwerkzeug\n")
    print("\n📄 requirements.txt written")

    # Write README
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
| Variable | Description |
|----------|-------------|
| DATABASE_URL | postgresql://user:pass@localhost/dbname |
| SECRET_KEY | Flask secret key |
| JWT_SECRET_KEY | JWT signing key |
| REACT_APP_API_URL | http://localhost:5000 |
"""
    with open(os.path.join(project_path, "README.md"), "w") as f:
        f.write(readme)
    print("📄 README.md written")

    print(f"\n{'='*50}")
    print(f"✅ Project built: {len(existing_files)}/{len(files)} files")
    if failed_files:
        critical_failures = [f for f in failed_files if f in CRITICAL_FILES]
        non_critical = [f for f in failed_files if f not in CRITICAL_FILES]
        if critical_failures:
            print(f"🔴 Critical (used fallback template): {critical_failures}")
        if non_critical:
            print(f"⚠️  Non-critical failures: {non_critical}")
    print(f"📁 Location: {project_path}")

    return project_path, existing_files, failed_files


if __name__ == "__main__":
    from agent.planner import generate_blueprint
    blueprint = generate_blueprint("A simple e-commerce store where users can browse products, add to cart, and place orders")
    if blueprint:
        project_path, built, failed = build_project(blueprint)
        print(f"\nBuilt: {list(built.keys())}")