import os
import re
import time
import json
import hashlib
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import write_file, read_file, execute_python_code

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# REAL MEMORY SYSTEM
# SHORT-TERM: per-build session | LONG-TERM: persisted to disk
# ═══════════════════════════════════════════════════════════════

MEMORY_FILE = "sandbox/builder_memory.json"
MEMORY_ENABLED = True

_session_memory = {
    "provider_wins": defaultdict(int),
    "provider_latency": defaultdict(list),
    "fix_patterns": {},
    "failed_patterns": set(),
}
_session_lock = threading.Lock()

def _load_long_term_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            return json.loads(open(MEMORY_FILE).read())
    except Exception:
        pass
    return {"fix_patterns": {}, "provider_stats": {}, "build_count": 0}

def _save_long_term_memory(mem):
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem, f, indent=2)
    except Exception:
        pass

def remember_fix(error_type, file_ext, broken_snippet, fixed_snippet):
    if not broken_snippet or not fixed_snippet or broken_snippet == fixed_snippet:
        return
    key = f"{error_type}|{file_ext}"
    mem = _load_long_term_memory()
    existing = mem.setdefault("fix_patterns", {}).get(key, [])
    existing = existing[-4:] + [{"broken": broken_snippet[:300], "fixed": fixed_snippet[:300], "ts": datetime.now().isoformat()}]
    mem["fix_patterns"][key] = existing
    _save_long_term_memory(mem)

def recall_fixes(error_type, file_ext):
    key = f"{error_type}|{file_ext}"
    return _load_long_term_memory().get("fix_patterns", {}).get(key, [])

def record_provider_result(provider_name, success, latency_ms):
    with _session_lock:
        if success:
            _session_memory["provider_wins"][provider_name] += 1
        _session_memory["provider_latency"][provider_name].append(latency_ms)
    mem = _load_long_term_memory()
    p = mem.setdefault("provider_stats", {}).setdefault(provider_name, {"success": 0, "fail": 0, "latency_sum": 0, "calls": 0})
    p["calls"] += 1
    p["latency_sum"] += latency_ms
    if success:
        p["success"] += 1
    else:
        p["fail"] += 1
    _save_long_term_memory(mem)

def query_experience(description): return []
def add_experience(description, code, error=None): pass


# ═══════════════════════════════════════════════════════════════
# FILE CACHE — avoid rebuilding identical files within a session
# ═══════════════════════════════════════════════════════════════

_file_cache = {}
_cache_lock = threading.Lock()

def _make_cache_key(file_description, dependency_codes):
    combined = file_description + "".join(sorted(dependency_codes))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

def cache_get(key):
    with _cache_lock:
        return _file_cache.get(key)

def cache_set(key, code):
    with _cache_lock:
        _file_cache[key] = code


# ═══════════════════════════════════════════════════════════════
# PER-PROVIDER RATE LIMITER
# Gemini 2.5s gap, Groq 1.0s gap — each provider independent
# ═══════════════════════════════════════════════════════════════

class PerProviderRateLimiter:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            gap = self.min_interval - (now - self.last_time)
            if gap > 0:
                time.sleep(gap)
            self.last_time = time.time()

_provider_limiters = {}
_limiters_lock = threading.Lock()

def _get_limiter(provider_name):
    with _limiters_lock:
        if provider_name not in _provider_limiters:
            interval = 2.5 if "gemini" in provider_name.lower() else 1.0
            _provider_limiters[provider_name] = PerProviderRateLimiter(interval)
        return _provider_limiters[provider_name]


# ═══════════════════════════════════════════════════════════════
# SMART PROVIDER SCORING
# Ranked by: success rate (60%) + latency (40%) - fail penalty
# ═══════════════════════════════════════════════════════════════

class ProviderHealth:
    def __init__(self):
        self.consecutive_fails = defaultdict(int)
        self.blocked_until = defaultdict(lambda: datetime.min)
        self._lock = threading.Lock()

    def block(self, name, seconds=90):
        with self._lock:
            self.consecutive_fails[name] += 1
            self.blocked_until[name] = datetime.now() + timedelta(seconds=seconds)

    def ok(self, name):
        with self._lock:
            self.consecutive_fails[name] = 0
            self.blocked_until[name] = datetime.min

    def is_available(self, name):
        with self._lock:
            return datetime.now() >= self.blocked_until[name]

    def reset_all(self):
        with self._lock:
            self.consecutive_fails.clear()
            self.blocked_until.clear()

_health = ProviderHealth()

def _score_provider(provider_name):
    stats = _load_long_term_memory().get("provider_stats", {}).get(provider_name)
    if not stats or stats["calls"] == 0:
        base_score = 0.5
    else:
        success_rate = stats["success"] / stats["calls"]
        avg_latency = stats["latency_sum"] / stats["calls"]
        latency_score = max(0.0, 1.0 - (avg_latency - 500) / 4500)
        base_score = (success_rate * 0.6) + (latency_score * 0.4)
    return base_score - (_health.consecutive_fails.get(provider_name, 0) * 0.15)

def _ranked_providers(provider_list):
    available = [p for p in provider_list if _health.is_available(p["name"])]
    if not available:
        _health.reset_all()
        available = provider_list
    return sorted(available, key=lambda p: _score_provider(p["name"]), reverse=True)


# ═══════════════════════════════════════════════════════════════
# SPECIALIZED MODEL POOLS PER TASK TYPE
# ═══════════════════════════════════════════════════════════════

groq1   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2   = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3   = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",   api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1",   api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

BACKEND_PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-pro",   messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-pro",   messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-pro",   messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.3-70b",      "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",      "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",      "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",   "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.1, max_tokens=mt)},
]

UI_PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-pro",   messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-pro",   messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-pro",   messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.3-70b",      "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",      "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",      "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",   "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.2, max_tokens=mt)},
]

DEBUG_PROVIDERS = [
    {"name": "Groq-1 / llama-3.3-70b",      "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",      "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",      "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-35B",    "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",  messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",   "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.0-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.0-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.0-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.0-flash", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "OpenRouter / llama-3.3-70b",  "call": lambda msgs, mt: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash", messages=msgs, temperature=0.05, max_tokens=mt)},
]

FAST_PROVIDERS = [
    {"name": "Groq-1 / llama-3.1-8b",       "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.1-8b-instant", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.1-8b",       "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.1-8b-instant", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.1-8b",       "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.1-8b-instant", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.0-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash",   messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.0-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash",   messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "OpenRouter / gemma-3-27b",    "call": lambda msgs, mt: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free", messages=msgs, temperature=0.1, max_tokens=mt)},
]

TASK_PROVIDERS = {
    "backend": BACKEND_PROVIDERS,
    "ui":      UI_PROVIDERS,
    "debug":   DEBUG_PROVIDERS,
    "fast":    FAST_PROVIDERS,
}


def get_optimal_tokens(file_path):
    if file_path.endswith(".env.example"):          return 600
    elif file_path.endswith("config.py"):           return 800
    elif file_path.endswith("index.css"):           return 600
    elif file_path.endswith("package.json"):        return 800
    elif file_path.endswith("index.html"):          return 1000
    elif "components/" in file_path and file_path.endswith((".js", ".jsx")): return 3500
    elif "routes.py" in file_path:                 return 2500
    elif "App.js" in file_path:                    return 1800
    elif "api.js" in file_path:                    return 2000
    elif "models.py" in file_path:                 return 1800
    elif file_path.endswith(".py"):                return 1500
    elif file_path.endswith("index.js"):           return 600
    else:                                          return 1200


def _get_task_type(file_path):
    if file_path in (".env.example", "frontend/src/index.js",
                     "frontend/package.json", "frontend/public/index.html", "backend/config.py"):
        return "fast"
    elif file_path.startswith("backend/") and file_path.endswith(".py"):
        return "backend"
    elif file_path.startswith("frontend/") and file_path.endswith((".js", ".jsx", ".css", ".html")):
        return "ui"
    return "backend"


def call_llm(messages, max_tokens=4096, task_type="backend"):
    provider_list = TASK_PROVIDERS.get(task_type, BACKEND_PROVIDERS)
    ranked = _ranked_providers(provider_list)
    last_error = None

    for provider in ranked:
        name = provider["name"]
        t0 = time.time()
        try:
            print(f"  🤖 [{task_type}] Using {name} (score: {_score_provider(name):.2f})...")
            _get_limiter(name).wait()
            response = provider["call"](messages, max_tokens)
            latency_ms = (time.time() - t0) * 1000
            _health.ok(name)
            record_provider_result(name, success=True, latency_ms=latency_ms)
            print(f"  ✅ {name} responded in {latency_ms:.0f}ms")
            return response
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            record_provider_result(name, success=False, latency_ms=latency_ms)
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "quota", "503", "402", "temporarily", "overloaded", "upstream"]):
                cooldown = 120 if "gemini" in name.lower() else 60
                _health.block(name, seconds=cooldown)
                print(f"  ⚠️  {name} rate limited (cooldown {cooldown}s), skipping...")
                last_error = e
            elif any(x in err for x in ["decommission", "deprecated", "no longer supported", "invalid model"]):
                _health.block(name, seconds=600)
                print(f"  ⚠️  {name} model unavailable, skipping...")
                last_error = e
            else:
                print(f"  ⚠️  {name} error: {str(e)[:80]}, trying next...")
                last_error = e
            continue

    raise Exception(f"All providers failed for task_type={task_type}. Last error: {last_error}")


# ═══════════════════════════════════════════════════════════════
# BACKEND PROMPT
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
❌ user.password == data['password'] — NEVER compare passwords with ==. Use check_password().
❌ User(password=data['password']) — NEVER store raw passwords. Use set_password() after creation.

For backend/__init__.py:
- ONLY: db = SQLAlchemy() and jwt = JWTManager() — nothing else.

For backend/config.py:
- Load all from os.environ.get() with fallbacks.
- Include: SECRET_KEY, DATABASE_URL (fallback sqlite), DEBUG, JWT_SECRET_KEY.
- JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24). SQLALCHEMY_TRACK_MODIFICATIONS = False.

For backend/models.py:
- from backend import db  (NEVER redefine db here)
- db.relationship() for EVERY foreign key.
- password hashing via werkzeug: set_password() and check_password() on User model.
- created_at on every model, to_dict() with .isoformat() for datetimes.
- db.Index() OUTSIDE and AFTER class definitions. Names MUST be unique.
- Every model: id (PK), created_at, to_dict(), __repr__().
- FIX: ALWAYS define __tablename__ = 'table_name' explicitly on EVERY model.
  Never rely on SQLAlchemy defaults — they cause FK reference mismatches.
  class User        → __tablename__ = 'users'
  class OrderItem   → __tablename__ = 'order_items'
  class BlogPost    → __tablename__ = 'blog_posts'
  FK references MUST match __tablename__ exactly: db.ForeignKey('users.id'), db.ForeignKey('order_items.id')

For backend/routes.py:
- from backend import db (NEVER from backend.api import anything)
- from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token

REGISTRATION — correct pattern (MANDATORY):
  new_user = User(username=data['username'], email=data['email'])
  new_user.set_password(data['password'])
  db.session.add(new_user)
  db.session.commit()

LOGIN — correct pattern (MANDATORY):
  user = User.query.filter_by(username=data['username']).first()
  if user and user.check_password(data['password']):
      token = create_access_token(identity=user.id)
      return jsonify({'token': token, 'user': user.to_dict()}), 200

- Login returns: {"token": ..., "user": ...} — key MUST be "token" not "access_token".
- Every POST/PUT validates required fields → 400 if missing.
- All GET list endpoints: paginate(), return {"items":[...], "total":n, "page":n, "pages":n}
- Wrap all db writes in try/except with db.session.rollback() on error.
- @jwt_required() on all write endpoints.

For backend/app.py:
- from backend import db, jwt (NEVER redefine db = SQLAlchemy() here)
- Factory: def create_app()
- Extensions: db.init_app(app), jwt.init_app(app), CORS(app), Migrate(app, db)
- Register blueprint from backend.routes.
- /health route → {"status": "ok"}
"""

# ═══════════════════════════════════════════════════════════════
# FRONTEND PROMPT
# ═══════════════════════════════════════════════════════════════

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
❌ response.data.access_token — NEVER. Backend returns {"token": ...}. Use response.data.token.
❌ import api from '../api' — NEVER default import. api.js has no default export.
   Correct: import { getProducts, login } from '../api'

CORRECT PATTERNS:
✅ API calls: import { getProducts } from '../api'; call in useEffect with [].
✅ React 18: const root = ReactDOM.createRoot(...); root.render(...)
✅ Errors: error.response?.data?.message || error.message || 'Something went wrong'
✅ JWT save: localStorage.setItem('token', response.data.token)

CRITICAL — AUTH:
- Login uses 'username' + 'password' only (NOT email).
- Register uses username + email + password.
- Backend /api/login returns: { "token": "eyJ...", "user": {...} }

For frontend/src/App.js — PURE ROUTER:
- No API calls, no useEffect, no useState, no data fetching.
- <Navbar /> MUST be inside <BrowserRouter>.
- PrivateRoute Outlet pattern:
    <Route element={<PrivateRoute />}>
      <Route path="/x" element={<X />} />
    </Route>

For frontend/src/api.js:
- axios instance + request interceptor (JWT) + response interceptor (401→redirect)
- loginUser saves: localStorage.setItem('token', response.data.token)
- One named export per endpoint. Do not stop early.

For components/:
- Every useEffect with []. Navbar checks token before getUser().
- Forms: controlled inputs, preventDefault, refresh data after mutation.
- ONLY import from: react, react-router-dom, ../api.

COMPONENT STRUCTURE (all 8 steps required):
1. imports  2. const X = () => {  3. useState  4. useEffect+[]
5. handlers  6. return (  7. complete JSX  8. }; export default X;

UI TOKENS:
- Wrapper: min-h-screen bg-gray-50 py-8 px-4
- Card: bg-white rounded-2xl shadow-md hover:shadow-xl transition p-6
- Button: bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-6 py-3 rounded-xl transition
- Input: w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-indigo-500
- Loading: <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent"></div>
- Error: bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm
- Navbar: bg-gray-900 sticky top-0 z-50 shadow-lg, h-16
- Home hero: bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500
- Login/Register: max-w-md bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl shadow-xl p-8
"""


# ═══════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION FOR SURGICAL DEBUGGING
# ═══════════════════════════════════════════════════════════════

ERROR_BUCKETS = {
    "structural": [
        "truncated_component", "truncated_api", "critical_missing_export",
        "critical_missing_component", "critical_missing_router", "empty_file",
        "app_js_api_import", "navbar_outside_router", "private_route_wrong_pattern",
    ],
    "semantic": [
        "token_key_mismatch", "plaintext_password", "plaintext_password_check",
        "missing_set_password_call", "raw_fetch_instead_of_api", "missing_api_export",
    ],
    "syntactic": ["syntax_error", "invalid_json"],
    "import": [
        "phantom_backend_api", "duplicate_db_instance", "missing_import",
        "bad_import", "missing_component", "css_import",
    ],
    "style": [
        "missing_deps_array", "unsafe_error_access", "react17_api",
        "double_router", "wrong_tailwind_tag", "missing_tailwind",
        "missing_root_div", "misplaced_db_index", "duplicate_index_name",
    ],
    "config": [
        "missing_proxy", "missing_field", "missing_dependency",
        "missing_env_var", "missing_init",
    ],
}

def classify_errors(errors):
    buckets = defaultdict(list)
    for err in errors:
        placed = False
        for bucket, types in ERROR_BUCKETS.items():
            if err["type"] in types:
                buckets[bucket].append(err)
                placed = True
                break
        if not placed:
            buckets["structural"].append(err)
    return dict(buckets)


def build_file(file_info, blueprint, project_path, existing_files=None):
    if existing_files is None:
        existing_files = {}

    file_path = file_info["path"]
    file_description = file_info["description"]
    depends_on = file_info.get("depends_on", [])

    print(f"\n📝 Building: {file_path}")

    # Check file cache
    dep_codes = [existing_files[d] for d in depends_on if d in existing_files]
    cache_key = _make_cache_key(file_description, dep_codes)
    cached = cache_get(cache_key)
    if cached:
        print(f"  ⚡ Cache hit for {file_path} — skipping LLM call")
        full_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(cached)
        return cached

    # Template injection — guaranteed-correct, never sent to LLM
    SKELETON_TEMPLATES = {
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
        cache_set(cache_key, code)
        print(f"  ✅ {file_path} written from template (guaranteed correct)")
        return code

    # Memory context injection
    memory_context = ""
    if MEMORY_ENABLED:
        file_ext = os.path.splitext(file_path)[1]
        for err_type in ["syntax_error", "truncated_component", "token_key_mismatch",
                         "plaintext_password", "raw_fetch_instead_of_api"]:
            fixes = recall_fixes(err_type, file_ext)
            if fixes:
                examples = "\n".join([
                    f"  [{err_type}] BROKEN: {f['broken'][:100]} → FIXED: {f['fixed'][:100]}"
                    for f in fixes[-2:]
                ])
                memory_context += f"\nKNOWN FIX PATTERN [{err_type}]:\n{examples}"
        with _session_lock:
            failed = list(_session_memory["failed_patterns"])
        if failed:
            memory_context += "\n\nAVOID THESE PATTERNS (caused failures this session):\n"
            for p in failed[-5:]:
                memory_context += f"  ❌ {p}\n"

    # File-aware context
    file_aware_context = ""
    if "frontend/src/api.js" in existing_files:
        api_exports = re.findall(
            r"^export\s+(?:const|async function|function)\s+(\w+)",
            existing_files["frontend/src/api.js"], re.MULTILINE
        )
        if api_exports:
            file_aware_context += f"\nAPI functions in '../api': {', '.join(api_exports)}"
            file_aware_context += "\nUse ONLY these exact names — do not invent new ones."

    built_components = [
        os.path.basename(p).replace(".js", "")
        for p in existing_files
        if p.startswith("frontend/src/components/") and p.endswith(".js")
    ]
    if built_components:
        file_aware_context += f"\nComponents already built: {', '.join(built_components)}"
        file_aware_context += "\nOnly import from this list."

    if file_path == "frontend/src/App.js":
        all_component_names = [
            os.path.basename(p).replace(".js", "")
            for p in existing_files
            if p.startswith("frontend/src/components/") and p.endswith(".js")
            and os.path.basename(p).replace(".js", "") not in ("PrivateRoute", "Navbar", "Footer", "Layout")
        ]
        blueprint_components = [
            os.path.basename(f["path"]).replace(".js", "")
            for f in blueprint.get("files", [])
            if "components/" in f.get("path", "") and f["path"].endswith(".js")
            and os.path.basename(f["path"]).replace(".js", "") not in ("PrivateRoute", "Navbar", "Footer", "Layout")
        ]
        all_page_components = list(dict.fromkeys(all_component_names + blueprint_components))
        if all_page_components:
            file_aware_context += f"\n\nPage components to route: {', '.join(all_page_components)}"
            file_aware_context += (
                "\nApp.js must route EVERY component above."
                "\nPURE ROUTER — no useEffect, no useState, no API calls."
                "\n<Navbar /> inside <BrowserRouter>."
                "\nPrivateRoute Outlet pattern only."
                "\nNEVER use 'import api from' — api.js has no default export."
            )

    dependency_context = ""
    for dep in depends_on:
        if dep in existing_files:
            dependency_context += f"\n\n--- {dep} ---\n{existing_files[dep]}"

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

    task_type = _get_task_type(file_path)
    system_prompt = BACKEND_PROMPT if task_type in ("backend", "fast") else FRONTEND_PROMPT
    max_tokens = get_optimal_tokens(file_path)

    print(f"  🎯 Task type: {task_type} | tokens: {max_tokens}")

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    last_error = ""
    final_code = ""
    failure_patterns = []

    for attempt in range(1, 4):
        if attempt > 1:
            print(f"  🔄 Retry attempt {attempt}...")
            if failure_patterns:
                hint = "\n\nPREVIOUS ATTEMPT FAILURES (do NOT repeat):\n" + "\n".join(f"- {p}" for p in failure_patterns)
                history[-1]["content"] = user_prompt + hint

        try:
            response = call_llm(history, max_tokens=max_tokens, task_type=task_type)
        except Exception as e:
            print(f"  ❌ All providers failed: {str(e)[:120]}")
            return final_code

        if not response or not response.choices:
            failure_patterns.append("empty response — output complete code")
            continue

        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            failure_patterns.append("blank content — output complete code")
            continue

        code = re.sub(r"^```[\w]*\n?", "", raw.strip())
        code = re.sub(r"\n?```$", "", code).strip()

        if len(code) < 50:
            failure_patterns.append(f"output too short ({len(code)} chars)")
            continue

        # Component truncation detection
        if file_path.endswith(".js") and "components/" in file_path:
            issues = []
            if "export default" not in code:
                issues.append("missing 'export default'")
            if "return (" not in code and "return(" not in code:
                issues.append("missing 'return ('")
            if issues:
                component_name = os.path.basename(file_path).replace(".js", "")
                fp = " | ".join(issues)
                failure_patterns.append(f"TRUNCATED — {fp}. Write all 8 steps through export default {component_name};")
                with _session_lock:
                    _session_memory["failed_patterns"].add(f"truncation in {component_name}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"TRUNCATED: {fp}. Rewrite completely."})
                continue

        # api.js validation
        if file_path == "frontend/src/api.js":
            issues = []
            if "interceptors" not in code:
                issues.append("missing interceptors")
            export_count = len(re.findall(r"^export\s+const\s+\w+", code, re.MULTILINE))
            endpoint_count = len(blueprint.get("api_endpoints", []))
            if export_count < max(2, endpoint_count - 2):
                issues.append(f"only {export_count}/{endpoint_count} exports")
            if "response.data.access_token" in code:
                issues.append("wrong token key (access_token → token)")
                with _session_lock:
                    _session_memory["failed_patterns"].add("access_token used instead of token")
            if issues:
                fp = " | ".join(issues)
                failure_patterns.append(f"api.js incomplete: {fp}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"api.js broken: {fp}. Rewrite fully. Token key = response.data.token."})
                continue

        # App.js structural validation
        if file_path == "frontend/src/App.js":
            issues = []
            if "from '../api'" in code or "from './api'" in code:
                issues.append("imports from api — App.js is router only")
            if "useEffect" in code:
                issues.append("has useEffect — remove all hooks")
            if "useState" in code:
                issues.append("has useState — remove all state")
            br_pos = code.find("<BrowserRouter")
            nav_pos = code.find("<Navbar")
            if nav_pos != -1 and br_pos != -1 and nav_pos < br_pos:
                issues.append("Navbar before BrowserRouter — move it inside")
            if re.search(r"<PrivateRoute\s*>\s*<\w", code):
                issues.append("old PrivateRoute children pattern — use Outlet")
            # Fix 2: 'import api from' is a default import — api.js has no default export
            if re.search(r"import\s+api\s+from", code):
                issues.append("'import api from' is a default import — api.js has no default export, use named imports: import { getProducts } from '../api'")
            if issues:
                fp = " | ".join(issues)
                failure_patterns.append(f"App.js structural: {fp}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"App.js wrong: {fp}. Pure router, no hooks, Navbar inside BrowserRouter, Outlet pattern, named imports only."})
                continue

        # Write file
        full_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)

        if file_path.endswith(".py"):
            result = execute_python_code(f"import ast\nast.parse(open('{full_path}').read())\nprint('syntax ok')")
            if "syntax ok" in result["stdout"]:
                print(f"  ✅ {file_path} built successfully")
                cache_set(cache_key, code)
                return code
            else:
                last_error = result["stderr"]
                print(f"  ❌ Syntax error: {last_error[:100]}")
                failure_patterns.append(f"syntax error: {last_error[:150]}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"Syntax error:\n{last_error}\n\nFix and output complete file."})
        else:
            print(f"  ✅ {file_path} built successfully")
            cache_set(cache_key, code)
            return code

    print(f"  ⚠️  Could not fix {file_path} after 3 attempts")
    return final_code


# ═══════════════════════════════════════════════════════════════
# PARALLEL FILE BUILDING
# ═══════════════════════════════════════════════════════════════

_existing_files_lock = threading.Lock()


def build_project(blueprint, output_dir="sandbox/projects", on_file_start=None, on_file_done=None):

    with _session_lock:
        _session_memory["provider_wins"].clear()
        _session_memory["provider_latency"].clear()
        _session_memory["fix_patterns"].clear()
        _session_memory["failed_patterns"].clear()

    mem = _load_long_term_memory()
    mem["build_count"] = mem.get("build_count", 0) + 1
    _save_long_term_memory(mem)

    project_name = blueprint["project_name"]
    project_path = os.path.join(output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"\n🚀 Building project: {project_name} (build #{mem['build_count']})")
    print(f"📁 Output: {project_path}")

    existing_files = {}
    failed_files = []
    files = blueprint["files"]

    def get_waves(files):
        completed, waves, remaining = set(), [], list(files)
        while remaining:
            wave, still_remaining = [], []
            for f in remaining:
                (wave if all(d in completed for d in f.get("depends_on", [])) else still_remaining).append(f)
            if not wave:
                wave, still_remaining = still_remaining, []
            for f in wave:
                completed.add(f["path"])
            waves.append(wave)
            remaining = still_remaining
        return waves

    waves = get_waves(files)
    print(f"⚡ Building in {len(waves)} wave(s) with parallel execution per wave")

    TEMPLATE_PATHS = {".env.example", "backend/__init__.py", "frontend/src/index.js", "frontend/src/components/PrivateRoute.js"}

    for wave_idx, wave in enumerate(waves):
        print(f"\n🌊 Wave {wave_idx + 1}/{len(waves)}: {len(wave)} file(s)")

        template_files = [f for f in wave if f["path"] in TEMPLATE_PATHS]
        llm_files = [f for f in wave if f not in template_files]

        for file_info in template_files:
            if on_file_start: on_file_start(file_info["path"])
            code = build_file(file_info, blueprint, project_path, dict(existing_files))
            if code:
                with _existing_files_lock:
                    existing_files[file_info["path"]] = code
                if on_file_done: on_file_done(file_info["path"], success=True)
            else:
                failed_files.append(file_info["path"])
                if on_file_done: on_file_done(file_info["path"], success=False)

        if llm_files:
            existing_snapshot = dict(existing_files)
            results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_path = {
                    executor.submit(build_file, fi, blueprint, project_path, existing_snapshot): fi["path"]
                    for fi in llm_files
                }
                if on_file_start:
                    for fi in llm_files: on_file_start(fi["path"])
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[path] = future.result()
                    except Exception as e:
                        print(f"  ❌ Thread error for {path}: {e}")
                        results[path] = None

            for path, code in results.items():
                if code:
                    with _existing_files_lock:
                        existing_files[path] = code
                    if on_file_done: on_file_done(path, success=True)
                else:
                    failed_files.append(path)
                    if on_file_done: on_file_done(path, success=False)

        if wave_idx < len(waves) - 1:
            time.sleep(2)

    print(f"\n{'='*50}")
    print("🧪 RUNNING TESTER + DEBUGGER...")
    print(f"{'='*50}")

    try:
        from agent.tester import run_tests, format_errors_for_log
        from agent.debugger import run_debug_loop
        existing_files, final_test_result, attempts = run_debug_loop(files=existing_files, tester_fn=run_tests, max_retries=3)
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

    readme = f"# {project_name.replace('_', ' ').title()}\n\n{blueprint['description']}\n\n## Setup\n\n### Backend\n```bash\npip install -r requirements.txt\ncp .env.example .env\nflask db init && flask db migrate && flask db upgrade\npython -m backend.app\n```\n\n### Frontend\n```bash\ncd frontend && npm install && npm start\n```\n"
    with open(os.path.join(project_path, "README.md"), "w") as f:
        f.write(readme)
    print("📄 README.md written")

    print(f"\n{'='*50}")
    print(f"✅ Project built: {len(existing_files)}/{len(files)} files")
    if failed_files:
        print(f"⚠️  Failed files: {failed_files}")

    with _session_lock:
        wins = dict(_session_memory["provider_wins"])
    if wins:
        top = sorted(wins.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"🏆 Top providers this build: {', '.join(f'{n}({c})' for n,c in top)}")

    print(f"📁 Location: {project_path}")
    return project_path, existing_files, failed_files


if __name__ == "__main__":
    from agent.planner import generate_blueprint
    blueprint = generate_blueprint("A simple e-commerce store where users can browse products, add to cart, and place orders")
    if blueprint:
        project_path, built, failed = build_project(blueprint)
        print(f"\nBuilt files: {list(built.keys())}")