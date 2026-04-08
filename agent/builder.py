import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import write_file, read_file, execute_python_code
try:
    from agent.memory import query_experience, add_experience
    MEMORY_ENABLED = True
except Exception:
    MEMORY_ENABLED = False
    def query_experience(desc): return ""
    def add_experience(desc, code, error=None): pass

load_dotenv()

# Pre-initialize all clients once at startup (faster than creating per call)
groq1   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2   = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3   = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter  = OpenAI(base_url="https://openrouter.ai/api/v1",      api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword  = OpenAI(base_url="https://api.doubleword.ai/v1",       api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

# 3 Groq + 3x2 Gemini + 3 OpenRouter + 2 Doubleword (paid fallback) = 14 providers
PROVIDERS = [
    # Groq — llama-3.3-70b (3 keys)
    {"name": "Groq-1 / llama-3.3-70b",       "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",       "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",       "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.15, max_tokens=mt)},
    # Groq — llama3-groq-70b (3 keys, separate rate limit pool)
    {"name": "Groq-1 / llama3-70b",          "call": lambda msgs, mt: groq1.chat.completions.create(model="llama3-groq-70b-8192-tool-use-preview", messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama3-70b",          "call": lambda msgs, mt: groq2.chat.completions.create(model="llama3-groq-70b-8192-tool-use-preview", messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama3-70b",          "call": lambda msgs, mt: groq3.chat.completions.create(model="llama3-groq-70b-8192-tool-use-preview", messages=msgs, temperature=0.15, max_tokens=mt)},
    # Groq — llama-3.1-8b (3 keys, separate rate limit pool)
    {"name": "Groq-1 / llama-3.1-8b",        "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.1-8b",        "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.1-8b",        "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.1-8b-instant",             messages=msgs, temperature=0.15, max_tokens=mt)},
    # Gemini 2.0 Flash (3 keys)
    {"name": "Gemini-1 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.0-flash",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.0-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    # Gemini 2.5 Flash (3 keys)
    {"name": "Gemini-1 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.15, max_tokens=mt)},
    # Gemini 2.5 Pro (3 keys — best free model for UI quality)
    {"name": "Gemini-1 / gemini-2.5-pro",    "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-pro-preview-05-06",       messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-pro",    "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-pro-preview-05-06",       messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-pro",    "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-pro-preview-05-06",       messages=msgs, temperature=0.15, max_tokens=mt)},
    # OpenRouter free models
    {"name": "OpenRouter / llama-3.3-70b",   "call": lambda msgs, mt: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "OpenRouter / gemma-3-27b",     "call": lambda msgs, mt: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free",  messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "OpenRouter / gemma-3-12b",     "call": lambda msgs, mt: openrouter.chat.completions.create(model="google/gemma-3-12b-it:free",  messages=msgs, temperature=0.15, max_tokens=mt)},
    # Paid fallback — only used when all free providers fail
    {"name": "Doubleword / Qwen3.5-35B",     "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",   messages=msgs, temperature=0.15, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",    "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.15, max_tokens=mt)},
]

# ─────────────────────────────────────────────
# UI-SPECIALIZED PROVIDERS
# Only best models for frontend — design needs taste
# ─────────────────────────────────────────────
UI_PROVIDERS = [
    {"name": "Gemini-1 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-pro-preview-05-06",       messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-pro-preview-05-06",       messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-pro",   "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-pro-preview-05-06",       messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash",               messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-1 / llama-3.3-70b",      "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",      "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",      "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile",          messages=msgs, temperature=0.2, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",   "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.2, max_tokens=mt)},
]

def call_llm(messages, max_tokens=4096, task_type="general"):
    """Try each provider in order. Uses UI_PROVIDERS for frontend files."""
    provider_list = UI_PROVIDERS if task_type == "ui" else PROVIDERS
    last_error = None
    for attempt, provider in enumerate(provider_list):
        try:
            print(f"  🤖 Using {provider['name']}...")
            return provider["call"](messages, max_tokens)
        except Exception as e:
            err = str(e).lower()
            # Always continue to next provider — never crash the build
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "quota", "503", "402", "temporarily", "overloaded", "upstream"]):
                wait = min(2 ** (attempt % 4), 16)
                print(f"  ⚠️  {provider['name']} rate limited, waiting {wait}s then trying next...")
                last_error = e
                time.sleep(wait)
                continue
            elif any(x in err for x in ["decommission", "deprecated", "no longer supported", "400", "404", "not found", "invalid model"]):
                print(f"  ⚠️  {provider['name']} model unavailable, trying next...")
                last_error = e
                continue
            else:
                print(f"  ⚠️  {provider['name']} error: {str(e)[:80]}, trying next...")
                last_error = e
                continue
    raise Exception(f"All providers failed. Last error: {last_error}")

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

For backend/config.py:
- Load all config from environment variables using os.environ.get()
- Include: SECRET_KEY, DATABASE_URL (fallback to SQLite), DEBUG, JWT_SECRET_KEY
- Set JWT_ACCESS_TOKEN_EXPIRES to timedelta(hours=24)
- Set SQLALCHEMY_TRACK_MODIFICATIONS = False

For backend/models.py:
- Use SQLAlchemy models with proper relationships using db.relationship() and backref
- ALWAYS add db.relationship() for every foreign key — never leave FK without a relationship
- Example: posts = db.relationship('Post', backref='author', lazy=True)
- Include password hashing via werkzeug: generate_password_hash, check_password_hash
- Include created_at = db.Column(db.DateTime, default=datetime.utcnow) on every model
- Include to_dict() method on every model returning all fields as JSON-serializable dict
- Place db.Index() OUTSIDE and AFTER class definitions, never inside
- Use Flask-SQLAlchemy with proper column types (String, Integer, Float, Boolean, DateTime, Text)
- Every model MUST have: id (primary key), created_at (DateTime, default=datetime.utcnow)
- Every string field MUST have a max length: String(100), String(255), etc.
- Add db.Index() for any foreign key column — MUST be placed OUTSIDE and AFTER the class definition, never inside it. Example: db.Index('ix_user_id', MyModel.user_id)
- Add __repr__ for every model
- to_dict() method must include ALL fields, converting datetime with .isoformat()
- Hash passwords using werkzeug.security.generate_password_hash — NEVER store plain text passwords
- Add a check_password(password) method to any User model

For backend/routes.py:
- Generate EVERY endpoint from the blueprint api_endpoints — never skip any
- Every POST/PUT endpoint MUST validate required fields and return 400 with clear error messages if missing
- Login endpoint MUST accept username field (not email) to match the Register form: data.get('username'), data.get('password')
- Register endpoint MUST accept: username, email, password
- Login success response MUST return a JWT token: {"token": create_access_token(identity=user.id), "user": user.to_dict()}
- Every GET list endpoint MUST support pagination: ?page=1&per_page=20 using .paginate()
- Return paginated responses as: {"items": [...], "total": n, "page": n, "pages": n}
- Every DELETE endpoint returns {"message": "Deleted successfully"}
- Use proper HTTP status codes: 200, 201, 400, 401, 403, 404, 409, 500
- Wrap all database writes in try/except, return 500 on db errors with db.session.rollback()
- Use @jwt_required() decorator to protect any route that modifies data
- Add CORS-safe responses using flask_cors

For backend/app.py:
- Create app factory pattern: def create_app()
- Initialize extensions: db, jwt, CORS, migrate
- MUST include Flask-Migrate: from flask_migrate import Migrate — then migrate = Migrate(app, db)
- Register blueprints
- Add a health check route: GET /health returns {"status": "ok"}
- ALWAYS import jsonify from flask: from flask import Flask, jsonify
- if __name__ == "__main__": app.run(debug=True, port=5000)
- With Flask-Migrate, users run: flask db init && flask db migrate && flask db upgrade

For requirements.txt:
- Include: flask, flask-cors, flask-sqlalchemy, flask-jwt-extended, flask-migrate, sqlalchemy, psycopg2-binary, python-dotenv, werkzeug

═══════════════════════════════════════════
FRONTEND RULES (React)
═══════════════════════════════════════════

For frontend/src/App.js:
- MUST contain real routing using React Router v6 (BrowserRouter, Routes, Route)
- Include routes for every major page inferred from the blueprint
- Include a Navbar component with navigation links
- Handle auth state: check localStorage for JWT token, show login/logout accordingly
- Every page component must be imported and rendered — no empty shells

For frontend/src/api.js:
- Use axios with baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000'
- Add axios request interceptor to inject Authorization: Bearer <token> from localStorage
- Add axios response interceptor: on 401, clear localStorage and redirect to /login
- Export individual async functions for EVERY API endpoint in the blueprint
- Each function uses try/catch and re-throws errors for the caller to handle

For frontend/public/index.html:
- Standard React HTML template with <div id="root"></div>
- Include proper meta charset, viewport tags
- Title should match the project name
- Include Tailwind CSS CDN: <script src="https://cdn.tailwindcss.com"></script>
- Include Google Fonts: Inter font family

For frontend/src/index.js:
- MUST use React 18 createRoot API: const root = ReactDOM.createRoot(document.getElementById('root')); root.render(<React.StrictMode><App /></React.StrictMode>)
- NEVER wrap App in BrowserRouter here — App.js already has BrowserRouter
- NEVER use ReactDOM.render() — it is deprecated in React 18
- NEVER add empty imports like import {} from 'react-router-dom' — only import what is actually used
- Only import: react, react-dom/client, ./index.css, ./App

For frontend/src/App.js:
- MUST contain real routing using React Router v6 (BrowserRouter, Routes, Route)
- Include routes for every major page inferred from the blueprint
- Include a Navbar component with navigation links
- NEVER add a second BrowserRouter — only one at the top level
- Handle auth with localStorage directly — NO onLogin props passed to children
- CRITICAL: ONLY import components that are explicitly listed in the blueprint files array
- NEVER invent new component names like CreatePostPage, EditPostPage, ProfilePage — use the exact filenames from the blueprint
- If a form is needed for creating/editing, use the existing [Resource]Form.js component with a route param
- Example: <Route path="/posts/new" element={<PostForm />} /> and <Route path="/posts/:id/edit" element={<PostForm />} />
- NEVER create inline placeholder components like const CreatePostPage = () => <div>...</div>
- Login/Register components handle their own redirect using useNavigate() after success

For frontend/src/api.js:
- Use axios with baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000'
- Add axios request interceptor to inject Authorization: Bearer <token> from localStorage
- Add axios response interceptor: on 401, clear localStorage and redirect to /login
- Export individual async functions for EVERY API endpoint in the blueprint
- Each function uses try/catch and re-throws errors for the caller to handle
- NEVER export the axios instance as default — only export named functions
- NEVER import the axios instance directly in components — always import named functions
  Correct: import { getProducts, login } from '../api'
  Wrong: import axios from '../api' or import api from '../api'

For frontend/src/components/PrivateRoute.js:
- A route guard component that checks localStorage for a JWT token
- If token exists: render the child component (use React Router v6 Outlet pattern)
- If no token: redirect to /login using Navigate from react-router-dom
- Exact implementation:
  import React from 'react';
  import { Navigate, Outlet } from 'react-router-dom';
  const PrivateRoute = () => {
    const token = localStorage.getItem('token');
    return token ? <Outlet /> : <Navigate to="/login" replace />;
  };
  export default PrivateRoute;
- App.js must wrap all protected routes with PrivateRoute:
  <Route element={<PrivateRoute />}>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/orders" element={<OrderList />} />
    <Route path="/cart" element={<Cart />} />
  </Route>
- Public routes (login, register, home) must NOT be inside PrivateRoute


- ALL styling MUST use Tailwind CSS utility classes. Never write inline styles or import CSS files.
- NEVER import individual CSS files per component (no './Home.css' etc.)
- NEVER link to internal API URLs in the UI (never show /api/posts as a link)

═══════════════════════════════════════════
UI DESIGN PATTERNS (copy these exactly)
═══════════════════════════════════════════

NAVBAR pattern — dark, sticky, professional:
  <nav className="bg-gray-900 sticky top-0 z-50 shadow-lg">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between h-16">
        <Link to="/" className="text-white font-bold text-xl tracking-tight">AppName</Link>
        <div className="flex items-center gap-6">
          <Link to="/posts" className="text-gray-300 hover:text-white transition text-sm font-medium">Posts</Link>
          {user ? (
            <div className="flex items-center gap-4">
              <span className="text-gray-300 text-sm">Hi, {user.username}</span>
              <button onClick={handleLogout} className="bg-red-600 hover:bg-red-700 text-white text-sm font-medium px-4 py-2 rounded-lg transition">Logout</button>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <Link to="/login" className="text-gray-300 hover:text-white text-sm font-medium transition">Login</Link>
              <Link to="/register" className="bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium px-4 py-2 rounded-lg transition">Sign up</Link>
            </div>
          )}
        </div>
      </div>
    </div>
  </nav>

HOME page pattern — hero section with gradient, feature cards:
  <div className="min-h-screen bg-gray-50">
    {/* Hero */}
    <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 text-white py-24 px-4">
      <div className="max-w-4xl mx-auto text-center">
        <h1 className="text-5xl font-bold mb-6 leading-tight">App Title Here</h1>
        <p className="text-xl text-indigo-100 mb-10 max-w-2xl mx-auto">One line description of what this app does.</p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link to="/register" className="bg-white text-indigo-600 font-bold px-8 py-3 rounded-xl hover:bg-indigo-50 transition shadow-lg">Get Started Free</Link>
          <Link to="/login" className="border-2 border-white text-white font-bold px-8 py-3 rounded-xl hover:bg-white hover:text-indigo-600 transition">Sign In</Link>
        </div>
      </div>
    </div>
    {/* Feature cards */}
    <div className="max-w-6xl mx-auto px-4 py-16">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="bg-white rounded-2xl shadow-md p-8 hover:shadow-xl transition">
          <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          </div>
          <h3 className="text-lg font-bold text-gray-900 mb-2">Feature Title</h3>
          <p className="text-gray-500 text-sm leading-relaxed">Feature description here.</p>
        </div>
      </div>
    </div>
  </div>

LOGIN/REGISTER pattern — centered card with branding:
  <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50 flex items-center justify-center px-4">
    <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-8">
      <div className="text-center mb-8">
        <div className="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900">Welcome back</h2>
        <p className="text-gray-500 text-sm mt-1">Sign in to your account</p>
      </div>
      {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-6 text-sm">{error}</div>}
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Username</label>
          <input className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition" type="text" value={username} onChange={e => setUsername(e.target.value)} placeholder="Enter your username" required />
        </div>
        <button type="submit" className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-xl transition shadow-md">Sign In</button>
      </form>
      <p className="text-center text-sm text-gray-500 mt-6">Don't have an account? <Link to="/register" className="text-indigo-600 font-medium hover:underline">Sign up</Link></p>
    </div>
  </div>

LIST page pattern — page header + grid of cards:
  <div className="min-h-screen bg-gray-50 py-8 px-4">
    <div className="max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Page Title</h1>
          <p className="text-gray-500 mt-1">Subtitle description</p>
        </div>
        <Link to="/new" className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-6 py-3 rounded-xl transition shadow-md flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4"/></svg>
          New Item
        </Link>
      </div>
      {loading && <div className="flex justify-center py-16"><div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-600 border-t-transparent"></div></div>}
      {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm">{error}</div>}
      {!loading && items.length === 0 && (
        <div className="bg-white rounded-2xl shadow p-16 text-center">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"/></svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-1">No items yet</h3>
          <p className="text-gray-500 text-sm">Get started by creating your first one.</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {items.map(item => (
          <div key={item.id} className="bg-white rounded-2xl shadow-md hover:shadow-xl transition p-6 cursor-pointer group">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-bold text-sm">{item.username?.[0]?.toUpperCase() || 'U'}</div>
              <div>
                <p className="font-semibold text-gray-900 text-sm">{item.username}</p>
                <p className="text-gray-400 text-xs">{new Date(item.created_at).toLocaleDateString()}</p>
              </div>
            </div>
            <p className="text-gray-700 text-sm leading-relaxed line-clamp-3">{item.content || item.title}</p>
          </div>
        ))}
      </div>
    </div>
  </div>

FORM page pattern — clean centered form:
  <div className="min-h-screen bg-gray-50 py-8 px-4">
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Create New Item</h1>
        <p className="text-gray-500 mt-1">Fill in the details below</p>
      </div>
      <div className="bg-white rounded-2xl shadow-md p-8">
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-6 text-sm">{error}</div>}
        {success && <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-xl mb-6 text-sm">{success}</div>}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Field Label</label>
            <input className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition" type="text" placeholder="Enter value..." required />
          </div>
          <div className="flex gap-4 pt-2">
            <button type="submit" className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-8 py-3 rounded-xl transition shadow-md">Save</button>
            <button type="button" onClick={() => navigate(-1)} className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold px-8 py-3 rounded-xl transition">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  </div>

GENERAL UI RULES:
- Use rounded-2xl for cards and modals (not rounded-xl)
- Use shadow-md normally, shadow-xl on hover
- Avatar initials: first letter of username in a colored circle
- Dates: always use toLocaleDateString() not raw ISO string
- Page wrapper: min-h-screen bg-gray-50 py-8 px-4
- Content width: max-w-7xl mx-auto for lists, max-w-2xl for forms, max-w-md for auth
- Transitions: always add "transition" class to interactive elements
- NEVER use bullet point lists as page features — use feature cards with icons instead

For frontend/src/components/ files — HOOKS & BUG-FREE RULES:
- Every component must use React hooks: useState for local state, useEffect for data fetching
- CRITICAL: Every useEffect MUST have a dependency array []. NEVER write useEffect(() => {}) without []
- Correct: useEffect(() => { fetchData() }, []) — runs once on mount only
- CRITICAL: Navbar MUST check localStorage.getItem('token') before calling getUser(). If no token, skip API call entirely.
- CRITICAL: All error handling must check if error.response exists before accessing error.response.status
  Safe pattern: const msg = error.response?.data?.message || error.message || 'Something went wrong'
- Forms must have controlled inputs with onChange handlers and onSubmit with preventDefault()
- After successful POST/PUT/DELETE, refresh the data list automatically
- NEVER import LoadingSpinner, Pagination, ErrorAlert or any helper component not in the blueprint file list
- Write loading/error/empty/pagination logic INLINE
- Loading inline: {loading && <div className="flex justify-center py-12"><div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent"></div></div>}
- ONLY import from: react, react-router-dom, ../api

For frontend/package.json:
- Include: react, react-dom, react-scripts, axios, react-router-dom as dependencies
- Include start, build, test scripts
- Set proxy: "http://localhost:5000" for development

For frontend/src/index.css:
- Minimal CSS — just body font-family: 'Inter', sans-serif and box-sizing: border-box
- All real styling is done via Tailwind classes in components

═══════════════════════════════════════════
GENERAL FILES
═══════════════════════════════════════════

For .env.example:
- Include: DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY, DEBUG, FLASK_ENV, REACT_APP_API_URL

For README.md:
- Include: project description, tech stack, prerequisites, setup steps (backend + frontend), environment variables table, API endpoints table with method/path/description/auth columns

QUALITY BAR: The code you write must be indistinguishable from code written by a senior engineer at a real software company. It must be immediately runnable with no modifications needed beyond filling in .env values.

FRONTEND UI QUALITY BAR — STRICTLY ENFORCED:
UI must feel like a premium SaaS product (Stripe, Linear, Vercel level). If you generate boring UI, you have failed.

✅ REQUIRED in every frontend component:
- Generous spacing: padding-8, gap-6, py-16 for sections
- Large headings: text-3xl to text-5xl for page titles
- Hover effects on EVERY interactive element
- Smooth transitions: className="... transition duration-200"
- Card-based layouts with rounded-2xl and shadow-md
- Avatar initials for user content
- Proper empty states with SVG icons
- Gradient hero sections on Home page
- Color-coded status badges

❌ STRICTLY FORBIDDEN — never generate these:
- Plain vertical stack of inputs with no spacing
- Small text (text-sm) for main content
- No hover states on buttons or cards
- Flat gray divs with no visual hierarchy
- Bullet point lists as page features
- Linking to /api/... URLs in the UI
- No loading states or empty states
- Plain white pages with no background color
"""

def build_file(file_info, blueprint, project_path, existing_files={}):
    """Builds a single file based on blueprint context."""
    
    file_path = file_info["path"]
    file_description = file_info["description"]
    depends_on = file_info.get("depends_on", [])

    print(f"\n📝 Building: {file_path}")

    # Build context from dependencies
    dependency_context = ""
    for dep in depends_on:
        if dep in existing_files:
            dependency_context += f"\n\n--- {dep} ---\n{existing_files[dep]}"

    # Query memory for similar past solutions
    past_experience = query_experience(file_description)
    memory_context = ""
    if past_experience and past_experience[0]:
        memory_context = "\nPAST SIMILAR CODE (use as reference):\n" + "\n".join(past_experience[0])

    # Build the prompt
    user_prompt = f"""
Project: {blueprint['description']}
Stack: {blueprint['stack']}
Database tables: {json.dumps(blueprint['database_schema']['tables'], indent=2)}
API endpoints: {json.dumps(blueprint['api_endpoints'], indent=2)}

File to write: {file_path}
Purpose: {file_description}

Dependencies already written:
{dependency_context if dependency_context else "None"}

{memory_context}

Write the COMPLETE, PRODUCTION-READY code for {file_path} now.
Remember: No placeholders, no TODOs, no stubs. Real working code only.
"""

    history = [
        {"role": "system", "content": BUILDER_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    last_error = ""
    final_code = ""

    # Use UI-specialized pipeline + more tokens for frontend files
    is_frontend = file_path.startswith("frontend/") and file_path.endswith((".js", ".jsx", ".css", ".html"))
    task_type = "ui" if is_frontend else "general"
    max_tokens = 7000 if is_frontend else 4096

    if is_frontend:
        print(f"  🎨 Using UI pipeline (Gemini 2.5 Pro) with {max_tokens} tokens")

    for attempt in range(1, 4):
        if attempt > 1:
            print(f"  🔄 Retry attempt {attempt}...")

        try:
            response = call_llm(history, max_tokens=max_tokens, task_type=task_type)
        except Exception as e:
            print(f"  ❌ All providers failed: {str(e)[:120]}")
            return final_code

        code = response.choices[0].message.content.strip()

        # Clean up backticks safely - replace known patterns directly
        code = code.replace("```python", "").replace("```javascript", "").replace("```jsx", "").replace("```css", "").replace("```json", "").replace("```html", "").replace("```", "").strip()

        # Write file to project
        full_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)

        # Only run syntax check for Python files
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
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"Syntax error found:\n{last_error}\n\nFix the syntax error and output the complete corrected file."})
        else:
            print(f"  ✅ {file_path} built successfully")
            final_code = code
            return code

    print(f"  ⚠️ Could not fix {file_path} after 3 attempts")
    return final_code


def build_project(blueprint, output_dir="sandbox/projects", on_file_start=None, on_file_done=None):
    """Builds an entire project from a blueprint — parallel where possible."""

    project_name = blueprint["project_name"]
    project_path = os.path.join(output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"\n🚀 Building project: {project_name}")
    print(f"📁 Output: {project_path}")

    existing_files = {}
    failed_files = []
    files = blueprint["files"]

    # ─────────────────────────────────────────────
    # Split files into dependency waves:
    # Wave 0: no dependencies (build in parallel)
    # Wave 1: depends on wave 0 (build in parallel after wave 0)
    # Wave 2: depends on wave 1, etc.
    # ─────────────────────────────────────────────
    def get_waves(files):
        """Group files into waves based on dependencies."""
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
                # Circular deps or unresolvable — just add all remaining
                wave = still_remaining
                still_remaining = []
            for f in wave:
                completed.add(f["path"])
            waves.append(wave)
            remaining = still_remaining
        return waves

    waves = get_waves(files)
    print(f"⚡ Building in {len(waves)} wave(s) — parallel within each wave")

    for wave_idx, wave in enumerate(waves):
        print(f"\n🌊 Wave {wave_idx + 1}/{len(waves)}: {len(wave)} file(s)")

        if len(wave) == 1:
            # Single file — build directly, no threading overhead
            file_info = wave[0]
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
        else:
            # Multiple files — build in parallel with ThreadPoolExecutor
            # Use max 4 workers to avoid hammering the LLM API
            # Max 2 workers on free tier (512MB RAM limit)
            max_workers = min(2, len(wave))
            wave_results = {}

            # Notify all files as "building" before starting threads
            for file_info in wave:
                if on_file_start:
                    on_file_start(file_info["path"])

            def build_one(file_info):
                """Build a single file — runs in thread."""
                code = build_file(
                    file_info=file_info,
                    blueprint=blueprint,
                    project_path=project_path,
                    existing_files=dict(existing_files)  # snapshot for thread safety
                )
                return file_info["path"], code

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(build_one, f): f for f in wave}
                for future in as_completed(futures):
                    try:
                        file_path, code = future.result()
                        if code:
                            wave_results[file_path] = code
                            if on_file_done:
                                on_file_done(file_path, success=True)
                        else:
                            failed_files.append(file_path)
                            if on_file_done:
                                on_file_done(file_path, success=False)
                    except Exception as e:
                        file_path = futures[future]["path"]
                        print(f"  ❌ Thread error for {file_path}: {e}")
                        failed_files.append(file_path)
                        if on_file_done:
                            on_file_done(file_path, success=False)

            # Merge wave results into existing_files
            existing_files.update(wave_results)

        # Small pause between waves (not between files) to be nice to APIs
        if wave_idx < len(waves) - 1:
            time.sleep(2)

    # ─────────────────────────────────────────────
    # PHASE 2: TEST + DEBUG LOOP
    # ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("🧪 RUNNING TESTER + DEBUGGER...")
    print(f"{'='*50}")

    try:
        from agent.tester import run_tests, format_errors_for_log
        from agent.debugger import run_debug_loop

        # Run full test → debug → retest loop (max 3 attempts)
        existing_files, final_test_result, attempts = run_debug_loop(
            files=existing_files,
            tester_fn=run_tests,
            max_retries=3
        )

        # Write back any fixed files to disk
        for file_path, code in existing_files.items():
            full_path = os.path.join(project_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(code)

        print(f"\n🔬 Test result after {attempts} debug attempt(s): {final_test_result['summary']}")

    except Exception as e:
        print(f"\n⚠️  Tester/Debugger failed: {e} — continuing with unvalidated build")

    # Write requirements.txt
    requirements = """flask
flask-cors
flask-sqlalchemy
flask-jwt-extended
flask-migrate
sqlalchemy
psycopg2-binary
python-dotenv
werkzeug
"""
    with open(os.path.join(project_path, "requirements.txt"), "w") as f:
        f.write(requirements)
    print("\n📄 requirements.txt written")

    # Write README
    readme = f"""# {project_name.replace('_', ' ').title()}

{blueprint['description']}

## Stack
- Frontend: {blueprint['stack']['frontend']}
- Backend: {blueprint['stack']['backend']}
- Database: {blueprint['stack']['database']}

## Prerequisites
- Python 3.9+
- Node.js 16+
- PostgreSQL

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env      # Fill in your values
flask db init && flask db migrate && flask db upgrade
python app.py
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env      # Set REACT_APP_API_URL
npm start
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

    # Summary
    print(f"\n{'='*50}")
    print(f"✅ Project built: {len(existing_files)}/{len(files)} files")
    if failed_files:
        print(f"⚠️  Failed files: {failed_files}")
    print(f"📁 Location: {project_path}")

    return project_path, existing_files, failed_files


# Test it
if __name__ == "__main__":
    from agent.planner import generate_blueprint

    blueprint = generate_blueprint(
        "A simple e-commerce store where users can browse products, add to cart, and place orders"
    )

    if blueprint:
        project_path, built, failed = build_project(blueprint)
        print(f"\nBuilt files: {list(built.keys())}")