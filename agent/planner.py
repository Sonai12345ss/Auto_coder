import os
import json
import time
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Pre-initialize all clients once at startup
groq1   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2   = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3   = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter  = OpenAI(base_url="https://openrouter.ai/api/v1",  api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword  = OpenAI(base_url="https://api.doubleword.ai/v1",  api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

def call_llm(messages):
    """Try each provider in order with exponential backoff on rate limits."""
    providers = [
        ("Groq-1 / llama-3.3-70b",      lambda: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-2 / llama-3.3-70b",      lambda: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Groq-3 / llama-3.3-70b",      lambda: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-1 / gemini-2.0-flash",  lambda: gemini1.chat.completions.create(model="gemini-2.0-flash", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-2 / gemini-2.0-flash",  lambda: gemini2.chat.completions.create(model="gemini-2.0-flash", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-3 / gemini-2.0-flash",  lambda: gemini3.chat.completions.create(model="gemini-2.0-flash", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-1 / gemini-2.5-flash",  lambda: gemini1.chat.completions.create(model="gemini-2.5-flash", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-2 / gemini-2.5-flash",  lambda: gemini2.chat.completions.create(model="gemini-2.5-flash", messages=messages, temperature=0.2, max_tokens=4096)),
        ("Gemini-3 / gemini-2.5-flash",  lambda: gemini3.chat.completions.create(model="gemini-2.5-flash", messages=messages, temperature=0.2, max_tokens=4096)),
        ("OpenRouter / llama-3.3-70b",   lambda: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=messages, temperature=0.2, max_tokens=4096)),
        ("OpenRouter / gemma-3-27b",     lambda: openrouter.chat.completions.create(model="google/gemma-3-27b-it:free", messages=messages, temperature=0.2, max_tokens=4096)),
        ("OpenRouter / gemma-3-12b",     lambda: openrouter.chat.completions.create(model="google/gemma-3-12b-it:free",   messages=messages, temperature=0.2, max_tokens=4096)),
        # Paid fallback — only used when all free providers fail
        ("Doubleword / Qwen3.5-35B",     lambda: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",    messages=messages, temperature=0.2, max_tokens=4096)),
        ("Doubleword / Qwen3.5-397B",    lambda: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8",  messages=messages, temperature=0.2, max_tokens=4096)),
    ]
    last_error = None
    for attempt, (name, fn) in enumerate(providers):
        try:
            print(f"  🤖 Planner using {name}...")
            return fn()
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "404", "402", "503", "quota", "temporarily", "overloaded", "upstream"]):
                wait = min(2 ** (attempt % 4), 16)
                print(f"  ⚠️  {name} rate limited, waiting {wait}s then trying next...")
                last_error = e
                time.sleep(wait)
                continue
            last_error = e
            print(f"  ⚠️  {name} failed with: {str(e)[:80]}, trying next...")
            continue
    raise Exception(f"All planner providers failed. Last error: {last_error}")

PLANNER_PROMPT = """
You are a senior software architect. Your job is to take a project description and create a detailed project blueprint in JSON format.

RULES:
1. Output ONLY valid JSON, nothing else. No explanation, no markdown, no backticks.
2. Every file that is IMPORTED by another file MUST be listed in the files array.
3. App.js imports components — every component it imports MUST have its own file entry.
4. NEVER import a component in App.js that is not listed as a file to be generated.

BACKEND FILES (always include all of these):
- backend/config.py — database URL, SECRET_KEY, JWT_SECRET_KEY, debug settings
- backend/models.py — SQLAlchemy models with password hashing, to_dict(), created_at
- backend/routes.py — ALL API endpoints including /api/register, /api/login, /api/user
- backend/app.py — Flask app factory, register blueprints, init db and JWT

FRONTEND FILES (always include all of these):
- frontend/package.json — react, react-dom, react-scripts, axios, react-router-dom
- frontend/src/index.js — ReactDOM.render entry point
- frontend/src/index.css — complete stylesheet with variables, buttons, forms, navbar, cards
- frontend/src/App.js — routing with BrowserRouter, Routes, Route for every page
- frontend/src/api.js — axios instance with JWT interceptor, one function per endpoint

FRONTEND COMPONENT FILES (generate based on the project — include ALL of these):
- frontend/src/components/Navbar.js — navigation bar with auth state (login/logout links)
- frontend/src/components/Login.js — login form with controlled inputs, error handling
- frontend/src/components/Register.js — register form with validation
- frontend/src/components/Home.js — landing/home page component
- For each main resource in the project, generate MAXIMUM 2 components:
  - frontend/src/components/[Resource]List.js — list view WITH inline create form (combines List + Form in one file)
  - frontend/src/components/[Resource]Detail.js — detail/single view (only if truly needed)
- STRICT LIMIT: Maximum 12 total files in the files array. If you need more, combine related components.
- NEVER generate separate [Resource]Form.js — put the form inside [Resource]List.js instead
- For a blog: PostList.js handles list + create post. CommentList.js handles comments inline inside PostDetail.js

CRITICAL RULES FOR FILES:
- App.js must ONLY import components that exist in the files list
- Every component listed must have a complete, working implementation
- routes.py MUST include /api/register (POST), /api/login (POST), /api/user (GET)
- All list endpoints must support ?page=1&per_page=20 pagination
- All write endpoints must use @jwt_required() and validate input fields
- NEVER generate a file called Routing.js — it does not exist, routing is handled inside App.js
- NEVER generate Router.js, Routes.js, or any wrapper routing component
- ALWAYS include frontend/src/components/PrivateRoute.js — a route guard that redirects to /login if no JWT token in localStorage
- ALWAYS include frontend/public/index.html — the React HTML template with Tailwind CDN
- TOTAL FILE COUNT must be 12 or fewer — combine components to stay within this limit

OUTPUT FORMAT — output exactly this JSON structure:
{
  "project_name": "snake_case_name",
  "description": "one line description",
  "stack": {
    "frontend": "React",
    "backend": "Flask",
    "database": "PostgreSQL"
  },
  "files": [
    {
      "path": "backend/config.py",
      "description": "Flask config with DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY from env",
      "depends_on": []
    },
    {
      "path": "backend/models.py",
      "description": "SQLAlchemy models with password hashing via werkzeug, to_dict(), created_at on every model",
      "depends_on": ["backend/config.py"]
    },
    {
      "path": "backend/routes.py",
      "description": "All API endpoints including /api/register, /api/login, /api/user plus all resource endpoints with pagination and JWT protection",
      "depends_on": ["backend/models.py"]
    },
    {
      "path": "backend/app.py",
      "description": "Flask app factory pattern, init db/jwt/cors, register blueprint, health check route",
      "depends_on": ["backend/config.py", "backend/models.py", "backend/routes.py"]
    },
    {
      "path": "frontend/package.json",
      "description": "React dependencies: react, react-dom, react-scripts, axios, react-router-dom",
      "depends_on": []
    },
    {
      "path": "frontend/src/index.css",
      "description": "Complete stylesheet with CSS variables, body, buttons, forms, inputs, navbar, cards, loading spinner, error states, responsive breakpoints",
      "depends_on": []
    },
    {
      "path": "frontend/src/api.js",
      "description": "Axios instance with baseURL, JWT request interceptor, 401 response interceptor, one async function per API endpoint",
      "depends_on": []
    },
    {
      "path": "frontend/src/components/Navbar.js",
      "description": "Navbar component with navigation links, shows login/register when logged out, shows username and logout button when logged in",
      "depends_on": ["frontend/src/api.js"]
    },
    {
      "path": "frontend/src/components/Home.js",
      "description": "Home page component with welcome message and links to main features",
      "depends_on": []
    },
    {
      "path": "frontend/src/components/Login.js",
      "description": "Login form with email and password controlled inputs, onSubmit calls api login, shows inline error, calls onLogin prop with token on success",
      "depends_on": ["frontend/src/api.js"]
    },
    {
      "path": "frontend/src/components/Register.js",
      "description": "Register form with username, email, password fields, validation, calls api register, redirects to login on success",
      "depends_on": ["frontend/src/api.js"]
    }
  ],
  "database_schema": {
    "tables": [
      {
        "name": "users",
        "columns": [
          {"name": "id", "type": "Integer PK"},
          {"name": "username", "type": "String(100)"},
          {"name": "email", "type": "String(255)"},
          {"name": "password", "type": "String(255)"},
          {"name": "created_at", "type": "DateTime"}
        ]
      }
    ]
  },
  "api_endpoints": [
    {"method": "POST", "path": "/api/register", "description": "Register new user", "auth_required": false},
    {"method": "POST", "path": "/api/login", "description": "Login and get JWT token", "auth_required": false},
    {"method": "GET", "path": "/api/user", "description": "Get current user info", "auth_required": true}
  ],
  "setup_instructions": [
    "Copy .env.example to .env and fill in your values",
    "pip install -r requirements.txt",
    "python backend/app.py",
    "cd frontend && npm install && npm start"
  ]
}

IMPORTANT: The files array in your output must include ALL component files that App.js will import.
Add resource-specific components based on the project (e.g. for a todo app add TodoList.js, TodoForm.js).
"""

REQUIRED_FILES = [
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
    ".env.example"
]


def generate_blueprint(project_description):
    """Takes a project description and returns a structured JSON blueprint."""
    print("\n🧠 Planner Agent thinking...")

    response = call_llm([
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"Create a complete blueprint for: {project_description}\n\nRemember: every component that App.js imports MUST be in the files list."}
    ])

    raw = response.choices[0].message.content.strip()

    # Clean up in case model adds backticks anyway
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        blueprint = json.loads(raw)

        # Enforce required files are always present
        existing_paths = [f["path"] for f in blueprint["files"]]
        for required in REQUIRED_FILES:
            if required not in existing_paths:
                print(f"⚠️  Adding missing required file: {required}")
                blueprint["files"].append({
                    "path": required,
                    "description": f"Required file: {required}",
                    "depends_on": []
                })

        # KEY FIX: Parse App.js description to find any extra components mentioned
        # and ensure they are in the files list
        app_js_entry = next((f for f in blueprint["files"] if f["path"] == "frontend/src/App.js"), None)
        if app_js_entry:
            existing_paths = [f["path"] for f in blueprint["files"]]
            desc = app_js_entry.get("description", "")
            import re
            component_names = re.findall(r'\b([A-Z][a-zA-Z]+)\b', desc)
            EXCLUDED = {"React", "Route", "Routes", "BrowserRouter", "Navigate", "Routing",
                        "Link", "NavLink", "App", "Switch", "Router", "Component", "Fragment",
                        "Provider", "Context", "Suspense", "Redirect", "RouterProvider",
                        "PrivateRoute"}
            for name in component_names:
                component_path = f"frontend/src/components/{name}.js"
                if component_path not in existing_paths and name not in EXCLUDED:
                    print(f"⚠️  Auto-adding component from App.js description: {component_path}")
                    blueprint["files"].append({
                        "path": component_path,
                        "description": f"{name} component used in App.js routing",
                        "depends_on": ["frontend/src/api.js"]
                    })

        # Hard cap: never more than 16 files total (keeps builds under 8 mins)
        if len(blueprint["files"]) > 16:
            print(f"⚠️  Blueprint has {len(blueprint['files'])} files — capping at 16")
            # Keep all required files + first N component files
            required = [f for f in blueprint["files"] if f["path"] in REQUIRED_FILES or
                       not f["path"].startswith("frontend/src/components/")]
            components = [f for f in blueprint["files"] if f["path"].startswith("frontend/src/components/")
                         and f["path"] not in REQUIRED_FILES]
            blueprint["files"] = required + components[:16 - len(required)]

        # Remove ghost routing files — React Router concepts, not real components
        GHOST_FILES = {
            "frontend/src/components/Routing.js",
            "frontend/src/components/Router.js",
            "frontend/src/components/Routes.js",
            "frontend/src/Routing.js",
            "frontend/src/Router.js",
        }
        blueprint["files"] = [f for f in blueprint["files"] if f["path"] not in GHOST_FILES]

        # Sort files by dependency order
        blueprint["files"] = sorted(
            blueprint["files"],
            key=lambda f: len(f.get("depends_on", []))
        )

        print(f"✅ Blueprint generated: {blueprint['project_name']}")
        print(f"📁 Files to create: {len(blueprint['files'])}")
        print(f"🗄️  Database tables: {[t['name'] if isinstance(t, dict) else t for t in blueprint['database_schema']['tables']]}")
        print(f"🔗 API endpoints: {len(blueprint['api_endpoints'])}")
        print(f"📦 Components: {[f['path'].split('/')[-1] for f in blueprint['files'] if 'components/' in f['path']]}")
        return blueprint

    except json.JSONDecodeError as e:
        print(f"❌ Planner failed to generate valid JSON: {e}")
        print(f"Raw output was:\n{raw}")
        return None


# Test it
if __name__ == "__main__":
    blueprint = generate_blueprint(
        "A real-time chat app where users can create rooms and send messages"
    )
    if blueprint:
        import os
        os.makedirs("sandbox", exist_ok=True)
        with open("sandbox/blueprint.json", "w") as f:
            json.dump(blueprint, f, indent=2)
        print("\n📄 Blueprint saved to sandbox/blueprint.json")