import os
import json
from groq import Groq
from dotenv import load_dotenv
from agent.tools import write_file, read_file, execute_python_code
from agent.memory import query_experience, add_experience

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
- Use Flask-SQLAlchemy with proper column types (String, Integer, Float, Boolean, DateTime, Text)
- Every model MUST have: id (primary key), created_at (DateTime, default=datetime.utcnow)
- Every string field MUST have a max length: String(100), String(255), etc.
- Add db.Index() for any foreign key column for query performance
- Add __repr__ for every model
- to_dict() method must include ALL fields, converting datetime with .isoformat()
- Hash passwords using werkzeug.security.generate_password_hash — NEVER store plain text passwords
- Add a check_password(password) method to any User model

For backend/routes.py:
- Generate EVERY endpoint from the blueprint api_endpoints — never skip any
- Every POST/PUT endpoint MUST validate required fields and return 400 with clear error messages if missing
- Every GET list endpoint MUST support pagination: ?page=1&per_page=20 using .paginate()
- Return paginated responses as: {"items": [...], "total": n, "page": n, "pages": n}
- Every DELETE endpoint returns {"message": "Deleted successfully"}
- Use proper HTTP status codes: 200, 201, 400, 401, 403, 404, 409, 500
- Wrap all database writes in try/except, return 500 on db errors with db.session.rollback()
- Use @jwt_required() decorator to protect any route that modifies data
- Add CORS-safe responses using flask_cors

For backend/app.py:
- Create app factory pattern: def create_app()
- Initialize extensions: db, jwt, CORS
- Register blueprints
- Add a health check route: GET /health returns {"status": "ok"}
- if __name__ == "__main__": app.run(debug=True, port=5000)

For requirements.txt:
- Include: flask, flask-cors, flask-sqlalchemy, flask-jwt-extended, sqlalchemy, psycopg2-binary, python-dotenv, werkzeug

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

For frontend/src/components/ files:
- Every component must use React hooks: useState for local state, useEffect for data fetching
- Every component that fetches data must show: loading spinner while loading, error message on failure, empty state when no data
- Forms must have controlled inputs with onChange handlers and onSubmit with preventDefault()
- Forms must show validation errors inline before submitting
- After successful POST/PUT/DELETE, refresh the data list automatically
- Use async/await for all API calls, wrapped in try/catch with proper error state

For frontend/package.json:
- Include: react, react-dom, react-scripts, axios, react-router-dom as dependencies
- Include start, build, test scripts
- Set proxy: "http://localhost:5000" for development

For frontend/src/index.css:
- Write a complete, modern CSS stylesheet
- Use CSS variables for colors, font sizes, spacing
- Style: body, buttons (primary/secondary/danger), forms, inputs, tables, cards, navbar, loading spinner, error messages
- Make it responsive with at least one @media (max-width: 768px) breakpoint
- Use a clean color scheme — not just black and white

═══════════════════════════════════════════
GENERAL FILES
═══════════════════════════════════════════

For .env.example:
- Include: DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY, DEBUG, FLASK_ENV, REACT_APP_API_URL

For README.md:
- Include: project description, tech stack, prerequisites, setup steps (backend + frontend), environment variables table, API endpoints table with method/path/description/auth columns

QUALITY BAR: The code you write must be indistinguishable from code written by a senior engineer at a real software company. It must be immediately runnable with no modifications needed beyond filling in .env values.
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

    for attempt in range(1, 4):
        if attempt > 1:
            print(f"  🔄 Retry attempt {attempt}...")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            temperature=0.15,
            max_tokens=4096
        )

        code = response.choices[0].message.content.strip()

        # Clean up backticks if model adds them
        if "```" in code:
            lines = code.split('\n')
            clean_lines = []
            inside_block = False
            for line in lines:
                if line.startswith("```"):
                    inside_block = not inside_block
                    continue
                if inside_block or not line.startswith("```"):
                    clean_lines.append(line)
            code = '\n'.join(clean_lines).strip()

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


def build_project(blueprint, output_dir="sandbox/projects"):
    """Builds an entire project from a blueprint."""

    project_name = blueprint["project_name"]
    project_path = os.path.join(output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"\n🚀 Building project: {project_name}")
    print(f"📁 Output: {project_path}")

    existing_files = {}
    failed_files = []

    # Sort files by dependency order
    files = blueprint["files"]
    ordered_files = sorted(files, key=lambda f: len(f.get("depends_on", [])))

    for file_info in ordered_files:
        code = build_file(
            file_info=file_info,
            blueprint=blueprint,
            project_path=project_path,
            existing_files=existing_files
        )
        if code:
            existing_files[file_info["path"]] = code
        else:
            failed_files.append(file_info["path"])

    # Write requirements.txt
    requirements = """flask
flask-cors
flask-sqlalchemy
flask-jwt-extended
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