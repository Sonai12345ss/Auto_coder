import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PLANNER_PROMPT = """
You are a senior software architect. Your job is to take a project description and create a detailed project blueprint in JSON format.

RULES:
1. Output ONLY valid JSON, nothing else. No explanation, no markdown, no backticks.
2. Every Flask project MUST include these backend files:
   - backend/config.py (database URL and app config)
   - backend/models.py (SQLAlchemy models)
   - backend/routes.py (API endpoints)
   - backend/app.py (main Flask app)
3. Every React project MUST include these frontend files:
   - frontend/package.json (dependencies)
   - frontend/src/index.js (entry point)
   - frontend/src/index.css (basic styles)
   - frontend/src/api.js (axios API client)
4. Always include a .env.example file showing required environment variables.
5. Never generate database.py separately — all database logic goes in models.py.
6. All datetime fields in routes must use .isoformat() for JSON serialization.

OUTPUT FORMAT:
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
      "description": "Flask config with DATABASE_URL from environment variables",
      "depends_on": []
    },
    {
      "path": "backend/models.py",
      "description": "SQLAlchemy models",
      "depends_on": ["backend/config.py"]
    },
    {
      "path": "backend/routes.py",
      "description": "API endpoints, all datetime fields must use .isoformat() for JSON serialization",
      "depends_on": ["backend/models.py"]
    },
    {
      "path": "backend/app.py",
      "description": "Main Flask app entry point",
      "depends_on": ["backend/config.py", "backend/models.py", "backend/routes.py"]
    },
    {
      "path": "frontend/package.json",
      "description": "React app dependencies including axios and react-scripts",
      "depends_on": []
    },
    {
      "path": "frontend/src/index.css",
      "description": "Basic global styles",
      "depends_on": []
    },
    {
      "path": "frontend/src/api.js",
      "description": "Axios API client matching all backend endpoints",
      "depends_on": []
    },
    {
      "path": "frontend/src/index.js",
      "description": "React entry point",
      "depends_on": ["frontend/src/api.js"]
    },
    {
      "path": ".env.example",
      "description": "Example environment variables",
      "depends_on": []
    }
  ],
  "database_schema": {
    "tables": ["table1", "table2"]
  },
  "api_endpoints": [
    "GET /api/resource",
    "POST /api/resource"
  ],
  "setup_instructions": [
    "Copy .env.example to .env and fill in your values",
    "pip install -r requirements.txt",
    "python backend/app.py",
    "cd frontend && npm install && npm start"
  ]
}
"""

REQUIRED_FILES = [
    "backend/config.py",
    "backend/models.py",
    "backend/routes.py",
    "backend/app.py",
    "frontend/package.json",
    "frontend/src/App.js",
    "frontend/src/index.css",
    "frontend/src/api.js",
    "frontend/src/index.js",
    ".env.example"
]

def generate_blueprint(project_description):
    """Takes a project description and returns a structured JSON blueprint."""
    print("\n🧠 Planner Agent thinking...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"Create a blueprint for: {project_description}"}
        ],
        temperature=0.3
    )

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

        print(f"✅ Blueprint generated: {blueprint['project_name']}")
        print(f"📁 Files to create: {len(blueprint['files'])}")
        print(f"🗄️  Database tables: {blueprint['database_schema']['tables']}")
        print(f"🔗 API endpoints: {len(blueprint['api_endpoints'])}")
        return blueprint

    except json.JSONDecodeError as e:
        print(f"❌ Planner failed to generate valid JSON: {e}")
        print(f"Raw output was:\n{raw}")
        return None


# Test it
if __name__ == "__main__":
    blueprint = generate_blueprint(
        "A simple todo app where users can add, complete, and delete tasks"
    )
    if blueprint:
        with open("sandbox/blueprint.json", "w") as f:
            json.dump(blueprint, f, indent=2)
        print("\n📄 Blueprint saved to sandbox/blueprint.json")