import os
import json
from groq import Groq
from dotenv import load_dotenv
from agent.tools import write_file, read_file, execute_python_code
from agent.memory import query_experience, add_experience

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

BUILDER_PROMPT = """
You are an expert full stack developer. Your job is to write a single file as part of a larger project.

RULES:
1. Output ONLY the raw code. No explanation, no markdown, no backticks.
2. The code must be complete and production ready.
3. Stay consistent with the blueprint provided.
4. Make sure imports match the actual file structure.

SPECIFIC RULES BY FILE TYPE:

For backend/config.py:
- Read DATABASE_URL from environment variables using os.environ.get()
- Include DEBUG, SECRET_KEY, and SQLALCHEMY_DATABASE_URI settings
- Use a default SQLite URL as fallback for development

For backend/models.py:
- Use Flask-SQLAlchemy
- Always add a to_dict() method to every model
- In to_dict(), convert datetime fields using .isoformat()

For backend/routes.py:
- You MUST generate ALL endpoints listed in the blueprint api_endpoints
- For every resource, generate GET (list), GET (single), POST (create), PUT (update), DELETE endpoints
- Never skip any endpoint from the blueprint

For backend/app.py:
- Import config from backend.config
- Register blueprints from backend.routes
- Initialize db with app

For frontend/package.json:
- Include react, react-dom, react-scripts, axios as dependencies
- Include start, build, test scripts

For frontend/src/index.css:
- Write clean minimal CSS, basic body and app styles only

For .env.example:
- Include DATABASE_URL, SECRET_KEY, DEBUG, FLASK_ENV

For frontend/src/api.js:
- Use axios with baseURL pointing to http://localhost:5000
- Export individual functions for each API endpoint
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
Database tables: {blueprint['database_schema']['tables']}
API endpoints: {blueprint['api_endpoints']}

File to write: {file_path}
Purpose: {file_description}

Dependencies already written:
{dependency_context if dependency_context else "None"}

{memory_context}

Write the complete code for {file_path} now.
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
            temperature=0.2
        )

        code = response.choices[0].message.content.strip()

        # Clean up backticks if model adds them
        if "```" in code:
            for lang in ["```python", "```javascript", "```html", "```json", "```"]:
                if lang in code:
                    code = code.split(lang)[1].split("```")[0].strip()
                    break

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
                # Save to memory
                add_experience(file_description, code, error=last_error)
                return code
            else:
                last_error = result["stderr"]
                print(f"  ❌ Syntax error: {last_error[:100]}")
                history.append({"role": "assistant", "content": code})
                history.append({"role": "user", "content": f"Syntax error:\n{last_error}\nFix it."})
        else:
            # For JS/HTML files just accept the output
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
    # Files with no dependencies get built first
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

## Setup
"""
    for step in blueprint["setup_instructions"]:
        readme += f"- {step}\n"

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