import os
import time
import re
import json
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  DEBUGGER AGENT
#  Takes errors from Tester, fixes them.
#  Rule-based fixes first, LLM for complex errors.
#  Learning memory: stores successful fixes.
# ─────────────────────────────────────────────

MAX_RETRIES = 3

MEMORY_FILE = "sandbox/debugger_memory.json"

# ─────────────────────────────────────────────
# FIX 2: PRIVATEROUTE TEMPLATE
# PrivateRoute must NEVER be LLM-rebuilt.
# This is the canonical 3-line guard pattern
# using React Router v6 Outlet.
# ─────────────────────────────────────────────
PRIVATEROUTE_TEMPLATE = """\
import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';

const PrivateRoute = () => {
  const token = localStorage.getItem('token');
  return token ? <Outlet /> : <Navigate to="/login" replace />;
};

export default PrivateRoute;
"""

def _load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            return json.loads(open(MEMORY_FILE).read())
    except Exception:
        pass
    return {}

def _save_memory(mem):
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem, f, indent=2)
    except Exception:
        pass

def remember_fix(error_type, file_path, broken_snippet, fixed_snippet):
    if not broken_snippet or not fixed_snippet or broken_snippet == fixed_snippet:
        return
    mem = _load_memory()
    key = error_type
    if key not in mem:
        mem[key] = []
    mem[key] = mem[key][-4:] + [{
        "file_type": os.path.splitext(file_path)[1],
        "broken": broken_snippet[:300],
        "fixed": fixed_snippet[:300],
    }]
    _save_memory(mem)
    print(f"    💾 Fix pattern saved for '{error_type}'")

def recall_fix(error_type):
    mem = _load_memory()
    return mem.get(error_type, [])

groq1   = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2   = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3   = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3 = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1", api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

PROVIDERS = [
    {"name": "Groq-1",    "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-2",    "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Groq-3",    "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-1",  "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-2",  "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Gemini-3",  "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.0-flash", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "OpenRouter","call": lambda msgs, mt: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-35B",  "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",   messages=msgs, temperature=0.1, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B", "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.1, max_tokens=mt)},
]

def call_llm(messages, max_tokens=4096):
    last_error = None
    for attempt, provider in enumerate(PROVIDERS):
        try:
            print(f"  🤖 Debugger using {provider['name']}...")
            return provider["call"](messages, max_tokens)
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "quota", "503", "404", "402", "temporarily", "overloaded", "upstream"]):
                wait = min(2 ** (attempt % 4), 16)
                print(f"  ⚠️  {provider['name']} rate limited, waiting {wait}s...")
                last_error = e
                time.sleep(wait)
                continue
            last_error = e
            continue
    raise Exception(f"All providers failed. Last error: {last_error}")

def clean_code(raw):
    raw = raw.strip()
    raw = re.sub(r"^```[\w]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

# ─────────────────────────────────────────────
#  RULE-BASED FIXES (no LLM needed)
# ─────────────────────────────────────────────

def autofix_css_imports(code):
    lines = code.split("\n")
    fixed = [l for l in lines if not (
        re.match(r"^import\s+['\"]\.\/\w+\.css['\"]", l) or
        re.match(r"^import\s+['\"]\.\.\/\w+\.css['\"]", l)
    )]
    return "\n".join(fixed), len(lines) - len(fixed)

def autofix_react17_api(code):
    if "ReactDOM.render(" not in code:
        return code, 0
    new_code = re.sub(
        r"ReactDOM\.render\(\s*(<[\s\S]*?>)\s*,\s*document\.getElementById\(['\"]root['\"]\)\s*\)",
        lambda m: f"const root = ReactDOM.createRoot(document.getElementById('root'));\nroot.render({m.group(1)})",
        code
    )
    if "from 'react-dom/client'" not in new_code:
        new_code = new_code.replace("import ReactDOM from 'react-dom';", "import ReactDOM from 'react-dom/client';")
    return new_code, 1

def autofix_double_router(code):
    if "BrowserRouter" not in code:
        return code, 0
    code = re.sub(r",?\s*BrowserRouter\s*,?", "", code)
    code = re.sub(r"<BrowserRouter>\s*(<App\s*/>)\s*</BrowserRouter>", r"\1", code)
    return code, 1

def autofix_missing_deps_array(code):
    fixed = re.sub(
        r"(useEffect\s*\(\s*(?:async\s*)?\(\s*\)\s*=>\s*\{[^}]*\}\s*)\)",
        r"\1, [])", code
    )
    return fixed, 1 if fixed != code else 0

def autofix_unsafe_error_access(code):
    original = code
    code = code.replace("error.response.status", "error.response?.status")
    code = code.replace("error.response.data", "error.response?.data")
    return code, 1 if code != original else 0

def autofix_raw_fetch(code, file_path):
    if "fetch(" not in code:
        return code, 0
    original = code

    def infer_api_function(url):
        url = re.sub(r'\?.*', '', url).rstrip('/')
        parts = [p for p in url.split('/') if p and p != 'api']
        if not parts:
            return "apiCall"
        resource = parts[0]
        return f"get{resource.capitalize()}"

    def replace_fetch(m):
        url = m.group(1)
        fn = infer_api_function(url)
        return f"await {fn}()"

    code = re.sub(
        r"await\s+fetch\(['\"]([^'\"]*\/api\/[^'\"]*)['\"][^\)]*\)",
        replace_fetch, code
    )
    code = re.sub(
        r"fetch\(['\"][^'\"]*\/api\/[^'\"]*['\"][^\)]*\)",
        "Promise.resolve({})", code
    )
    changed = 1 if code != original else 0

    if changed and "from '../api'" not in code and "from '../../api'" not in code:
        fns = re.findall(r"await (get\w+)\(\)", code)
        if fns:
            unique_fns = list(dict.fromkeys(fns))
            rel = "../../api" if "components/" in file_path else "../api"
            import_line = f"import {{ {', '.join(unique_fns)} }} from '{rel}';\n"
            lines = code.split("\n")
            last_import = max((i for i, l in enumerate(lines) if l.startswith("import ")), default=0)
            lines.insert(last_import + 1, import_line)
            code = "\n".join(lines)

    return code, changed

def autofix_wrong_tailwind_tag(code):
    new_code = re.sub(
        r'<link[^>]*cdn\.tailwindcss\.com[^>]*>',
        '<script src="https://cdn.tailwindcss.com"></script>', code
    )
    return new_code, 1 if new_code != code else 0

def autofix_missing_jwt_import(code):
    match = re.search(r"from flask_jwt_extended import ([^\n]+)", code)
    if match:
        current = match.group(1).strip()
        if "create_access_token" not in current:
            code = code.replace(
                f"from flask_jwt_extended import {current}",
                f"from flask_jwt_extended import {current}, create_access_token"
            )
            return code, 1
    elif "create_access_token" in code:
        code = "from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token\n" + code
        return code, 1
    return code, 0

def autofix_missing_tailwind(code):
    if "cdn.tailwindcss.com" in code:
        return code, 0
    return code.replace("</head>", '    <script src="https://cdn.tailwindcss.com"></script>\n</head>'), 1

def autofix_missing_root_div(code):
    if '<div id="root">' in code:
        return code, 0
    return code.replace("<body>", '<body>\n    <div id="root"></div>'), 1

def autofix_phantom_backend_api(code):
    """Remove phantom 'from backend.api import ...' lines from routes.py."""
    original = code
    code = re.sub(r"^from backend\.api import[^\n]*\n", "", code, flags=re.MULTILINE)
    return code, 1 if code != original else 0

def autofix_duplicate_db(code, file_path):
    """Remove rogue db = SQLAlchemy() from app.py or models.py."""
    original = code
    code = re.sub(r"^from flask_sqlalchemy import SQLAlchemy\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^db = SQLAlchemy\(\)\n", "", code, flags=re.MULTILINE)
    if "from backend import db" not in code and "from backend import" in code:
        code = re.sub(r"from backend import ([^\n]+)", lambda m: f"from backend import {m.group(1)}, db" if "db" not in m.group(1) else m.group(0), code)
    elif "from backend import" not in code:
        code = "from backend import db\n" + code
    return code, 1 if code != original else 0

def autofix_duplicate_index_names(code):
    """Make db.Index names unique by appending table context."""
    seen = {}
    lines = code.split("\n")
    fixed_lines = []
    changed = 0
    for line in lines:
        m = re.match(r"(\s*db\.Index\(')(\w+)('.*)", line)
        if m:
            name = m.group(2)
            if name in seen:
                seen[name] += 1
                new_name = f"{name}_{seen[name]}"
                line = f"{m.group(1)}{new_name}{m.group(3)}"
                changed += 1
            else:
                seen[name] = 0
        fixed_lines.append(line)
    return "\n".join(fixed_lines), changed

# ─────────────────────────────────────────────
# FIX 1: MISSING PROXY AUTOFIX
# Adds "proxy": "http://localhost:5000" to
# package.json when it's absent.
# ─────────────────────────────────────────────
def autofix_missing_proxy(code):
    """Add proxy field to package.json if missing."""
    try:
        pkg = json.loads(code)
        if "proxy" not in pkg:
            pkg["proxy"] = "http://localhost:5000"
            return json.dumps(pkg, indent=2), 1
    except Exception:
        pass
    return code, 0

def apply_rule_based_fixes(file_path, code, errors):
    total_fixes = 0
    error_types = {e["type"] for e in errors}

    fixers = [
        ("css_import",         lambda c: autofix_css_imports(c)),
        ("react17_api",        lambda c: autofix_react17_api(c)),
        ("double_router",      lambda c: autofix_double_router(c)),
        ("missing_deps_array", lambda c: autofix_missing_deps_array(c)),
        ("unsafe_error_access",lambda c: autofix_unsafe_error_access(c)),
        ("raw_fetch_instead_of_api", lambda c: autofix_raw_fetch(c, file_path)),
        ("wrong_tailwind_tag", lambda c: autofix_wrong_tailwind_tag(c)),
        ("missing_tailwind",   lambda c: autofix_missing_tailwind(c)),
        ("missing_root_div",   lambda c: autofix_missing_root_div(c)),
        ("missing_import",     lambda c: autofix_missing_jwt_import(c)),
        ("phantom_backend_api",lambda c: autofix_phantom_backend_api(c)),
        ("duplicate_db_instance", lambda c: autofix_duplicate_db(c, file_path)),
        ("duplicate_index_name",  lambda c: autofix_duplicate_index_names(c)),
        # FIX 1: missing_proxy now has a rule-based autofix
        ("missing_proxy",      lambda c: autofix_missing_proxy(c)),
    ]
    for err_type, fixer in fixers:
        if err_type in error_types:
            code, n = fixer(code)
            total_fixes += n

    return code, total_fixes

# ─────────────────────────────────────────────
#  LLM-BASED FIX
# ─────────────────────────────────────────────

def fix_with_llm(file_path, code, errors):
    error_descriptions = "\n".join([
        f"- [{e['type']}] line {e.get('line', '?')}: {e['message']}"
        for e in errors
    ])

    error_types = {e["type"] for e in errors}

    # Inject past fix examples
    memory_context = ""
    for err_type in error_types:
        past_fixes = recall_fix(err_type)
        if past_fixes:
            examples = "\n".join([
                f"  Example fix for [{err_type}]:\n  BROKEN: {f['broken'][:150]}\n  FIXED: {f['fixed'][:150]}"
                for f in past_fixes[-2:]
            ])
            memory_context += f"\nPAST FIX PATTERNS (use as reference):\n{examples}\n"
            print(f"    🧠 Injecting {len(past_fixes)} past fix(es) for '{err_type}'")

    # Type-specific instructions
    extra = ""

    if "raw_fetch_instead_of_api" in error_types:
        extra += """
CRITICAL — RAW FETCH FIX:
- REMOVE every fetch() call that hits /api/... endpoints
- REPLACE with named imports from '../api' or '../../api'
- The axios interceptor in api.js handles JWT automatically
- After fix: ZERO fetch('/api/...) calls must remain
"""

    if "phantom_backend_api" in error_types:
        extra += """
CRITICAL — PHANTOM IMPORT FIX:
- REMOVE the line: from backend.api import ...
- backend/api.py does NOT exist. There is no backend API module.
- Call db and models directly in routes.py:
  from backend.models import User, Post, ...
  from backend import db
- Rewrite any calls to registerUser(), loginUser() etc. as direct db operations.
"""

    if "duplicate_db_instance" in error_types:
        extra += """
CRITICAL — DUPLICATE DB FIX:
- REMOVE: from flask_sqlalchemy import SQLAlchemy  (if in app.py or models.py)
- REMOVE: db = SQLAlchemy()  (if in app.py or models.py)
- backend/__init__.py already defines db and jwt — use those.
- For app.py: from backend import db, jwt
- For models.py: from backend import db
"""

    if "truncated_component" in error_types:
        component_name = os.path.basename(file_path).replace(".js", "")
        extra += f"""
CRITICAL — TRUNCATED COMPONENT FIX:
This component was cut off before completion. You MUST write ALL 9 steps:
1. import statements (react, react-router-dom, ../api)
2. const {component_name} = () => {{
3.   useState declarations for: data, loading, error
4.   useEffect(() => {{ fetchData(); }}, [])  — dependency array REQUIRED
5.   handler functions (handleSubmit, handleDelete, etc.)
6.   return (
7.     complete JSX with: loading spinner, error message, empty state, real content
8.   )
9. }};
10. export default {component_name};

Do NOT stop at step 5, 6, 7, or 8. Write all the way to 'export default {component_name};'
Use Tailwind CSS classes only. Page wrapper: min-h-screen bg-gray-50 py-8 px-4
"""

    if "truncated_api" in error_types:
        extra += """
CRITICAL — TRUNCATED api.js FIX:
api.js must contain ALL of these sections in order:
1. import axios from 'axios';
2. const api = axios.create({ baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000' });
3. api.interceptors.request.use(...) — injects Authorization Bearer token from localStorage
4. api.interceptors.response.use(...) — on 401: clear localStorage, redirect to /login
5. export const login = async (...) => { ... }
6. export const register = async (...) => { ... }
7. export const getUser = async () => { ... }
8. [one export const per remaining API endpoint]

Do NOT stop after the interceptors. Write EVERY endpoint function.

CRITICAL — TOKEN KEY:
When saving JWT from login response, use response.data.token NOT response.data.access_token.
The backend returns { "token": "...", "user": {...} }.
"""

    if "syntax_error" in error_types:
        extra += """
CRITICAL — SYNTAX FIX:
- Fix the syntax error at the indicated line
- Return the complete file with only the syntax fixed
"""

    # ─────────────────────────────────────────────
    # FIX 3: TOKEN KEY MISMATCH
    # Inject token-key correction hint for any
    # file that touches loginUser / login response.
    # ─────────────────────────────────────────────
    if file_path == "frontend/src/api.js" or (
        "login" in file_path.lower() and file_path.endswith(".js")
    ):
        extra += """
IMPORTANT — JWT TOKEN KEY:
- The backend /api/login route returns: { "token": "...", "user": {...} }
- When saving the JWT to localStorage, use: response.data.token
- Do NOT use response.data.access_token — that key does not exist in the response.
- Correct pattern:
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
    }
"""

    prompt = f"""You are an expert debugger. Fix ALL errors in this file.

FILE: {file_path}

ERRORS:
{error_descriptions}
{memory_context}
CURRENT CODE:
{code}

RULES:
- Return ONLY the complete fixed file — no markdown fences, no explanation
- Fix ALL listed errors
- Preserve all working logic
- Tailwind CSS only — no inline styles
- NEVER truncate the output. Write every line to completion.
{extra}
FIXED CODE:"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, max_tokens=4096)
    fixed_code = clean_code(response.choices[0].message.content)
    return fixed_code

# ─────────────────────────────────────────────
#  MAIN DEBUGGER ENTRY POINT
# ─────────────────────────────────────────────

def debug_files(files, test_result):
    if test_result["passed"]:
        print("\n✅ DEBUGGER: No errors to fix.")
        return files

    print(f"\n🔧 DEBUGGER: Fixing {test_result['error_count']} error(s)...")

    # Remove rogue files at wrong paths
    ROGUE_PATHS = {
        "frontend/api.js",
        "frontend/api/rooms.js",
        "frontend/api/messages.js",
        "frontend/api/users.js",
    }
    for rogue in list(ROGUE_PATHS):
        if rogue in files:
            del files[rogue]
            print(f"  🗑️  Removed rogue file: {rogue}")

    errors_by_file = {}
    for error in test_result["errors"]:
        fp = error["file"]
        errors_by_file.setdefault(fp, []).append(error)

    fixed_files = dict(files)

    for file_path, file_errors in errors_by_file.items():

        # Generate backend/__init__.py if missing
        if file_path == "backend/__init__.py" and file_path not in fixed_files:
            fixed_files[file_path] = (
                "from flask_sqlalchemy import SQLAlchemy\n"
                "from flask_jwt_extended import JWTManager\n\n"
                "db = SQLAlchemy()\n"
                "jwt = JWTManager()\n"
            )
            print(f"  ✅ Generated backend/__init__.py")
            continue

        # ─────────────────────────────────────────────
        # FIX 2: PRIVATEROUTE HARDGUARD
        # PrivateRoute.js must NEVER go to the LLM.
        # Always restore from the canonical template.
        # ─────────────────────────────────────────────
        if file_path == "frontend/src/components/PrivateRoute.js":
            fixed_files[file_path] = PRIVATEROUTE_TEMPLATE
            print(f"  ✅ Restored PrivateRoute.js from canonical template (LLM bypassed)")
            continue

        # Bad api import in component — fix the component, not create files
        api_import_errors = [
            e for e in file_errors
            if e["type"] == "missing_component" and re.search(r"[./]api[./]?", e.get("message", ""))
        ]
        if api_import_errors:
            for err in api_import_errors:
                print(f"  🔧 Bad api import path in {file_path} — fixing import (not creating new file)")

        if file_path not in fixed_files:
            print(f"  ⚠️  Cannot fix {file_path} — not in generated set")
            continue

        code = fixed_files[file_path]
        print(f"\n  🔧 Fixing {file_path} ({len(file_errors)} error(s))...")

        # Step 1: Rule-based fixes
        code, rule_fixes = apply_rule_based_fixes(file_path, code, file_errors)
        if rule_fixes > 0:
            print(f"    ✅ Applied {rule_fixes} rule-based fix(es)")

        # Step 2: LLM for complex errors
        needs_llm = [e for e in file_errors if e["type"] in (
            "syntax_error", "missing_component", "misplaced_db_index",
            "missing_field", "missing_dependency", "empty_file",
            "bad_import", "invalid_json", "raw_fetch_instead_of_api",
            "critical_missing_export", "critical_missing_component",
            "critical_missing_router", "critical_react18_missing",
            "phantom_backend_api", "duplicate_db_instance",
            "truncated_component",
            "truncated_api",
        )]

        if needs_llm:
            print(f"    🤖 LLM fixing {len(needs_llm)} complex error(s)...")
            original_code = code
            try:
                fixed_code = fix_with_llm(file_path, code, needs_llm)
                print(f"    ✅ LLM fix applied")

                # Post-fix validation for raw_fetch
                if any(e["type"] == "raw_fetch_instead_of_api" for e in needs_llm):
                    if "fetch(" in fixed_code and "/api/" in fixed_code:
                        print(f"    ⚠️  LLM didn't remove fetch() — applying forced rule fix...")
                        fixed_code, n = autofix_raw_fetch(fixed_code, file_path)
                        if n:
                            print(f"    ✅ Force-removed raw fetch() calls")

                # Save successful fix patterns
                for err in needs_llm:
                    err_type = err["type"]
                    line_num = err.get("line")
                    if line_num:
                        orig_lines = original_code.split("\n")
                        start = max(0, line_num - 2)
                        end = min(len(orig_lines), line_num + 2)
                        broken_snippet = "\n".join(orig_lines[start:end])
                        fixed_lines = fixed_code.split("\n")
                        fixed_snippet = "\n".join(fixed_lines[start:end]) if len(fixed_lines) > start else fixed_code[:200]
                    else:
                        broken_snippet = original_code[:200]
                        fixed_snippet = fixed_code[:200]
                    remember_fix(err_type, file_path, broken_snippet, fixed_snippet)

                code = fixed_code

            except Exception as e:
                print(f"    ❌ LLM fix failed: {e}")

        fixed_files[file_path] = code

    return fixed_files


def run_debug_loop(files, tester_fn, max_retries=MAX_RETRIES):
    from agent.tester import run_tests, format_errors_for_log

    current_files = files
    attempt = 0

    while attempt < max_retries:
        print(f"\n{'='*50}")
        print(f"🧪 TEST RUN {attempt + 1}/{max_retries}")
        print(f"{'='*50}")

        test_result = run_tests(current_files)
        format_errors_for_log(test_result)

        if test_result["passed"]:
            print(f"\n🎉 All tests passed after {attempt} fix attempt(s)!")
            return current_files, test_result, attempt

        if attempt < max_retries - 1:
            print(f"\n🔧 Running debugger (attempt {attempt + 1}/{max_retries - 1})...")
            current_files = debug_files(current_files, test_result)

        attempt += 1

    final_result = run_tests(current_files)
    format_errors_for_log(final_result)

    if not final_result["passed"]:
        print(f"\n⚠️  {final_result['error_count']} error(s) remain after {max_retries} attempts.")
        print("   Returning best available version.")

    return current_files, final_result, attempt