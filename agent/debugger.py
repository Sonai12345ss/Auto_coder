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
#  Decision engine chooses the right strategy.
# ─────────────────────────────────────────────

MAX_RETRIES = 3

groq1      = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2      = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3      = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
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
            if any(x in err for x in ["rate_limit", "rate-limit", "429", "quota", "503",
                                       "404", "402", "temporarily", "overloaded", "upstream"]):
                wait = min(2 ** (attempt % 4), 16)
                print(f"  ⚠️  {provider['name']} rate limited, waiting {wait}s...")
                last_error = e
                time.sleep(wait)
                continue
            last_error = e
            continue
    raise Exception(f"All providers failed. Last error: {last_error}")

def clean_code(raw):
    """Strip markdown code fences from LLM response."""
    raw = raw.strip()
    raw = re.sub(r"^```[\w]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

# ─────────────────────────────────────────────
#  RULE-BASED FIXES (no LLM needed — fast)
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

def autofix_missing_tailwind(code):
    if "cdn.tailwindcss.com" in code:
        return code, 0
    return code.replace("</head>", '    <script src="https://cdn.tailwindcss.com"></script>\n</head>'), 1

def autofix_wrong_tailwind_tag(code):
    new_code = re.sub(r'<link[^>]*cdn\.tailwindcss\.com[^>]*>', '<script src="https://cdn.tailwindcss.com"></script>', code)
    return new_code, 1 if new_code != code else 0

def autofix_missing_root_div(code):
    if '<div id="root">' in code:
        return code, 0
    return code.replace("<body>", '<body>\n    <div id="root"></div>'), 1

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
    else:
        code = "from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token\n" + code
        return code, 1
    return code, 0

def apply_rule_based_fixes(file_path, code, errors):
    """Apply all rule-based fixes that don't need LLM."""
    total_fixes = 0
    error_types = {e["type"] for e in errors}

    fixers = [
        ("css_import",         autofix_css_imports),
        ("react17_api",        autofix_react17_api),
        ("double_router",      autofix_double_router),
        ("missing_deps_array", autofix_missing_deps_array),
        ("unsafe_error_access",autofix_unsafe_error_access),
        ("missing_tailwind",   autofix_missing_tailwind),
        ("wrong_tailwind_tag", autofix_wrong_tailwind_tag),
        ("missing_root_div",   autofix_missing_root_div),
        ("missing_jwt_import", autofix_missing_jwt_import),
    ]
    for err_type, fixer in fixers:
        if err_type in error_types:
            code, n = fixer(code)
            total_fixes += n

    return code, total_fixes

# ─────────────────────────────────────────────
#  DECISION ENGINE
#  Decides HOW to fix each error type
#  instead of blindly sending everything to LLM
# ─────────────────────────────────────────────

def decide_fix_strategy(error, all_files):
    """
    For each error, decide the best fix strategy:
    - "generate_file" → create a missing file
    - "llm_fix"       → send to LLM
    - "rule_fix"      → already handled by rule-based fixes
    - "skip"          → not fixable
    """
    err_type = error["type"]
    message = error.get("message", "")
    file_path = error.get("file", "")

    if err_type == "missing_component":
        # Parse: "Imports 'ComponentName' from './path' but this component was not generated."
        name_match = re.search(r"Imports '(\w+)'", message)
        path_match = re.search(r"from '([^']+)'", message)

        if name_match and path_match:
            imported_name = name_match.group(1)
            import_rel = path_match.group(1)
            base_dir = os.path.dirname(file_path)

            # Resolve relative path to full project path
            if import_rel.startswith("../"):
                resolved = os.path.normpath(os.path.join(base_dir, import_rel + ".js"))
            elif import_rel.startswith("./"):
                resolved = os.path.normpath(os.path.join(base_dir, import_rel[2:] + ".js"))
            else:
                resolved = f"frontend/src/components/{imported_name}.js"

            resolved = resolved.replace("\\", "/")

            if resolved not in all_files:
                return "generate_file", {"component_name": imported_name, "file_path": resolved}
            return "skip", {}

        return "llm_fix", {}  # Can't parse, try LLM

    elif err_type == "missing_init_file":
        return "generate_file", {"component_name": "__init__", "file_path": "backend/__init__.py"}

    elif err_type in ("raw_fetch_instead_of_api", "syntax_error", "misplaced_db_index",
                      "missing_field", "missing_dependency", "empty_file", "bad_import",
                      "invalid_json", "critical_missing_export", "critical_missing_component",
                      "critical_missing_router", "critical_react18_missing"):
        return "llm_fix", {}

    else:
        return "rule_fix", {}

# ─────────────────────────────────────────────
#  GENERATE MISSING COMPONENT
#  Creates a working component when one is missing
# ─────────────────────────────────────────────

def generate_missing_component(component_name, file_path, all_files):
    """Generate a minimal but working React component for a missing file."""

    # Get available API functions for context
    api_code = all_files.get("frontend/src/api.js", "")
    api_exports = re.findall(r"^export const (\w+)", api_code, re.MULTILINE)
    api_list = ", ".join(api_exports) if api_exports else "check api.js"

    # Determine component type
    is_list   = component_name.endswith("List")
    is_form   = component_name.endswith("Form")
    is_detail = component_name.endswith("Detail")
    resource  = re.sub(r"(List|Form|Detail)$", "", component_name)

    # Pick correct relative import path
    depth = file_path.count("/") - 1  # e.g. frontend/src/components/X.js = 3 slashes = depth 2
    rel_api = "../api" if "components/" in file_path else "./api"

    type_rules = ""
    if is_list:
        type_rules = f"- Fetch and display {resource} items in a card grid using getTodos/get{resource}s from api.js\n- Add pagination controls\n- Link each card to detail view"
    elif is_form:
        type_rules = f"- Controlled form inputs for creating a {resource}\n- Call the appropriate create function from api.js\n- Navigate to list on success using useNavigate"
    elif is_detail:
        type_rules = f"- Fetch single {resource} by id using useParams\n- Display all fields\n- Back button"
    else:
        type_rules = f"- Display {component_name} content with proper Tailwind styling"

    prompt = f"""Generate a complete React component for {component_name}.

FILE: {file_path}
AVAILABLE API FUNCTIONS: {api_list}
IMPORT API FROM: '{rel_api}'

REQUIREMENTS:
1. Output ONLY raw code — no markdown, no backticks, no explanation
2. Tailwind CSS classes only — no inline styles, no CSS file imports
3. Import only named functions from '{rel_api}'
4. Every useEffect MUST have a dependency array []
5. Include loading, error, and empty states
6. error.response?.data?.message || error.message for errors

COMPONENT TYPE:
{type_rules}

UI STYLE:
- Wrapper: className="min-h-screen bg-gray-50 py-8 px-4"
- Cards: bg-white rounded-2xl shadow-md hover:shadow-xl transition p-6
- Buttons: bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-6 py-3 rounded-xl transition
- Loading: <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent"></div>

Generate {component_name} now:"""

    try:
        response = call_llm([{"role": "user", "content": prompt}], max_tokens=2500)
        code = clean_code(response.choices[0].message.content)
        if code and len(code) > 50:
            return code
    except Exception as e:
        print(f"    ❌ LLM failed to generate {component_name}: {e}")

    # Fallback: minimal working stub
    return f"""import React, {{ useState }} from 'react';
import {{ useNavigate }} from 'react-router-dom';

const {component_name} = () => {{
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gray-900">{component_name}</h1>
          <button onClick={{() => navigate(-1)}} className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold px-4 py-2 rounded-xl transition">
            Back
          </button>
        </div>
        {{error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm mb-4">{{error}}</div>}}
        <div className="bg-white rounded-2xl shadow-md p-8 text-center">
          <p className="text-gray-500">{component_name} — content loading.</p>
        </div>
      </div>
    </div>
  );
}};

export default {component_name};
"""

# ─────────────────────────────────────────────
#  GHOST FILE REMOVAL
#  Files that should never exist in a project
# ─────────────────────────────────────────────

GHOST_FILES = {
    "frontend/src/components/Protected.js",  # Duplicate of PrivateRoute
    "frontend/src/components/Routing.js",
    "frontend/src/components/Router.js",
    "frontend/src/components/Routes.js",
}

def remove_ghost_files(files):
    """Remove ghost files that should never be in a project."""
    removed = []
    for ghost in list(GHOST_FILES):
        if ghost in files:
            del files[ghost]
            removed.append(ghost)
    if removed:
        print(f"\n  🗑️  Removed ghost files: {removed}")
    return files, len(removed)

# ─────────────────────────────────────────────
#  LLM-BASED FIX (for complex errors)
# ─────────────────────────────────────────────

def fix_with_llm(file_path, code, errors):
    """Send broken file + errors to LLM for fixing."""
    error_descriptions = "\n".join([
        f"- [{e['type']}] line {e.get('line', '?')}: {e['message']}"
        for e in errors
    ])

    error_types = {e["type"] for e in errors}

    # Special instructions per error type
    extra = ""
    if "raw_fetch_instead_of_api" in error_types:
        extra += """
CRITICAL: Replace ALL raw fetch() calls with named imports from '../api' or '../../api'.
- Add: import { functionName } from '../api'
- The JWT token is handled automatically by the axios interceptor — do NOT manually add Authorization headers
- Replace all inline styles with Tailwind CSS classes
"""
    if any(t in error_types for t in ["critical_missing_export", "critical_missing_component", "critical_missing_router"]):
        extra += """
CRITICAL App.js fix required:
- Must have: export default function App()
- Must include BrowserRouter, Routes, Route
- Must return valid JSX with at least one Route
"""

    prompt = f"""You are an expert debugger. Fix ALL errors in this file.

FILE: {file_path}

ERRORS:
{error_descriptions}

CURRENT CODE:
{code}

RULES:
- Return ONLY the complete fixed file — no markdown fences, no explanation
- Fix ALL listed errors
- Preserve all working logic
- Tailwind CSS only — no inline styles
{extra}

FIXED CODE:"""

    response = call_llm([{"role": "user", "content": prompt}], max_tokens=4096)
    return clean_code(response.choices[0].message.content)

# ─────────────────────────────────────────────
#  MAIN DEBUGGER ENTRY POINT
# ─────────────────────────────────────────────

def debug_files(files, test_result):
    """
    Takes files dict and test_result.
    Uses decision engine to choose the right fix per error type.
    Returns fixed files dict.
    """
    if test_result["passed"]:
        print("\n✅ DEBUGGER: No errors to fix.")
        return files

    print(f"\n🔧 DEBUGGER: Fixing {test_result['error_count']} error(s)...")

    # Step 0: Remove ghost files
    files, _ = remove_ghost_files(files)

    # Group errors by file
    errors_by_file = {}
    for error in test_result["errors"]:
        fp = error["file"]
        errors_by_file.setdefault(fp, []).append(error)

    fixed_files = dict(files)

    for file_path, file_errors in errors_by_file.items():

        # ── Special case: missing backend/__init__.py ──
        if file_path == "backend/__init__.py" and file_path not in fixed_files:
            if any(e["type"] == "missing_init_file" for e in file_errors):
                print(f"\n  🔧 Generating missing {file_path}...")
                fixed_files[file_path] = (
                    "from flask_sqlalchemy import SQLAlchemy\n"
                    "from flask_jwt_extended import JWTManager\n\n"
                    "db = SQLAlchemy()\n"
                    "jwt = JWTManager()\n"
                )
                print(f"    ✅ Generated backend/__init__.py")
            continue

        # ── Decision Engine: categorize each error ──
        generate_actions = []  # Files to generate
        llm_errors = []        # Errors needing LLM
        # rule errors are handled by apply_rule_based_fixes below

        for error in file_errors:
            strategy, meta = decide_fix_strategy(error, fixed_files)
            if strategy == "generate_file":
                generate_actions.append((error, meta))
            elif strategy == "llm_fix":
                llm_errors.append(error)
            # rule_fix and skip: handled by apply_rule_based_fixes

        # ── Action: Generate missing files ──
        for error, meta in generate_actions:
            component_name = meta.get("component_name", "")
            missing_path = meta.get("file_path", "")

            if not missing_path or missing_path == "backend/__init__.py":
                continue  # Already handled above

            print(f"\n  🏗️  Decision: GENERATE '{missing_path}' (was missing)")
            generated = generate_missing_component(component_name, missing_path, fixed_files)
            fixed_files[missing_path] = generated
            print(f"    ✅ Generated {missing_path}")

        if file_path not in fixed_files:
            print(f"  ⚠️  Cannot fix {file_path} — not in generated set")
            continue

        code = fixed_files[file_path]
        print(f"\n  🔧 Fixing {file_path} ({len(file_errors)} error(s))...")

        # ── Step 1: Rule-based fixes ──
        code, rule_fixes = apply_rule_based_fixes(file_path, code, file_errors)
        if rule_fixes > 0:
            print(f"    ✅ Applied {rule_fixes} rule-based fix(es)")

        # ── Step 2: LLM fixes for complex errors ──
        llm_error_types = {
            "syntax_error", "missing_component", "misplaced_db_index",
            "missing_field", "missing_dependency", "empty_file", "bad_import",
            "invalid_json", "raw_fetch_instead_of_api",
            "critical_missing_export", "critical_missing_component",
            "critical_missing_router", "critical_react18_missing"
        }
        needs_llm = [e for e in file_errors if e["type"] in llm_error_types]

        if needs_llm:
            print(f"    🤖 LLM fixing {len(needs_llm)} complex error(s)...")
            try:
                code = fix_with_llm(file_path, code, needs_llm)
                print(f"    ✅ LLM fix applied")
            except Exception as e:
                print(f"    ❌ LLM fix failed: {e}")

        fixed_files[file_path] = code

    return fixed_files


def run_debug_loop(files, tester_fn, max_retries=MAX_RETRIES):
    """
    Full debug loop with decision engine:
    1. Remove ghost files
    2. Run tester
    3. Debug using decision engine (generate / rule-fix / LLM-fix)
    4. Re-run tester
    5. Repeat up to max_retries
    """
    from agent.tester import run_tests, format_errors_for_log

    # Remove ghost files before first test
    current_files, _ = remove_ghost_files(dict(files))
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

    # Final test
    final_result = run_tests(current_files)
    format_errors_for_log(final_result)

    if not final_result["passed"]:
        print(f"\n⚠️  {final_result['error_count']} error(s) remain after {max_retries} attempts.")
        print("   Returning best available version.")

    return current_files, final_result, attempt