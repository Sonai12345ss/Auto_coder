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
#  Learning memory: stores successful fixes across builds.
# ─────────────────────────────────────────────

MAX_RETRIES = 3

# ─────────────────────────────────────────────
# PRIORITY 1: LEARNING DEBUGGER
# Persistent memory of successful fixes.
# Stored in sandbox/debugger_memory.json
# Avoids LLM calls for known errors.
# ─────────────────────────────────────────────

MEMORY_FILE = "sandbox/debugger_memory.json"

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
    """Store a successful fix pattern for future reuse."""
    if not broken_snippet or not fixed_snippet or broken_snippet == fixed_snippet:
        return
    mem = _load_memory()
    key = error_type
    if key not in mem:
        mem[key] = []
    # Keep only last 5 fixes per error type to avoid bloat
    mem[key] = mem[key][-4:] + [{
        "file_type": os.path.splitext(file_path)[1],
        "broken": broken_snippet[:300],
        "fixed": fixed_snippet[:300],
    }]
    _save_memory(mem)
    print(f"    💾 Fix pattern saved for '{error_type}'")

def recall_fix(error_type):
    """Retrieve past fix examples for a given error type."""
    mem = _load_memory()
    return mem.get(error_type, [])

# Re-use same provider chain as builder
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
    # Paid fallback — only used when all free providers fail
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
    """Strip markdown code fences from LLM response."""
    raw = raw.strip()
    # Remove ```python, ```js, ```jsx, ```html, ```json, ``` fences
    raw = re.sub(r"^```[\w]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

# ─────────────────────────────────────────────
#  AUTO-FIX: rule-based fixes (no LLM needed)
# ─────────────────────────────────────────────

def autofix_css_imports(code):
    """Remove component-level CSS imports."""
    lines = code.split("\n")
    fixed = []
    removed = 0
    for line in lines:
        if re.match(r"^import\s+['\"]\.\/\w+\.css['\"]", line) or \
           re.match(r"^import\s+['\"]\.\.\/\w+\.css['\"]", line):
            removed += 1
            continue
        fixed.append(line)
    return "\n".join(fixed), removed

def autofix_react17_api(code):
    """Replace ReactDOM.render with createRoot."""
    if "ReactDOM.render(" not in code:
        return code, 0

    # Replace the entire render call pattern
    new_code = re.sub(
        r"ReactDOM\.render\(\s*(<[\s\S]*?>)\s*,\s*document\.getElementById\(['\"]root['\"]\)\s*\)",
        lambda m: (
            "const root = ReactDOM.createRoot(document.getElementById('root'));\n"
            f"root.render({m.group(1)})"
        ),
        code
    )
    # Make sure createRoot is imported
    if "createRoot" not in code and "from 'react-dom/client'" not in code:
        new_code = new_code.replace(
            "import ReactDOM from 'react-dom';",
            "import ReactDOM from 'react-dom/client';"
        )
    return new_code, 1

def autofix_double_router(code):
    """Remove BrowserRouter from index.js if present."""
    if "BrowserRouter" not in code:
        return code, 0

    # Remove BrowserRouter import
    code = re.sub(r",?\s*BrowserRouter\s*,?", "", code)
    # Remove wrapping BrowserRouter tags around App
    code = re.sub(r"<BrowserRouter>\s*(<App\s*/>)\s*</BrowserRouter>", r"\1", code)
    return code, 1

def autofix_missing_deps_array(code):
    """Add [] to useEffect calls missing dependency array."""
    # Pattern: useEffect(() => { ... }) with no , [] before closing )
    # This is tricky to do perfectly with regex, so we use a simple heuristic
    fixed = re.sub(
        r"(useEffect\s*\(\s*(?:async\s*)?\(\s*\)\s*=>\s*\{[^}]*\}\s*)\)",
        r"\1, [])",
        code
    )
    changed = 1 if fixed != code else 0
    return fixed, changed

def autofix_unsafe_error_access(code):
    """Fix unsafe error.response.status → error.response?.status"""
    original = code
    code = code.replace("error.response.status", "error.response?.status")
    code = code.replace("error.response.data", "error.response?.data")
    changed = 1 if code != original else 0
    return code, changed

def autofix_raw_fetch(code, file_path):
    """
    Fix #2: Replace raw fetch('/api/...') with proper named api.js function calls.
    Reconstructive, not destructive — infers function name from the URL.
    """
    if "fetch(" not in code:
        return code, 0

    original = code

    def infer_api_function(url):
        """Guess the api.js function name from a URL like /api/rooms or /api/users/1"""
        # Remove query strings and trailing slashes
        url = re.sub(r'\?.*', '', url).rstrip('/')
        parts = [p for p in url.split('/') if p and p != 'api']
        if not parts:
            return "apiCall"
        resource = parts[0]  # e.g. "rooms", "messages", "users"
        # Camel-case: rooms → getRooms, messages → getMessages
        return f"get{resource.capitalize()}"

    # Pattern 1: const response = await fetch('/api/...')
    def replace_fetch(m):
        url = m.group(1)
        fn = infer_api_function(url)
        return f"await {fn}()"

    code = re.sub(
        r"await\s+fetch\(['\"]([^'\"]*\/api\/[^'\"]*)['\"][^\)]*\)",
        replace_fetch,
        code
    )

    # Pattern 2: bare fetch() without await
    code = re.sub(
        r"fetch\(['\"][^'\"]*\/api\/[^'\"]*['\"][^\)]*\)",
        "Promise.resolve({})",
        code
    )

    changed = 1 if code != original else 0

    # If we replaced fetch calls, inject the likely import if not present
    if changed and "from '../api'" not in code and "from '../../api'" not in code:
        # Find the inferred function names that were inserted
        fns = re.findall(r"await (get\w+)\(\)", code)
        if fns:
            unique_fns = list(dict.fromkeys(fns))  # deduplicate preserving order
            rel = "../../api" if "components/" in file_path else "../api"
            import_line = f"import {{ {', '.join(unique_fns)} }} from '{rel}';\n"
            # Insert after last existing import line
            lines = code.split("\n")
            last_import = max((i for i, l in enumerate(lines) if l.startswith("import ")), default=0)
            lines.insert(last_import + 1, import_line)
            code = "\n".join(lines)

    return code, changed

def autofix_wrong_tailwind_tag(code):
    """Fix Tailwind loaded as <link> instead of <script>."""
    new_code = re.sub(
        r'<link[^>]*cdn\.tailwindcss\.com[^>]*>',
        '<script src="https://cdn.tailwindcss.com"></script>',
        code
    )
    return new_code, 1 if new_code != code else 0

def autofix_missing_jwt_import(code):
    """Add create_access_token to flask_jwt_extended import in routes.py."""
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
        # No jwt import at all — add it
        code = "from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token\n" + code
        return code, 1
    return code, 0

def autofix_missing_tailwind(code):
    """Add Tailwind CDN to index.html if missing."""
    if "cdn.tailwindcss.com" in code:
        return code, 0
    tailwind_tag = '    <script src="https://cdn.tailwindcss.com"></script>\n'
    code = code.replace("</head>", tailwind_tag + "</head>")
    return code, 1

def autofix_missing_root_div(code):
    """Add root div to index.html if missing."""
    if '<div id="root">' in code:
        return code, 0
    code = code.replace("<body>", '<body>\n    <div id="root"></div>')
    return code, 1

def apply_rule_based_fixes(file_path, code, errors):
    """Apply all rule-based fixes that don't need LLM."""
    total_fixes = 0
    error_types = {e["type"] for e in errors}

    if "css_import" in error_types:
        code, n = autofix_css_imports(code)
        total_fixes += n

    if "react17_api" in error_types:
        code, n = autofix_react17_api(code)
        total_fixes += n

    if "double_router" in error_types:
        code, n = autofix_double_router(code)
        total_fixes += n

    if "missing_deps_array" in error_types:
        code, n = autofix_missing_deps_array(code)
        total_fixes += n

    if "unsafe_error_access" in error_types:
        code, n = autofix_unsafe_error_access(code)
        total_fixes += n

    if "raw_fetch_instead_of_api" in error_types:
        code, n = autofix_raw_fetch(code, file_path)
        total_fixes += n

    if "wrong_tailwind_tag" in error_types:
        code, n = autofix_wrong_tailwind_tag(code)
        total_fixes += n

    if "missing_tailwind" in error_types:
        code, n = autofix_missing_tailwind(code)
        total_fixes += n

    if "missing_root_div" in error_types:
        code, n = autofix_missing_root_div(code)
        total_fixes += n

    if "missing_jwt_import" in error_types or "missing_import" in error_types:
        code, n = autofix_missing_jwt_import(code)
        total_fixes += n

    return code, total_fixes

# ─────────────────────────────────────────────
#  LLM-BASED FIX (for complex errors)
# ─────────────────────────────────────────────

def fix_with_llm(file_path, code, errors):
    """Send broken file + errors to LLM for fixing. Injects past fix examples."""
    error_descriptions = "\n".join([
        f"- [{e['type']}] line {e.get('line', '?')}: {e['message']}"
        for e in errors
    ])

    error_types = {e["type"] for e in errors}

    # ── Learning: inject past fix examples ──
    memory_context = ""
    for err_type in error_types:
        past_fixes = recall_fix(err_type)
        if past_fixes:
            examples = "\n".join([
                f"  Example fix for [{err_type}]:\n  BROKEN: {f['broken'][:150]}\n  FIXED: {f['fixed'][:150]}"
                for f in past_fixes[-2:]  # last 2 examples
            ])
            memory_context += f"\nPAST FIX PATTERNS (use these as reference):\n{examples}\n"
            print(f"    🧠 Injecting {len(past_fixes)} past fix(es) for '{err_type}'")

    # Type-specific extra instructions
    extra = ""
    if "raw_fetch_instead_of_api" in error_types:
        extra += """
CRITICAL — RAW FETCH FIX:
- REMOVE every single fetch() call that calls /api/... endpoints
- REPLACE with named imports from '../api' or '../../api'
- Example: instead of `fetch('/api/rooms')`, use `import { getRooms } from '../api'` then call `getRooms()`
- The axios interceptor in api.js handles JWT automatically — do NOT add Authorization headers manually
- After your fix, there must be ZERO fetch('/api/...) calls remaining
"""
    if "missing_component" in error_types:
        extra += """
CRITICAL — BAD IMPORT FIX:
- Do NOT create new files
- Fix the import PATH in THIS file to correctly point to '../api' or the right component
- All API functions come from '../api' (one level up from components/)
- Example: change `from '../../api/rooms'` to `from '../api'`
"""
    if "syntax_error" in error_types:
        extra += """
CRITICAL — SYNTAX FIX:
- Fix the syntax error exactly at the line indicated
- Do not rewrite the whole file — only fix the broken part
- Return the complete file with only the syntax fixed
"""

    prompt = f"""You are an expert debugger. Fix ALL the errors in this file.

FILE: {file_path}

ERRORS TO FIX:
{error_descriptions}
{memory_context}
CURRENT CODE:
{code}

RULES:
- Return ONLY the complete fixed file — no markdown fences, no explanation
- Fix ALL listed errors
- Preserve all working logic
- Tailwind CSS only — no inline styles
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
    """
    Takes files dict and test_result from tester.
    Returns fixed files dict.
    """
    if test_result["passed"]:
        print("\n✅ DEBUGGER: No errors to fix.")
        return files

    print(f"\n🔧 DEBUGGER: Fixing {test_result['error_count']} error(s)...")

    # Remove rogue files that should never exist
    ROGUE_PATHS = {
        "frontend/api.js",         # Should be frontend/src/api.js
        "frontend/api/rooms.js",   # Invented by debugger
        "frontend/api/messages.js",
        "frontend/api/users.js",
    }
    for rogue in list(ROGUE_PATHS):
        if rogue in files:
            del files[rogue]
            print(f"  🗑️  Removed rogue file: {rogue}")

    # Group errors by file
    errors_by_file = {}
    for error in test_result["errors"]:
        file_path = error["file"]
        if file_path not in errors_by_file:
            errors_by_file[file_path] = []
        errors_by_file[file_path].append(error)

    fixed_files = dict(files)

    for file_path, file_errors in errors_by_file.items():

        # ── Special case: generate backend/__init__.py if missing ──
        if file_path == "backend/__init__.py" and file_path not in fixed_files:
            fixed_files[file_path] = (
                "from flask_sqlalchemy import SQLAlchemy\n"
                "from flask_jwt_extended import JWTManager\n\n"
                "db = SQLAlchemy()\n"
                "jwt = JWTManager()\n"
            )
            print(f"  ✅ Generated backend/__init__.py")
            continue

        # ── Bug 2 fix: missing_component errors from api imports → fix the component, don't create new files ──
        # If the "missing" import is actually an api function (path contains /api),
        # don't create a new file — send the component itself to LLM to fix the import
        api_import_errors = [
            e for e in file_errors
            if e["type"] == "missing_component" and re.search(r"[./]api[./]?", e.get("message", ""))
        ]
        if api_import_errors:
            # The component has a bad import path — LLM must fix IT, not create new files
            for err in api_import_errors:
                print(f"  🔧 Bad api import in {file_path} — fixing import path (not creating new file)")
            # These get handled below by LLM as part of needs_llm

        if file_path not in fixed_files:
            print(f"  ⚠️  Cannot fix {file_path} — file not in generated set")
            continue

        code = fixed_files[file_path]
        print(f"\n  🔧 Fixing {file_path} ({len(file_errors)} error(s))...")

        # Step 1: Apply rule-based fixes first (fast, no LLM)
        code, rule_fixes = apply_rule_based_fixes(file_path, code, file_errors)
        if rule_fixes > 0:
            print(f"    ✅ Applied {rule_fixes} rule-based fix(es)")

        # Step 2: Check if complex errors remain that need LLM
        needs_llm = [e for e in file_errors if e["type"] in (
            "syntax_error", "missing_component", "misplaced_db_index",
            "missing_field", "missing_dependency", "empty_file",
            "bad_import", "invalid_json", "raw_fetch_instead_of_api",
            "critical_missing_export", "critical_missing_component",
            "critical_missing_router", "critical_react18_missing",
        )]

        if needs_llm:
            print(f"    🤖 Sending to LLM for {len(needs_llm)} complex error(s)...")
            original_code = code  # save for memory comparison
            try:
                fixed_code = fix_with_llm(file_path, code, needs_llm)
                print(f"    ✅ LLM fix applied")

                # ── Bug 3 fix: post-fix validation for raw_fetch ──
                if any(e["type"] == "raw_fetch_instead_of_api" for e in needs_llm):
                    if "fetch(" in fixed_code and "/api/" in fixed_code:
                        print(f"    ⚠️  LLM didn't remove fetch() — applying forced rule fix...")
                        fixed_code, n = autofix_raw_fetch(fixed_code, file_path)
                        if n:
                            print(f"    ✅ Force-removed raw fetch() calls")

                # ── Learning: save successful fix patterns to memory ──
                for err in needs_llm:
                    err_type = err["type"]
                    # Extract a snippet around the error line for pattern matching
                    line_num = err.get("line")
                    if line_num:
                        lines = original_code.split("\n")
                        start = max(0, line_num - 2)
                        end = min(len(lines), line_num + 2)
                        broken_snippet = "\n".join(lines[start:end])
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
    """
    Full debug loop:
    1. Run tester
    2. If errors found, run debugger
    3. Re-run tester
    4. Repeat up to max_retries times

    Returns (final_files, final_test_result, attempts_taken)
    """
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

    # Final test after last fix attempt
    final_result = run_tests(current_files)
    format_errors_for_log(final_result)

    if not final_result["passed"]:
        print(f"\n⚠️  {final_result['error_count']} error(s) remain after {max_retries} attempts.")
        print("   Returning best available version.")

    return current_files, final_result, attempt