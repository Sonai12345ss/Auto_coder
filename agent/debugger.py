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
    """Remove component-level CSS imports."""
    lines = code.split("\n")
    fixed = [l for l in lines if not (
        re.match(r"^import\s+['\"]\.\/\w+\.css['\"]", l) or
        re.match(r"^import\s+['\"]\.\.\/\w+\.css['\"]", l)
    )]
    removed = len(lines) - len(fixed)
    return "\n".join(fixed), removed

def autofix_react17_api(code):
    """Replace ReactDOM.render with createRoot."""
    if "ReactDOM.render(" not in code:
        return code, 0
    new_code = re.sub(
        r"ReactDOM\.render\(\s*(<[\s\S]*?>)\s*,\s*document\.getElementById\(['\"]root['\"]\)\s*\)",
        lambda m: (
            "const root = ReactDOM.createRoot(document.getElementById('root'));\n"
            f"root.render({m.group(1)})"
        ),
        code
    )
    if "from 'react-dom/client'" not in new_code:
        new_code = new_code.replace(
            "import ReactDOM from 'react-dom';",
            "import ReactDOM from 'react-dom/client';"
        )
    return new_code, 1

def autofix_double_router(code):
    """Remove BrowserRouter from index.js."""
    if "BrowserRouter" not in code:
        return code, 0
    code = re.sub(r",?\s*BrowserRouter\s*,?", "", code)
    code = re.sub(r"<BrowserRouter>\s*(<App\s*/>)\s*</BrowserRouter>", r"\1", code)
    return code, 1

def autofix_missing_deps_array(code):
    """Add [] to useEffect calls missing dependency array."""
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
    return code, (1 if code != original else 0)

def autofix_missing_tailwind(code):
    """Add Tailwind CDN script tag to index.html if missing."""
    if "cdn.tailwindcss.com" in code:
        return code, 0
    tailwind_tag = '    <script src="https://cdn.tailwindcss.com"></script>\n'
    code = code.replace("</head>", tailwind_tag + "</head>")
    return code, 1

def autofix_wrong_tailwind_tag(code):
    """Fix Tailwind CDN from <link> to <script> tag."""
    # Replace any <link ...tailwindcss...> with the correct script tag
    new_code = re.sub(
        r'<link[^>]*cdn\.tailwindcss\.com[^>]*>',
        '<script src="https://cdn.tailwindcss.com"></script>',
        code
    )
    changed = 1 if new_code != code else 0
    return new_code, changed

def autofix_missing_root_div(code):
    """Add root div to index.html if missing."""
    if '<div id="root">' in code:
        return code, 0
    code = code.replace("<body>", '<body>\n    <div id="root"></div>')
    return code, 1

def autofix_missing_jwt_import(code):
    """Add create_access_token to flask_jwt_extended import in routes.py."""
    # Find existing flask_jwt_extended import and add create_access_token
    match = re.search(r"from flask_jwt_extended import ([^\n]+)", code)
    if match:
        current_imports = match.group(1).strip()
        if "create_access_token" not in current_imports:
            new_imports = current_imports + ", create_access_token"
            code = code.replace(
                f"from flask_jwt_extended import {current_imports}",
                f"from flask_jwt_extended import {new_imports}"
            )
            return code, 1
    else:
        # No flask_jwt_extended import at all — add one after other imports
        code = "from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token\n" + code
        return code, 1
    return code, 0

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

    if "missing_tailwind" in error_types:
        code, n = autofix_missing_tailwind(code)
        total_fixes += n

    # ── NEW: Fix wrong Tailwind <link> tag ──
    if "wrong_tailwind_tag" in error_types:
        code, n = autofix_wrong_tailwind_tag(code)
        total_fixes += n

    if "missing_root_div" in error_types:
        code, n = autofix_missing_root_div(code)
        total_fixes += n

    # ── NEW: Fix missing create_access_token import in routes.py ──
    if "missing_jwt_import" in error_types:
        code, n = autofix_missing_jwt_import(code)
        total_fixes += n

    return code, total_fixes

# ─────────────────────────────────────────────
#  LLM-BASED FIX (for complex errors)
# ─────────────────────────────────────────────

def fix_with_llm(file_path, code, errors):
    """Send broken file + errors to LLM for fixing."""
    error_descriptions = "\n".join([
        f"- [{e['type']}] line {e.get('line', '?')}: {e['message']}"
        for e in errors
    ])

    prompt = f"""You are an expert debugger. Fix ALL the errors in this file.

FILE: {file_path}

ERRORS TO FIX:
{error_descriptions}

CURRENT CODE:
{code}

RULES:
- Return ONLY the complete fixed file content
- Do NOT include markdown code fences (no ```)
- Do NOT include any explanation
- Fix ALL listed errors
- Do not change any working logic
- Preserve all existing functionality

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

    # Group errors by file
    errors_by_file = {}
    for error in test_result["errors"]:
        file_path = error["file"]
        if file_path not in errors_by_file:
            errors_by_file[file_path] = []
        errors_by_file[file_path].append(error)

    fixed_files = dict(files)

    for file_path, file_errors in errors_by_file.items():
        # ── NEW: Handle missing_init_file — generate backend/__init__.py ──
        if file_path == "backend/__init__.py" and file_path not in fixed_files:
            # Check if the error is that it's missing entirely
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

        if file_path not in fixed_files:
            print(f"  ⚠️  Cannot fix {file_path} — file not in generated set")
            continue

        code = fixed_files[file_path]
        print(f"\n  🔧 Fixing {file_path} ({len(file_errors)} error(s))...")

        # Step 1: Rule-based fixes (fast, no LLM)
        code, rule_fixes = apply_rule_based_fixes(file_path, code, file_errors)
        if rule_fixes > 0:
            print(f"    ✅ Applied {rule_fixes} rule-based fix(es)")

        # Step 2: LLM for complex errors that can't be rule-fixed
        needs_llm = [e for e in file_errors if e["type"] in (
            "syntax_error", "missing_component", "misplaced_db_index",
            "missing_field", "missing_dependency", "empty_file",
            "bad_import", "invalid_json"
        )]

        if needs_llm:
            print(f"    🤖 Sending to LLM for {len(needs_llm)} complex error(s)...")
            try:
                code = fix_with_llm(file_path, code, needs_llm)
                print(f"    ✅ LLM fix applied")
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

    # Final test after last attempt
    final_result = run_tests(current_files)
    format_errors_for_log(final_result)

    if not final_result["passed"]:
        print(f"\n⚠️  {final_result['error_count']} error(s) remain after {max_retries} attempts.")
        print("   Returning best available version.")

    return current_files, final_result, attempt