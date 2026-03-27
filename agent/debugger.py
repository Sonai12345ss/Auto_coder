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
#  Takes errors from Tester, fixes them using LLM.
#  Loops up to MAX_RETRIES times until all pass.
# ─────────────────────────────────────────────

MAX_RETRIES = 3

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

    if "missing_tailwind" in error_types:
        code, n = autofix_missing_tailwind(code)
        total_fixes += n

    if "missing_root_div" in error_types:
        code, n = autofix_missing_root_div(code)
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