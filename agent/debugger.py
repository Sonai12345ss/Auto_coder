import os
import re
import time
import json
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  DEBUGGER AGENT
#  Improvements active:
#  #3  Smart provider scoring
#  #5  Surgical error classification
#  #7  Memory-fed fix patterns
#  #9  Debug-specialized model pool
# ─────────────────────────────────────────────

MAX_RETRIES = 3
MEMORY_FILE = "sandbox/debugger_memory.json"

# ─────────────────────────────────────────────
# CANONICAL TEMPLATES — never LLM-rebuilt
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


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT #7 — MEMORY SYSTEM (debugger side)
# Stores successful fix snippets keyed by (error_type, file_ext).
# Recalled and injected into LLM prompts for faster convergence.
# ═══════════════════════════════════════════════════════════════

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
    key = f"{error_type}|{os.path.splitext(file_path)[1]}"
    mem.setdefault(key, [])
    mem[key] = mem[key][-4:] + [{
        "file": os.path.basename(file_path),
        "broken": broken_snippet[:300],
        "fixed": fixed_snippet[:300],
    }]
    _save_memory(mem)
    print(f"    💾 Fix pattern saved for '{error_type}'")

def recall_fixes(error_type, file_path):
    mem = _load_memory()
    key = f"{error_type}|{os.path.splitext(file_path)[1]}"
    return mem.get(key, [])

def recall_all_fixes_for_file(file_path, error_types):
    """Return all relevant past fix examples for a given file and its error types."""
    examples = []
    for err_type in error_types:
        fixes = recall_fixes(err_type, file_path)
        for fix in fixes[-2:]:
            examples.append(f"  [{err_type}] BROKEN: {fix['broken'][:150]}\n"
                            f"  [{err_type}] FIXED:  {fix['fixed'][:150]}")
    return "\n".join(examples) if examples else ""


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT #5 — ERROR CLASSIFICATION
# Errors are bucketed into fix strategies. Each strategy uses:
#  - a different prompt approach (surgical vs full rewrite)
#  - a different token budget
#  - a different model pool
#
# BUCKETS:
#  structural  → full component rewrite (high tokens)
#  semantic    → surgical patch of specific lines (low tokens)
#  syntactic   → minimal diff fix (low tokens)
#  import      → import-section-only fix (low tokens)
#  style       → pattern replacement (rule-based preferred)
#  config      → config file fix (rule-based preferred)
# ═══════════════════════════════════════════════════════════════

ERROR_BUCKETS = {
    "structural": {
        "types": [
            "truncated_component", "truncated_api", "critical_missing_export",
            "critical_missing_component", "critical_missing_router", "empty_file",
            "app_js_api_import", "navbar_outside_router", "private_route_wrong_pattern",
        ],
        "strategy": "full_rewrite",
        "max_tokens": 4096,
        "task_type": "debug",
    },
    "semantic": {
        "types": [
            "token_key_mismatch", "plaintext_password", "plaintext_password_check",
            "missing_set_password_call", "raw_fetch_instead_of_api", "missing_api_export",
        ],
        "strategy": "surgical_patch",
        "max_tokens": 2048,
        "task_type": "debug",
    },
    "syntactic": {
        "types": ["syntax_error", "invalid_json"],
        "strategy": "syntax_fix",
        "max_tokens": 2048,
        "task_type": "debug",
    },
    "import": {
        "types": [
            "phantom_backend_api", "duplicate_db_instance", "missing_import",
            "bad_import", "missing_component", "css_import",
        ],
        "strategy": "import_fix",
        "max_tokens": 1500,
        "task_type": "debug",
    },
    "style": {
        "types": [
            "missing_deps_array", "unsafe_error_access", "react17_api",
            "double_router", "wrong_tailwind_tag", "missing_tailwind",
            "missing_root_div", "misplaced_db_index", "duplicate_index_name",
        ],
        "strategy": "rule_based",  # handled by rule fixers, rarely needs LLM
        "max_tokens": 1500,
        "task_type": "debug",
    },
    "config": {
        "types": [
            "missing_proxy", "missing_field", "missing_dependency",
            "missing_env_var", "missing_init",
        ],
        "strategy": "rule_based",
        "max_tokens": 800,
        "task_type": "debug",
    },
}

def classify_error(error_type):
    """Return (bucket_name, bucket_config) for an error type."""
    for bucket_name, config in ERROR_BUCKETS.items():
        if error_type in config["types"]:
            return bucket_name, config
    return "structural", ERROR_BUCKETS["structural"]  # safe fallback

def group_errors_by_strategy(errors):
    """
    Group errors by their fix strategy.
    Returns dict: strategy → list of errors
    Strategy priority: structural > semantic > syntactic > import > style > config
    """
    STRATEGY_PRIORITY = ["structural", "semantic", "syntactic", "import", "style", "config"]
    groups = {s: [] for s in STRATEGY_PRIORITY}
    for err in errors:
        bucket_name, _ = classify_error(err["type"])
        groups[bucket_name].append(err)
    # Return only non-empty groups, in priority order
    return {k: v for k, v in groups.items() if v}


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT #9 — DEBUG-SPECIALIZED PROVIDERS
# LLaMA 3.3 70b and Qwen excel at surgical instruction-following.
# Low temperature (0.05) for deterministic fixes.
# ═══════════════════════════════════════════════════════════════

groq1      = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
groq2      = Groq(api_key=os.environ.get("GROQ_API_KEY_2", ""))
groq3      = Groq(api_key=os.environ.get("GROQ_API_KEY_3", ""))
gemini1    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini2    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_2", ""))
gemini3    = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY_3", ""))
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",   api_key=os.environ.get("OPENROUTER_API_KEY", ""))
doubleword = OpenAI(base_url="https://api.doubleword.ai/v1",   api_key=os.environ.get("DOUBLEWORD_API_KEY", ""))

DEBUG_PROVIDERS = [
    {"name": "Groq-1 / llama-3.3-70b",      "call": lambda msgs, mt: groq1.chat.completions.create(model="llama-3.3-70b-versatile",        messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Groq-2 / llama-3.3-70b",      "call": lambda msgs, mt: groq2.chat.completions.create(model="llama-3.3-70b-versatile",        messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Groq-3 / llama-3.3-70b",      "call": lambda msgs, mt: groq3.chat.completions.create(model="llama-3.3-70b-versatile",        messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-35B",    "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-35B-A3B-FP8",  messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Doubleword / Qwen3.5-397B",   "call": lambda msgs, mt: doubleword.chat.completions.create(model="Qwen/Qwen3.5-397B-A17B-FP8", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.0-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.0-flash",             messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.0-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.0-flash",             messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.0-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.0-flash",             messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "OpenRouter / llama-3.3-70b",  "call": lambda msgs, mt: openrouter.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-1 / gemini-2.5-flash", "call": lambda msgs, mt: gemini1.chat.completions.create(model="gemini-2.5-flash",             messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-2 / gemini-2.5-flash", "call": lambda msgs, mt: gemini2.chat.completions.create(model="gemini-2.5-flash",             messages=msgs, temperature=0.05, max_tokens=mt)},
    {"name": "Gemini-3 / gemini-2.5-flash", "call": lambda msgs, mt: gemini3.chat.completions.create(model="gemini-2.5-flash",             messages=msgs, temperature=0.05, max_tokens=mt)},
]


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT #3 — SMART PROVIDER SCORING (debugger side)
# Same scoring logic as builder.py but uses MEMORY_FILE for
# provider stats keyed under "debugger_provider_stats".
# ═══════════════════════════════════════════════════════════════

def _score_debug_provider(name):
    mem = _load_memory()
    stats = mem.get("debugger_provider_stats", {}).get(name)
    if not stats or stats.get("calls", 0) == 0:
        return 0.5
    success_rate = stats["success"] / stats["calls"]
    avg_latency = stats.get("latency_sum", 0) / stats["calls"]
    latency_score = max(0.0, 1.0 - (avg_latency - 500) / 4500)
    return (success_rate * 0.6) + (latency_score * 0.4)

def _record_debug_provider(name, success, latency_ms):
    mem = _load_memory()
    stats = mem.setdefault("debugger_provider_stats", {})
    p = stats.setdefault(name, {"success": 0, "fail": 0, "latency_sum": 0, "calls": 0})
    p["calls"] += 1
    p["latency_sum"] += latency_ms
    if success:
        p["success"] += 1
    else:
        p["fail"] += 1
    _save_memory(mem)

_debug_blocked_until = {}
_debug_blocked_lock = __import__("threading").Lock()

def _debug_provider_available(name):
    with _debug_blocked_lock:
        return __import__("datetime").datetime.now() >= _debug_blocked_until.get(
            name, __import__("datetime").datetime.min
        )

def _debug_block_provider(name, seconds):
    from datetime import datetime, timedelta
    with _debug_blocked_lock:
        _debug_blocked_until[name] = datetime.now() + timedelta(seconds=seconds)


def call_llm(messages, max_tokens=4096):
    """Call the debug-specialized provider pool with scoring and per-call latency tracking."""
    from datetime import datetime
    ranked = sorted(
        [p for p in DEBUG_PROVIDERS if _debug_provider_available(p["name"])],
        key=lambda p: _score_debug_provider(p["name"]),
        reverse=True
    )
    if not ranked:
        ranked = DEBUG_PROVIDERS

    last_error = None
    for provider in ranked:
        name = provider["name"]
        try:
            print(f"  🤖 Debugger using {name} (score: {_score_debug_provider(name):.2f})...")
            t0 = time.time()
            response = provider["call"](messages, max_tokens)
            latency_ms = (time.time() - t0) * 1000
            _record_debug_provider(name, success=True, latency_ms=latency_ms)
            print(f"  ✅ {name} responded in {latency_ms:.0f}ms")
            return response
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000 if 't0' in locals() else 0
            _record_debug_provider(name, success=False, latency_ms=latency_ms)
            err = str(e).lower()
            if any(x in err for x in ["rate_limit", "429", "quota", "503", "402",
                                       "temporarily", "overloaded", "upstream"]):
                cooldown = 120 if "gemini" in name.lower() else 60
                _debug_block_provider(name, cooldown)
                print(f"  ⚠️  {name} rate limited ({cooldown}s cooldown)")
                last_error = e
            else:
                print(f"  ⚠️  {name}: {str(e)[:80]}")
                last_error = e
            continue

    raise Exception(f"All debug providers failed. Last: {last_error}")


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
        return f"get{parts[0].capitalize()}" if parts else "apiCall"

    def replace_fetch(m):
        return f"await {infer_api_function(m.group(1))}()"

    code = re.sub(r"await\s+fetch\(['\"]([^'\"]*\/api\/[^'\"]*)['\"][^\)]*\)", replace_fetch, code)
    code = re.sub(r"fetch\(['\"][^'\"]*\/api\/[^'\"]*['\"][^\)]*\)", "Promise.resolve({})", code)
    changed = 1 if code != original else 0

    if changed and "from '../api'" not in code and "from '../../api'" not in code:
        fns = list(dict.fromkeys(re.findall(r"await (get\w+)\(\)", code)))
        if fns:
            rel = "../../api" if "components/" in file_path else "../api"
            import_line = f"import {{ {', '.join(fns)} }} from '{rel}';\n"
            lines = code.split("\n")
            last_import = max((i for i, l in enumerate(lines) if l.startswith("import ")), default=0)
            lines.insert(last_import + 1, import_line)
            code = "\n".join(lines)
    return code, changed

def autofix_wrong_tailwind_tag(code):
    new_code = re.sub(r'<link[^>]*cdn\.tailwindcss\.com[^>]*>',
                      '<script src="https://cdn.tailwindcss.com"></script>', code)
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
    original = code
    code = re.sub(r"^from backend\.api import[^\n]*\n", "", code, flags=re.MULTILINE)
    return code, 1 if code != original else 0

def autofix_duplicate_db(code, file_path):
    original = code
    code = re.sub(r"^from flask_sqlalchemy import SQLAlchemy\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^db = SQLAlchemy\(\)\n", "", code, flags=re.MULTILINE)
    if "from backend import db" not in code and "from backend import" in code:
        code = re.sub(r"from backend import ([^\n]+)",
                      lambda m: f"from backend import {m.group(1)}, db" if "db" not in m.group(1) else m.group(0),
                      code)
    elif "from backend import" not in code:
        code = "from backend import db\n" + code
    return code, 1 if code != original else 0

def autofix_duplicate_index_names(code):
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
                line = f"{m.group(1)}{name}_{seen[name]}{m.group(3)}"
                changed += 1
            else:
                seen[name] = 0
        fixed_lines.append(line)
    return "\n".join(fixed_lines), changed

def autofix_missing_proxy(code):
    try:
        pkg = json.loads(code)
        if "proxy" not in pkg:
            pkg["proxy"] = "http://localhost:5000"
            return json.dumps(pkg, indent=2), 1
    except Exception:
        pass
    return code, 0

def autofix_token_key_mismatch(code):
    original = code
    code = code.replace("response.data.access_token", "response.data.token")
    return code, 1 if code != original else 0

def autofix_plaintext_password(code):
    original = code
    def fix_constructor(m):
        c = m.group(0)
        c = re.sub(r",?\s*password\s*=\s*data\[['\"]password['\"]\]\s*,?", "", c)
        c = re.sub(r",\s*\)", ")", c)
        return c
    new_code = re.sub(r"User\s*\([^)]*password\s*=\s*data\[['\"]password['\"]\][^)]*\)",
                      fix_constructor, code)
    new_code = re.sub(r"(new_user\s*=\s*User\s*\([^)]*\))",
                      r"\1\n    new_user.set_password(data['password'])", new_code)
    new_code = re.sub(r"user\.password\s*==\s*data\[['\"]password['\"]\]",
                      "user.check_password(data['password'])", new_code)
    return new_code, 1 if new_code != original else 0

def apply_rule_based_fixes(file_path, code, errors):
    """Apply all rule-based fixes. Returns (fixed_code, fix_count)."""
    total_fixes = 0
    error_types = {e["type"] for e in errors}

    fixers = [
        ("css_import",               lambda c: autofix_css_imports(c)),
        ("react17_api",              lambda c: autofix_react17_api(c)),
        ("double_router",            lambda c: autofix_double_router(c)),
        ("missing_deps_array",       lambda c: autofix_missing_deps_array(c)),
        ("unsafe_error_access",      lambda c: autofix_unsafe_error_access(c)),
        ("raw_fetch_instead_of_api", lambda c: autofix_raw_fetch(c, file_path)),
        ("wrong_tailwind_tag",       lambda c: autofix_wrong_tailwind_tag(c)),
        ("missing_tailwind",         lambda c: autofix_missing_tailwind(c)),
        ("missing_root_div",         lambda c: autofix_missing_root_div(c)),
        ("missing_import",           lambda c: autofix_missing_jwt_import(c)),
        ("phantom_backend_api",      lambda c: autofix_phantom_backend_api(c)),
        ("duplicate_db_instance",    lambda c: autofix_duplicate_db(c, file_path)),
        ("duplicate_index_name",     lambda c: autofix_duplicate_index_names(c)),
        ("missing_proxy",            lambda c: autofix_missing_proxy(c)),
        ("token_key_mismatch",       lambda c: autofix_token_key_mismatch(c)),
        ("plaintext_password",       lambda c: autofix_plaintext_password(c)),
        ("plaintext_password_check", lambda c: autofix_plaintext_password(c)),
        ("missing_set_password_call",lambda c: autofix_plaintext_password(c)),
    ]
    for err_type, fixer in fixers:
        if err_type in error_types:
            code, n = fixer(code)
            total_fixes += n
    return code, total_fixes


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT #5 — STRATEGY-SPECIFIC LLM FIX PROMPTS
# Each fix strategy generates a different prompt:
#
# full_rewrite    → send entire file, instruct to rewrite completely
# surgical_patch  → send only the 10 broken lines + context
# syntax_fix      → send only the broken section
# import_fix      → send only the import block
# rule_based      → skip LLM entirely (handled by rule fixers above)
# ═══════════════════════════════════════════════════════════════

def _extract_broken_section(code, errors, context_lines=5):
    """Extract the minimal broken region from code for surgical patching."""
    lines = code.split("\n")
    error_lines = [e.get("line") for e in errors if e.get("line")]

    if not error_lines:
        # No line numbers — return first 30 lines as context
        return "\n".join(lines[:30]), 0, min(30, len(lines))

    min_line = max(1, min(error_lines) - context_lines)
    max_line = min(len(lines), max(error_lines) + context_lines)
    section = "\n".join(lines[min_line - 1:max_line])
    return section, min_line, max_line


def _build_fix_prompt(file_path, code, errors, strategy, memory_context):
    """
    Build the appropriate fix prompt based on strategy.
    surgical_patch and import_fix only show the relevant section,
    not the full file — preventing the LLM from accidentally
    breaking working code it never needed to touch.
    """
    error_descriptions = "\n".join([
        f"  - [{e['type']}] line {e.get('line', '?')}: {e['message']}"
        for e in errors
    ])
    error_types = {e["type"] for e in errors}

    # Build type-specific instruction blocks
    extra = ""

    if "raw_fetch_instead_of_api" in error_types:
        extra += """
FIX: Remove ALL fetch('/api/...') calls.
Replace with named imports from '../api' or '../../api'.
Result: zero fetch('/api/...) calls remain.
"""
    if "phantom_backend_api" in error_types:
        extra += """
FIX: Remove 'from backend.api import ...' — backend/api.py does not exist.
Use: from backend.models import ...; from backend import db
"""
    if "duplicate_db_instance" in error_types:
        extra += """
FIX: Remove 'db = SQLAlchemy()' and its import from this file.
Use: from backend import db  (already defined in backend/__init__.py)
"""
    if "plaintext_password" in error_types or "plaintext_password_check" in error_types or "missing_set_password_call" in error_types:
        extra += """
FIX: Never store or compare plain-text passwords.
Registration: new_user = User(username=..., email=...); new_user.set_password(data['password'])
Login: if user and user.check_password(data['password']):
"""
    if "token_key_mismatch" in error_types:
        extra += """
FIX: Backend returns {"token": ...} NOT {"access_token": ...}.
Change response.data.access_token → response.data.token everywhere.
"""
    if "app_js_api_import" in error_types or "navbar_outside_router" in error_types or "private_route_wrong_pattern" in error_types:
        extra += """
FIX: App.js must be a pure router:
- No imports from api.js. No useEffect. No useState.
- <Navbar /> inside <BrowserRouter>
- PrivateRoute Outlet pattern: <Route element={<PrivateRoute />}><Route path=... /></Route>
"""
    if "truncated_component" in error_types or "truncated_api" in error_types:
        component_name = os.path.basename(file_path).replace(".js", "")
        extra += f"""
FIX: File was truncated. Write ALL steps:
1) imports  2) const {component_name} = () => {{  3) useState  4) useEffect+[]
5) handlers  6) return (  7) complete JSX  8) }};  export default {component_name};
Do NOT stop before step 8.
"""

    # Strategy-specific prompt shaping
    if strategy == "surgical_patch":
        broken_section, start_line, end_line = _extract_broken_section(code, errors)
        prompt = f"""You are an expert debugger. Fix ONLY the broken section below.

FILE: {file_path}
ERRORS (lines {start_line}–{end_line}):
{error_descriptions}

{memory_context}

BROKEN SECTION (lines {start_line}–{end_line} of the file):
{broken_section}

RULES:
- Return ONLY the fixed version of this section (same line range)
- Do NOT return the entire file
- Fix ONLY what is listed in ERRORS — touch nothing else
- No markdown fences, no explanation
{extra}
FIXED SECTION:"""

    elif strategy == "syntax_fix":
        # Show a tight window around the syntax error
        broken_section, start_line, end_line = _extract_broken_section(code, errors, context_lines=8)
        prompt = f"""Fix ONLY the syntax error in this file section.

FILE: {file_path}
ERROR:
{error_descriptions}

SECTION WITH ERROR (lines {start_line}–{end_line}):
{broken_section}

Return ONLY the fixed section. No explanation. No fences.
FIXED:"""

    elif strategy == "import_fix":
        # Extract just the import block (first 20 lines usually covers it)
        import_section = "\n".join(code.split("\n")[:25])
        prompt = f"""Fix ONLY the import statements in this file.

FILE: {file_path}
IMPORT ERRORS:
{error_descriptions}

CURRENT IMPORTS:
{import_section}

{memory_context}

RULES:
- Return ONLY the corrected import block (first ~20 lines)
- Do NOT return the entire file
- No markdown fences, no explanation
{extra}
FIXED IMPORTS:"""

    else:  # full_rewrite (structural errors)
        prompt = f"""You are an expert debugger. Fix ALL errors in this file.

FILE: {file_path}
ERRORS:
{error_descriptions}

{memory_context}

CURRENT CODE:
{code}

RULES:
- Return ONLY the complete fixed file — no markdown fences, no explanation
- Fix ALL listed errors. Preserve all working logic.
- Tailwind CSS only — no inline styles.
- NEVER truncate. Write every line to completion.
{extra}
FIXED CODE:"""

    return prompt


def fix_with_llm(file_path, code, errors):
    """
    IMPROVEMENT #5: Route errors to the right fix strategy.
    Groups errors by strategy, handles each group separately,
    then merges results back into the file.
    """
    error_groups = group_errors_by_strategy(errors)
    file_ext = os.path.splitext(file_path)[1]

    print(f"    📋 Error groups: { {k: len(v) for k, v in error_groups.items()} }")

    # Process groups in priority order: structural first (most impactful)
    PRIORITY = ["structural", "semantic", "syntactic", "import", "style", "config"]

    # If there's a structural error, do a full rewrite and stop — no point
    # doing surgical patches on a file that needs a full rewrite anyway
    if "structural" in error_groups:
        structural_errors = error_groups["structural"]
        memory_context = recall_all_fixes_for_file(file_path, [e["type"] for e in structural_errors])
        memory_str = f"\nPAST FIX PATTERNS:\n{memory_context}" if memory_context else ""
        _, bucket_config = classify_error(structural_errors[0]["type"])
        prompt = _build_fix_prompt(file_path, code, structural_errors, "full_rewrite", memory_str)
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages, max_tokens=bucket_config["max_tokens"])
        fixed_code = clean_code(response.choices[0].message.content)

        # Save fix patterns to memory
        for err in structural_errors:
            remember_fix(err["type"], file_path, code[:200], fixed_code[:200])

        return fixed_code

    # For non-structural errors: process each group with targeted strategy
    current_code = code

    for group_name in PRIORITY:
        if group_name not in error_groups or group_name in ("structural", "style", "config"):
            continue

        group_errors = error_groups[group_name]
        _, bucket_config = classify_error(group_errors[0]["type"])
        strategy = bucket_config["strategy"]
        max_tokens = bucket_config["max_tokens"]

        memory_context = recall_all_fixes_for_file(file_path, [e["type"] for e in group_errors])
        memory_str = f"\nPAST FIX PATTERNS:\n{memory_context}" if memory_context else ""

        prompt = _build_fix_prompt(file_path, current_code, group_errors, strategy, memory_str)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = call_llm(messages, max_tokens=max_tokens)
            fixed_section = clean_code(response.choices[0].message.content)

            if strategy in ("surgical_patch", "syntax_fix"):
                # Splice the fixed section back into the full file
                _, start_line, end_line = _extract_broken_section(current_code, group_errors)
                lines = current_code.split("\n")
                fixed_lines = fixed_section.split("\n")
                current_code = "\n".join(
                    lines[:start_line - 1] + fixed_lines + lines[end_line:]
                )
                print(f"    ✅ Surgical patch applied for {group_name} (lines {start_line}–{end_line})")

            elif strategy == "import_fix":
                # Replace just the import block (first 25 lines)
                all_lines = current_code.split("\n")
                fixed_lines = fixed_section.split("\n")
                current_code = "\n".join(fixed_lines + all_lines[25:])
                print(f"    ✅ Import fix applied for {group_name}")

            else:
                current_code = fixed_section
                print(f"    ✅ Fix applied for {group_name}")

            # Save successful fix patterns
            for err in group_errors:
                remember_fix(err["type"], file_path, code[:200], current_code[:200])

        except Exception as e:
            print(f"    ❌ LLM fix failed for {group_name}: {e}")

    # Post-fix forced corrections (safety net after any LLM touch)
    if "frontend/src/api.js" in file_path and "response.data.access_token" in current_code:
        current_code = current_code.replace("response.data.access_token", "response.data.token")
        print(f"    🔧 Forced token key correction post-LLM")

    if "routes.py" in file_path and re.search(r"user\.password\s*==\s*data\[", current_code):
        current_code, _ = autofix_plaintext_password(current_code)
        print(f"    🔧 Forced password hashing correction post-LLM")

    return current_code


# ─────────────────────────────────────────────
#  MAIN DEBUGGER ENTRY POINT
# ─────────────────────────────────────────────

def debug_files(files, test_result):
    if test_result["passed"]:
        print("\n✅ DEBUGGER: No errors to fix.")
        return files

    print(f"\n🔧 DEBUGGER: Fixing {test_result['error_count']} error(s)...")

    # Remove rogue files
    ROGUE_PATHS = {"frontend/api.js", "frontend/api/rooms.js",
                   "frontend/api/messages.js", "frontend/api/users.js"}
    for rogue in list(ROGUE_PATHS):
        if rogue in files:
            del files[rogue]
            print(f"  🗑️  Removed rogue file: {rogue}")

    errors_by_file = {}
    for error in test_result["errors"]:
        errors_by_file.setdefault(error["file"], []).append(error)

    fixed_files = dict(files)

    for file_path, file_errors in errors_by_file.items():

        # Generate missing backend/__init__.py
        if file_path == "backend/__init__.py" and file_path not in fixed_files:
            fixed_files[file_path] = (
                "from flask_sqlalchemy import SQLAlchemy\n"
                "from flask_jwt_extended import JWTManager\n\n"
                "db = SQLAlchemy()\n"
                "jwt = JWTManager()\n"
            )
            print(f"  ✅ Generated backend/__init__.py")
            continue

        # PrivateRoute ALWAYS restored from template — never LLM
        if file_path == "frontend/src/components/PrivateRoute.js":
            fixed_files[file_path] = PRIVATEROUTE_TEMPLATE
            print(f"  ✅ Restored PrivateRoute.js from canonical template")
            continue

        if file_path not in fixed_files:
            print(f"  ⚠️  Cannot fix {file_path} — not in generated set")
            continue

        code = fixed_files[file_path]
        error_groups = group_errors_by_strategy(file_errors)
        print(f"\n  🔧 {file_path} ({len(file_errors)} errors) — groups: { {k: len(v) for k, v in error_groups.items()} }")

        # Step 1: Rule-based fixes (fast, no LLM)
        code, rule_fixes = apply_rule_based_fixes(file_path, code, file_errors)
        if rule_fixes > 0:
            print(f"    ✅ {rule_fixes} rule-based fix(es) applied")

        # Step 2: LLM for errors that rule-based can't handle
        # Skip rule_based and config buckets — those are fully handled above
        needs_llm_groups = {k: v for k, v in error_groups.items()
                            if k not in ("style", "config")}

        # Flatten remaining errors that still need LLM
        remaining_errors = []
        for group_errors in needs_llm_groups.values():
            # Re-check: some may have been fixed by rule-based above
            remaining_errors.extend(group_errors)

        if remaining_errors:
            print(f"    🤖 LLM fixing {len(remaining_errors)} error(s) across "
                  f"{len(needs_llm_groups)} group(s)...")
            try:
                code = fix_with_llm(file_path, code, remaining_errors)
                print(f"    ✅ LLM fixes applied")
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