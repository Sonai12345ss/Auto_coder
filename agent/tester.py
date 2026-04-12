import os
import ast
import json
import subprocess
import re

# ─────────────────────────────────────────────
#  TESTER AGENT
#  Validates every generated file in a project.
#  Returns a list of errors for the Debugger.
# ─────────────────────────────────────────────

def test_python_syntax(file_path, code):
    """Check Python file for syntax errors using AST parse."""
    try:
        ast.parse(code)
        return None  # No error
    except SyntaxError as e:
        return {
            "file": file_path,
            "type": "syntax_error",
            "message": str(e),
            "line": e.lineno,
        }

def test_python_imports(file_path, code):
    """Check for common bad import patterns in Python."""
    errors = []
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        # Check for imports that reference non-existent local modules
        if re.match(r"^from agent\.", line) and "tools" not in line and "memory" not in line and "builder" not in line and "planner" not in line:
            errors.append({
                "file": file_path,
                "type": "bad_import",
                "message": f"Suspicious local import: {line.strip()}",
                "line": i,
            })
        # Check for db.Index inside class body (indented)
        if re.match(r"^\s{4,}db\.Index\(", line):
            errors.append({
                "file": file_path,
                "type": "misplaced_db_index",
                "message": f"db.Index() must be outside class body, found indented at line {i}: {line.strip()}",
                "line": i,
            })

    # Bug 1: create_access_token used but not imported in routes.py
    if "routes.py" in file_path:
        if "create_access_token" in code and "from flask_jwt_extended import" in code:
            jwt_import_line = next((l for l in lines if "from flask_jwt_extended import" in l), "")
            if "create_access_token" not in jwt_import_line:
                errors.append({
                    "file": file_path,
                    "type": "missing_import",
                    "message": "routes.py uses create_access_token but it's not imported from flask_jwt_extended. Add it to the import line.",
                    "line": None,
                })
        elif "create_access_token" in code and "from flask_jwt_extended import" not in code:
            errors.append({
                "file": file_path,
                "type": "missing_import",
                "message": "routes.py uses create_access_token but flask_jwt_extended is never imported.",
                "line": None,
            })

    return errors

def test_js_syntax(file_path, code):
    """Check JS/JSX for common fatal errors."""
    errors = []
    lines = code.split("\n")

    # Check for individual CSS imports (causes missing file errors)
    # Whitelist: index.js importing index.css is always valid
    for i, line in enumerate(lines, 1):
        if file_path.endswith("index.js") and "index.css" in line:
            continue
        if re.match(r"^import\s+['\"]\.\/\w+\.css['\"]", line) or \
           re.match(r"^import\s+['\"]\.\.\/\w+\.css['\"]", line):
            errors.append({
                "file": file_path,
                "type": "css_import",
                "message": f"Component imports a CSS file that doesn't exist: {line.strip()}",
                "line": i,
            })

    # Check for ReactDOM.render (React 17 API, broken in React 18)
    for i, line in enumerate(lines, 1):
        if "ReactDOM.render(" in line:
            errors.append({
                "file": file_path,
                "type": "react17_api",
                "message": f"ReactDOM.render() is deprecated. Use ReactDOM.createRoot() instead. Line {i}: {line.strip()}",
                "line": i,
            })

    # Check for double BrowserRouter pattern
    if "BrowserRouter" in code and file_path.endswith("index.js"):
        errors.append({
            "file": file_path,
            "type": "double_router",
            "message": "index.js wraps App in BrowserRouter — App.js already has one. Remove it from index.js.",
            "line": None,
        })

    # Check for useEffect without dependency array
    effect_matches = list(re.finditer(r"useEffect\s*\(\s*(?:async\s*)?\(\s*\)\s*=>", code))
    for match in effect_matches:
        snippet = code[match.end():match.end()+300]
        # Find the closing of this useEffect call
        # If followed by }) with no , [] before it, it's missing deps
        if re.search(r"\}\s*\)\s*;?\s*\n", snippet):
            dep_check = re.search(r"\}\s*,\s*\[", snippet)
            if not dep_check:
                line_num = code[:match.start()].count("\n") + 1
                errors.append({
                    "file": file_path,
                    "type": "missing_deps_array",
                    "message": f"useEffect at line {line_num} is missing a dependency array [] — causes infinite re-renders.",
                    "line": line_num,
                })

    # Check for unsafe error.response access
    for i, line in enumerate(lines, 1):
        if "error.response.status" in line or "error.response.data" in line:
            if "error.response?." not in line:
                errors.append({
                    "file": file_path,
                    "type": "unsafe_error_access",
                    "message": f"Unsafe error.response access (crashes if network fails). Use error.response?.status instead. Line {i}: {line.strip()}",
                    "line": i,
                })

    # Critical App.js validations
    if "App.js" in file_path and "components/" not in file_path:
        if "export default" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_missing_export",
                "message": "App.js missing 'export default' — React will not render anything.",
                "line": None,
            })
        if "function App" not in code and "const App" not in code and "App =" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_missing_component",
                "message": "App.js missing App component definition (function App or const App).",
                "line": None,
            })
        if "BrowserRouter" not in code and "Routes" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_missing_router",
                "message": "App.js missing routing — no BrowserRouter or Routes found.",
                "line": None,
            })

    # Critical index.js validation
    if file_path.endswith("index.js") and "components/" not in file_path:
        if "createRoot" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_react18_missing",
                "message": "index.js missing createRoot — must use React 18 API, not ReactDOM.render().",
                "line": None,
            })

    # Raw fetch() detection — components must use api.js, not fetch directly
    if "components/" in file_path and file_path.endswith(".js"):
        fetch_api_calls = re.findall(r"fetch\s*\(['\"][^'\"]*\/api\/[^'\"]+['\"]", code)
        if fetch_api_calls:
            has_api_import = "from '../api'" in code or "from '../../api'" in code
            if not has_api_import:
                errors.append({
                    "file": file_path,
                    "type": "raw_fetch_instead_of_api",
                    "message": f"Component uses raw fetch() for API call instead of importing from api.js. JWT token won't be sent. Found: {fetch_api_calls[0]}",
                    "line": None,
                })

    return errors

def test_package_json(file_path, code):
    """Validate package.json is valid JSON and has required fields."""
    errors = []
    try:
        pkg = json.loads(code)
        required = ["name", "version", "scripts", "dependencies"]
        for field in required:
            if field not in pkg:
                errors.append({
                    "file": file_path,
                    "type": "missing_field",
                    "message": f"package.json missing required field: '{field}'",
                    "line": None,
                })
        if "dependencies" in pkg:
            deps = pkg["dependencies"]
            for req in ["react", "react-dom", "react-router-dom", "axios"]:
                if req not in deps:
                    errors.append({
                        "file": file_path,
                        "type": "missing_dependency",
                        "message": f"package.json missing required dependency: '{req}'",
                        "line": None,
                    })
        if "proxy" not in pkg:
            errors.append({
                "file": file_path,
                "type": "missing_proxy",
                "message": "package.json missing proxy field. Add: \"proxy\": \"http://localhost:5000\"",
                "line": None,
            })
    except json.JSONDecodeError as e:
        errors.append({
            "file": file_path,
            "type": "invalid_json",
            "message": f"package.json is not valid JSON: {e}",
            "line": None,
        })
    return errors

def test_env_example(file_path, code):
    """Check .env.example has required variables."""
    errors = []
    required_vars = ["DATABASE_URL", "SECRET_KEY", "JWT_SECRET_KEY"]
    for var in required_vars:
        if var not in code:
            errors.append({
                "file": file_path,
                "type": "missing_env_var",
                "message": f".env.example missing required variable: {var}",
                "line": None,
            })
    return errors

def test_html_file(file_path, code):
    """Check public/index.html has required elements."""
    errors = []
    if '<div id="root">' not in code and "<div id='root'>" not in code:
        errors.append({
            "file": file_path,
            "type": "missing_root_div",
            "message": "public/index.html missing <div id=\"root\"></div> — React can't mount.",
            "line": None,
        })
    if "cdn.tailwindcss.com" not in code:
        errors.append({
            "file": file_path,
            "type": "missing_tailwind",
            "message": "public/index.html missing Tailwind CDN. Add: <script src=\"https://cdn.tailwindcss.com\"></script>",
            "line": None,
        })
    # Bug 3: Tailwind loaded as <link> instead of <script> — styles won't apply
    elif '<link' in code and 'cdn.tailwindcss.com' in code:
        errors.append({
            "file": file_path,
            "type": "wrong_tailwind_tag",
            "message": "Tailwind CDN must use <script src=\"https://cdn.tailwindcss.com\"></script> NOT a <link> tag. CSS CDN link does not work.",
            "line": None,
        })
    return errors

def check_cross_file_consistency(files):
    """Check that imported components actually exist in the file list."""
    errors = []
    file_paths = set(files.keys())

    # Bug 2: backend/__init__.py must exist if any file does `from backend import ...`
    uses_backend_import = any(
        "from backend import" in code or "from backend." in code
        for code in files.values() if code
    )
    if uses_backend_import and "backend/__init__.py" not in file_paths:
        errors.append({
            "file": "backend/__init__.py",
            "type": "missing_init",
            "message": "backend/__init__.py is missing. Files use 'from backend import db, jwt' but no __init__.py exists — Python won't treat backend/ as a package and the app will crash on startup.",
            "line": None,
        })

    # Build set of component names that exist
    component_names = set()
    for path in file_paths:
        if path.startswith("frontend/src/") and path.endswith(".js"):
            name = os.path.basename(path).replace(".js", "")
            component_names.add(name)

    # Build set of named exports from api.js — imports of these are NOT missing components
    api_exports = set()
    api_code = files.get("frontend/src/api.js", "")
    if api_code:
        api_exports = set(re.findall(r"^export\s+(?:const|async function|function)\s+(\w+)", api_code, re.MULTILINE))

    # Check each JS file's imports
    for path, code in files.items():
        if not (path.endswith(".js") or path.endswith(".jsx")):
            continue
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            # Match: import Something from './Something' or import { something } from '../api'
            match = re.match(r"^import\s+\{?\s*(\w+)\s*\}?\s+from\s+['\"](.+)['\"]", line)
            if match:
                imported_name = match.group(1)
                import_path = match.group(2)
                # Only check local relative imports
                if import_path.startswith("./") or import_path.startswith("../"):
                    # Skip: known api paths — these import functions, not components
                    if re.search(r"[./]api[./]?", import_path) or import_path.endswith("/api"):
                        continue
                    # Skip: if the imported name is a known api export
                    if imported_name in api_exports:
                        continue
                    # Skip: React and router imports
                    if imported_name in (
                        "React", "useState", "useEffect", "useNavigate", "useParams",
                        "Link", "Routes", "Route", "BrowserRouter", "Navigate", "Outlet"
                    ):
                        continue
                    if imported_name not in component_names:
                        errors.append({
                            "file": path,
                            "type": "missing_component",
                            "message": f"Imports '{imported_name}' from '{import_path}' but this component was not generated.",
                            "line": i,
                        })
    return errors

def run_tests(files):
    """
    Main entry point. Takes a dict of {file_path: code_string}.
    Returns:
        {
            "passed": bool,
            "errors": [...],
            "summary": "X errors found in Y files"
        }
    """
    all_errors = []

    for file_path, code in files.items():
        if not code or not code.strip():
            all_errors.append({
                "file": file_path,
                "type": "empty_file",
                "message": "File is empty — builder failed to generate content.",
                "line": None,
            })
            continue

        # Route to appropriate tester
        if file_path.endswith(".py"):
            syntax_err = test_python_syntax(file_path, code)
            if syntax_err:
                all_errors.append(syntax_err)
            all_errors.extend(test_python_imports(file_path, code))

        elif file_path.endswith(".js") or file_path.endswith(".jsx"):
            all_errors.extend(test_js_syntax(file_path, code))

        elif file_path == "frontend/package.json":
            all_errors.extend(test_package_json(file_path, code))

        elif file_path == ".env.example":
            all_errors.extend(test_env_example(file_path, code))

        elif file_path == "frontend/public/index.html":
            all_errors.extend(test_html_file(file_path, code))

    # Cross-file checks
    all_errors.extend(check_cross_file_consistency(files))

    # Deduplicate errors by (file, line, type)
    seen = set()
    unique_errors = []
    for e in all_errors:
        key = (e["file"], e.get("line"), e["type"])
        if key not in seen:
            seen.add(key)
            unique_errors.append(e)

    passed = len(unique_errors) == 0
    files_with_errors = len(set(e["file"] for e in unique_errors))

    return {
        "passed": passed,
        "errors": unique_errors,
        "error_count": len(unique_errors),
        "summary": "All tests passed ✅" if passed else f"{len(unique_errors)} error(s) found in {files_with_errors} file(s)",
    }

def format_errors_for_log(test_result):
    """Pretty print test results to console."""
    if test_result["passed"]:
        print("\n✅ TESTER: All checks passed!")
        return

    print(f"\n❌ TESTER: {test_result['summary']}")
    for e in test_result["errors"]:
        loc = f" (line {e['line']})" if e.get("line") else ""
        print(f"  [{e['type']}] {e['file']}{loc}")
        print(f"    → {e['message']}")