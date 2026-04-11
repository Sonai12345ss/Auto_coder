import os
import ast
import json
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
        return None
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
        # Check for db.Index inside class body (indented)
        if re.match(r"^\s{4,}db\.Index\(", line):
            errors.append({
                "file": file_path,
                "type": "misplaced_db_index",
                "message": f"db.Index() must be outside class body, found indented at line {i}: {line.strip()}",
                "line": i,
            })

    # Check routes.py uses create_access_token but doesn't import it
    if "routes.py" in file_path and "create_access_token" in code:
        jwt_match = re.search(r"from flask_jwt_extended import ([^\n]+)", code)
        if jwt_match:
            if "create_access_token" not in jwt_match.group(1):
                errors.append({
                    "file": file_path,
                    "type": "missing_jwt_import",
                    "message": "routes.py uses create_access_token but doesn't import it from flask_jwt_extended",
                    "line": None,
                })
        else:
            errors.append({
                "file": file_path,
                "type": "missing_jwt_import",
                "message": "routes.py uses create_access_token but has no flask_jwt_extended import",
                "line": None,
            })

    return errors

def test_js_syntax(file_path, code):
    """Check JS/JSX for common fatal errors."""
    errors = []
    lines = code.split("\n")

    # Check for individual CSS imports
    for i, line in enumerate(lines, 1):
        if re.match(r"^import\s+['\"]\.\/\w+\.css['\"]", line) or \
           re.match(r"^import\s+['\"]\.\.\/\w+\.css['\"]", line):
            errors.append({
                "file": file_path,
                "type": "css_import",
                "message": f"Component imports a CSS file that doesn't exist: {line.strip()}",
                "line": i,
            })

    # Check for ReactDOM.render (React 17 API)
    for i, line in enumerate(lines, 1):
        if "ReactDOM.render(" in line:
            errors.append({
                "file": file_path,
                "type": "react17_api",
                "message": f"ReactDOM.render() is deprecated in React 18. Use createRoot. Line {i}: {line.strip()}",
                "line": i,
            })

    # Check for double BrowserRouter in index.js
    if "BrowserRouter" in code and file_path.endswith("index.js"):
        errors.append({
            "file": file_path,
            "type": "double_router",
            "message": "index.js wraps App in BrowserRouter — App.js already has one.",
            "line": None,
        })

    # Check for useEffect without dependency array
    effect_matches = list(re.finditer(r"useEffect\s*\(\s*(?:async\s*)?\(\s*\)\s*=>", code))
    for match in effect_matches:
        snippet = code[match.end():match.end()+300]
        if re.search(r"\}\s*\)\s*;?\s*\n", snippet):
            if not re.search(r"\}\s*,\s*\[", snippet):
                line_num = code[:match.start()].count("\n") + 1
                errors.append({
                    "file": file_path,
                    "type": "missing_deps_array",
                    "message": f"useEffect at line {line_num} is missing a dependency array [].",
                    "line": line_num,
                })

    # Check for unsafe error.response access
    for i, line in enumerate(lines, 1):
        if ("error.response.status" in line or "error.response.data" in line) and \
           "error.response?." not in line:
            errors.append({
                "file": file_path,
                "type": "unsafe_error_access",
                "message": f"Unsafe error.response access. Use error.response?.status instead. Line {i}: {line.strip()}",
                "line": i,
            })

    # ── NEW: Critical App.js export validation ──
    if "App.js" in file_path and "components/" not in file_path:
        if "export default" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_missing_export",
                "message": "App.js missing 'export default' — React will not render.",
                "line": None,
            })
        if "function App" not in code and "const App" not in code and "App =" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_missing_component",
                "message": "App.js missing App component definition.",
                "line": None,
            })
        if "BrowserRouter" not in code and "Routes" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_missing_router",
                "message": "App.js missing routing (BrowserRouter/Routes).",
                "line": None,
            })

    # ── NEW: Critical index.js validation ──
    if file_path.endswith("index.js") and "components/" not in file_path:
        if "createRoot" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_react18_missing",
                "message": "index.js missing createRoot — must use React 18 API.",
                "line": None,
            })

    # ── NEW: Raw fetch() instead of api.js functions ──
    # Catches components that bypass the JWT interceptor by using raw fetch
    if "components/" in file_path and file_path.endswith(".js"):
        if "fetch(" in code:
            # Check if fetch is calling an /api/... endpoint directly
            fetch_api_calls = re.findall(r"fetch\s*\(['\"]\/api\/[^'\"]+['\"]", code)
            if fetch_api_calls:
                # And it's not importing from api.js
                has_api_import = "from '../api'" in code or "from '../../api'" in code
                if not has_api_import:
                    errors.append({
                        "file": file_path,
                        "type": "raw_fetch_instead_of_api",
                        "message": f"Component calls {fetch_api_calls[0]} using raw fetch() instead of importing from api.js. JWT token won't be sent.",
                        "line": None,
                    })

    return errors

def test_package_json(file_path, code):
    """Validate package.json is valid JSON and has required fields."""
    errors = []
    try:
        pkg = json.loads(code)
        for field in ["name", "version", "scripts", "dependencies"]:
            if field not in pkg:
                errors.append({
                    "file": file_path,
                    "type": "missing_field",
                    "message": f"package.json missing required field: '{field}'",
                    "line": None,
                })
        if "dependencies" in pkg:
            for req in ["react", "react-dom", "react-router-dom", "axios"]:
                if req not in pkg["dependencies"]:
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
    for var in ["DATABASE_URL", "SECRET_KEY", "JWT_SECRET_KEY"]:
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
            "message": "public/index.html missing <div id=\"root\"></div>",
            "line": None,
        })

    # Check Tailwind CDN — must be <script> not <link>
    if "cdn.tailwindcss.com" in code:
        if re.search(r'<link[^>]*tailwindcss[^>]*>', code):
            errors.append({
                "file": file_path,
                "type": "wrong_tailwind_tag",
                "message": "Tailwind CDN must use <script src> not <link href>.",
                "line": None,
            })
    else:
        errors.append({
            "file": file_path,
            "type": "missing_tailwind",
            "message": "public/index.html missing Tailwind CDN script tag.",
            "line": None,
        })

    return errors

def check_cross_file_consistency(files):
    """Check imported components exist, and backend/__init__.py is present."""
    errors = []
    file_paths = set(files.keys())

    # Check backend/__init__.py exists when app.py imports from backend
    app_py = files.get("backend/app.py", "")
    if app_py and ("from backend import" in app_py or "from backend." in app_py):
        if "backend/__init__.py" not in file_paths:
            errors.append({
                "file": "backend/__init__.py",
                "type": "missing_init_file",
                "message": "backend/app.py uses 'from backend import' but backend/__init__.py is missing.",
                "line": None,
            })

    # Build set of component names that exist
    component_names = set()
    for path in file_paths:
        if path.startswith("frontend/src/") and path.endswith(".js"):
            component_names.add(os.path.basename(path).replace(".js", ""))

    # Check each JS file's imports
    for path, code in files.items():
        if not (path.endswith(".js") or path.endswith(".jsx")):
            continue
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            match = re.match(r"^import\s+\{?\s*(\w+)\s*\}?\s+from\s+['\"](.+)['\"]", line)
            if match:
                imported_name = match.group(1)
                import_path = match.group(2)
                if import_path.startswith("./") or import_path.startswith("../"):
                    if import_path in ("../api", "./api"):
                        continue
                    if imported_name not in component_names and imported_name not in (
                        "React", "useState", "useEffect", "useNavigate", "useParams",
                        "Link", "Routes", "Route", "BrowserRouter", "Navigate", "Outlet"
                    ):
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
    Returns: {"passed": bool, "errors": [...], "summary": "..."}
    """
    all_errors = []

    for file_path, code in files.items():
        if not code or not code.strip():
            all_errors.append({
                "file": file_path,
                "type": "empty_file",
                "message": "File is empty.",
                "line": None,
            })
            continue

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

    all_errors.extend(check_cross_file_consistency(files))

    # Deduplicate
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