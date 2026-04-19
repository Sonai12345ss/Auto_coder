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
    errors = []
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        if re.match(r"^from agent\.", line) and "tools" not in line and "memory" not in line and "builder" not in line and "planner" not in line:
            errors.append({
                "file": file_path,
                "type": "bad_import",
                "message": f"Suspicious local import: {line.strip()}",
                "line": i,
            })
        if re.match(r"^\s{4,}db\.Index\(", line):
            errors.append({
                "file": file_path,
                "type": "misplaced_db_index",
                "message": f"db.Index() must be outside class body, found indented at line {i}: {line.strip()}",
                "line": i,
            })

    # routes.py: create_access_token used but not imported
    if "routes.py" in file_path:
        if "create_access_token" in code and "from flask_jwt_extended import" in code:
            jwt_line = next((l for l in lines if "from flask_jwt_extended import" in l), "")
            if "create_access_token" not in jwt_line:
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

    # routes.py: phantom backend.api import (backend/api.py doesn't exist)
    if "routes.py" in file_path:
        if re.search(r"from backend\.api import", code):
            errors.append({
                "file": file_path,
                "type": "phantom_backend_api",
                "message": "routes.py imports from backend.api which doesn't exist. Call db/models directly — there is no backend/api.py.",
                "line": None,
            })

    # ─────────────────────────────────────────────────────────
    # NEW CHECK: Plain-text password storage in routes.py
    # Catches User(password=data['password']) without hashing.
    # Silent auth failure — login will never work.
    # ─────────────────────────────────────────────────────────
    if "routes.py" in file_path:
        # Detect if User (or any model) is constructed with a raw password field
        if re.search(r"User\s*\(.*password\s*=\s*data\[", code, re.DOTALL):
            # Make sure set_password isn't being called somewhere nearby
            if "set_password" not in code and "generate_password_hash" not in code:
                errors.append({
                    "file": file_path,
                    "type": "plaintext_password",
                    "message": (
                        "routes.py stores plain-text password in User constructor. "
                        "Use user.set_password(data['password']) AFTER creating the user object, "
                        "or User(password=generate_password_hash(data['password'])). "
                        "Also update login to use user.check_password(data['password'])."
                    ),
                    "line": None,
                })

    # routes.py: login compares plain-text passwords (==) instead of check_password
    if "routes.py" in file_path:
        if re.search(r"user\.password\s*==\s*data\[", code):
            errors.append({
                "file": file_path,
                "type": "plaintext_password_check",
                "message": (
                    "routes.py compares passwords with == (plain-text comparison). "
                    "Use user.check_password(data['password']) instead. "
                    "Plain-text comparison will fail once passwords are hashed."
                ),
                "line": None,
            })

    # app.py: duplicate db definition conflicts with __init__.py
    if "app.py" in file_path and "db = SQLAlchemy()" in code:
        errors.append({
            "file": file_path,
            "type": "duplicate_db_instance",
            "message": "app.py defines its own db = SQLAlchemy() — conflicts with backend/__init__.py. Remove it and use: from backend import db, jwt",
            "line": None,
        })

    # models.py: duplicate db definition conflicts with __init__.py
    if "models.py" in file_path and "db = SQLAlchemy()" in code:
        errors.append({
            "file": file_path,
            "type": "duplicate_db_instance",
            "message": "models.py defines its own db = SQLAlchemy() — conflicts with backend/__init__.py. Use: from backend import db",
            "line": None,
        })

    # models.py: duplicate db.Index names cause SQLAlchemy crash
    if "models.py" in file_path:
        index_names = re.findall(r"db\.Index\('(\w+)'", code)
        seen_names = set()
        for name in index_names:
            if name in seen_names:
                errors.append({
                    "file": file_path,
                    "type": "duplicate_index_name",
                    "message": f"db.Index name '{name}' is used more than once — SQLAlchemy will crash. Use unique names like ix_post_user_id, ix_comment_post_id.",
                    "line": None,
                })
            seen_names.add(name)

    return errors

def test_js_syntax(file_path, code):
    errors = []
    lines = code.split("\n")

    # CSS imports (whitelist index.js importing index.css)
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

    # React 17 API
    for i, line in enumerate(lines, 1):
        if "ReactDOM.render(" in line:
            errors.append({
                "file": file_path,
                "type": "react17_api",
                "message": f"ReactDOM.render() is deprecated. Use ReactDOM.createRoot() instead. Line {i}: {line.strip()}",
                "line": i,
            })

    # Double BrowserRouter in index.js
    if "BrowserRouter" in code and file_path.endswith("index.js"):
        errors.append({
            "file": file_path,
            "type": "double_router",
            "message": "index.js wraps App in BrowserRouter — App.js already has one. Remove it from index.js.",
            "line": None,
        })

    # useEffect missing dependency array
    effect_matches = list(re.finditer(r"useEffect\s*\(\s*(?:async\s*)?\(\s*\)\s*=>", code))
    for match in effect_matches:
        snippet = code[match.end():match.end()+300]
        if re.search(r"\}\s*\)\s*;?\s*\n", snippet):
            if not re.search(r"\}\s*,\s*\[", snippet):
                line_num = code[:match.start()].count("\n") + 1
                errors.append({
                    "file": file_path,
                    "type": "missing_deps_array",
                    "message": f"useEffect at line {line_num} is missing a dependency array [] — causes infinite re-renders.",
                    "line": line_num,
                })

    # Unsafe error.response access
    for i, line in enumerate(lines, 1):
        if ("error.response.status" in line or "error.response.data" in line) and "error.response?." not in line:
            errors.append({
                "file": file_path,
                "type": "unsafe_error_access",
                "message": f"Unsafe error.response access (crashes if network fails). Use error.response?.status instead. Line {i}: {line.strip()}",
                "line": i,
            })

    # App.js critical validations
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

        # ─────────────────────────────────────────────────────────
        # NEW CHECK: App.js must not render UI outside <BrowserRouter>
        # Navbar using <Link> outside <BrowserRouter> will crash.
        # ─────────────────────────────────────────────────────────
        if "Navbar" in code and "BrowserRouter" in code:
            # Rough structural check: Navbar should appear INSIDE BrowserRouter block
            browser_router_start = code.find("<BrowserRouter")
            navbar_in_return = code.find("<Navbar")
            if navbar_in_return != -1 and browser_router_start != -1:
                if navbar_in_return < browser_router_start:
                    errors.append({
                        "file": file_path,
                        "type": "navbar_outside_router",
                        "message": (
                            "App.js renders <Navbar /> BEFORE <BrowserRouter>. "
                            "Navbar uses <Link> which requires a Router context — "
                            "move <Navbar /> inside <BrowserRouter>."
                        ),
                        "line": None,
                    })

        # ─────────────────────────────────────────────────────────
        # NEW CHECK: App.js must not import api or make API calls
        # App.js is a router only. Data fetching belongs in pages.
        # ─────────────────────────────────────────────────────────
        if "from '../api'" in code or "from './api'" in code or "from '../../api'" in code:
            errors.append({
                "file": file_path,
                "type": "app_js_api_import",
                "message": (
                    "App.js imports from api.js and makes API calls — this is wrong. "
                    "App.js should be a pure router with no data fetching. "
                    "Remove the api import, useEffect, useState, and all fetch logic from App.js. "
                    "Data fetching belongs in individual page components."
                ),
                "line": None,
            })

        # ─────────────────────────────────────────────────────────
        # NEW CHECK: PrivateRoute must use Outlet pattern (v6)
        # Not the children prop wrapper pattern
        # ─────────────────────────────────────────────────────────
        if "PrivateRoute" in code:
            # Detect old children-prop pattern: <PrivateRoute><Component /></PrivateRoute>
            if re.search(r"<PrivateRoute\s*>\s*<\w", code):
                errors.append({
                    "file": file_path,
                    "type": "private_route_wrong_pattern",
                    "message": (
                        "App.js uses <PrivateRoute><Component /></PrivateRoute> — "
                        "this is the React Router v5 children pattern. "
                        "For React Router v6 with Outlet, use: "
                        "<Route element={<PrivateRoute />}><Route path='/x' element={<X />} /></Route>"
                    ),
                    "line": None,
                })

    # index.js React 18
    if file_path.endswith("index.js") and "components/" not in file_path:
        if "createRoot" not in code:
            errors.append({
                "file": file_path,
                "type": "critical_react18_missing",
                "message": "index.js missing createRoot — must use React 18 API.",
                "line": None,
            })

    # Raw fetch() bypassing api.js JWT interceptor
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

    # ─────────────────────────────────────────────────────────
    # NEW CHECK: api.js token key mismatch
    # Backend returns {"token": ...} but frontend may read
    # response.data.access_token — a silent auth failure.
    # ─────────────────────────────────────────────────────────
    if file_path == "frontend/src/api.js":
        if "response.data.access_token" in code:
            errors.append({
                "file": file_path,
                "type": "token_key_mismatch",
                "message": (
                    "api.js reads response.data.access_token but the backend /api/login "
                    "returns {\"token\": ...} not {\"access_token\": ...}. "
                    "Change to response.data.token. This causes silent auth failure — "
                    "login appears to work but the token is never saved."
                ),
                "line": None,
            })

    # TRUNCATED COMPONENT DETECTION (structural + semantic)
    if "components/" in file_path and file_path.endswith(".js"):
        truncation_issues = []
        component_name = os.path.basename(file_path).replace(".js", "")

        if "export default" not in code:
            truncation_issues.append("missing 'export default'")
        if "return (" not in code and "return(" not in code:
            truncation_issues.append("missing 'return (' — JSX block absent")

        has_imports = "import " in code
        has_hooks = "useState" in code or "useEffect" in code
        has_jsx = "<div" in code or "<span" in code or "<button" in code or "<form" in code or "<p " in code
        if has_imports and has_hooks and not has_jsx:
            truncation_issues.append("has hooks but no JSX elements — model stopped before return()")

        import_names = re.findall(r"^import\s+(?:React,\s*)?\{\s*([^}]+)\}", code, re.MULTILINE)
        all_imported = []
        for grp in import_names:
            all_imported.extend([n.strip() for n in grp.split(",")])
        if "useState" in all_imported:
            usage_count = code.count("useState(")
            if usage_count == 0:
                truncation_issues.append("useState imported but never called — component body is missing")

        if truncation_issues:
            errors.append({
                "file": file_path,
                "type": "truncated_component",
                "message": (
                    f"Component appears truncated: {', '.join(truncation_issues)}. "
                    f"Must include all steps: imports → const {component_name} = () => {{ → "
                    f"useState → useEffect with [] → handlers → return( → JSX → }} → export default {component_name};"
                ),
                "line": None,
            })

    # TRUNCATED api.js DETECTION
    if file_path == "frontend/src/api.js":
        api_issues = []
        if "interceptors" not in code:
            api_issues.append("missing axios interceptors (request + response)")
        export_count = len(re.findall(r"^export\s+const\s+\w+", code, re.MULTILINE))
        if export_count < 2:
            api_issues.append(f"only {export_count} exported function(s) — likely truncated before endpoint functions")
        if api_issues:
            errors.append({
                "file": file_path,
                "type": "truncated_api",
                "message": (
                    f"api.js appears truncated: {', '.join(api_issues)}. "
                    f"Must include: axios instance + request interceptor + response interceptor + all endpoint functions."
                ),
                "line": None,
            })

    return errors

def test_package_json(file_path, code):
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
    elif re.search(r'<link[^>]*cdn\.tailwindcss\.com[^>]*>', code):
        errors.append({
            "file": file_path,
            "type": "wrong_tailwind_tag",
            "message": "Tailwind CDN must use <script src=\"https://cdn.tailwindcss.com\"></script> NOT a <link> tag.",
            "line": None,
        })
    return errors

def check_cross_file_consistency(files):
    errors = []
    file_paths = set(files.keys())

    # backend/__init__.py must exist if any file does `from backend import ...`
    uses_backend_import = any(
        "from backend import" in code or "from backend." in code
        for code in files.values() if code
    )
    if uses_backend_import and "backend/__init__.py" not in file_paths:
        errors.append({
            "file": "backend/__init__.py",
            "type": "missing_init",
            "message": "backend/__init__.py is missing. Files use 'from backend import db, jwt' but no __init__.py exists.",
            "line": None,
        })

    # Component names that exist
    component_names = set()
    for path in file_paths:
        if path.startswith("frontend/src/") and path.endswith(".js"):
            component_names.add(os.path.basename(path).replace(".js", ""))

    # Named exports from api.js — imports of these are NOT missing components
    api_exports = set()
    api_code = files.get("frontend/src/api.js", "")
    if api_code:
        api_exports = set(re.findall(r"^export\s+(?:const|async function|function)\s+(\w+)", api_code, re.MULTILINE))

    # App.js route count vs component count
    app_js_code = files.get("frontend/src/App.js", "")
    if app_js_code:
        route_count = len(re.findall(r"<Route\s", app_js_code))
        EXCLUDED_COMPONENTS = {"PrivateRoute", "Navbar", "Footer", "Layout", "App"}
        page_components = sum(
            1 for p in file_paths
            if "components/" in p and p.endswith(".js")
            and not any(x in p for x in EXCLUDED_COMPONENTS)
        )
        if page_components > 2 and route_count < max(2, page_components - 3):
            errors.append({
                "file": "frontend/src/App.js",
                "type": "truncated_component",
                "message": (
                    f"App.js only has {route_count} <Route> elements but "
                    f"{page_components} page components exist. App.js was likely "
                    f"truncated — it must include routes for ALL components including "
                    f"ProductDetail, Cart, OrderForm, etc."
                ),
                "line": None,
            })

    # ─────────────────────────────────────────────────────────
    # NEW CHECK: Cross-file password consistency
    # If models.py has set_password/check_password, routes.py
    # must use them — not raw string comparison or assignment.
    # ─────────────────────────────────────────────────────────
    models_code = files.get("backend/models.py", "")
    routes_code = files.get("backend/routes.py", "")
    if models_code and routes_code:
        has_pw_methods = "set_password" in models_code and "check_password" in models_code
        if has_pw_methods:
            if "set_password" not in routes_code and re.search(r"User\s*\(.*password\s*=", routes_code, re.DOTALL):
                errors.append({
                    "file": "backend/routes.py",
                    "type": "missing_set_password_call",
                    "message": (
                        "models.py defines set_password() and check_password() but routes.py "
                        "never calls set_password() during registration. "
                        "After creating the User, call: new_user.set_password(data['password']). "
                        "Also ensure login uses: user.check_password(data['password'])."
                    ),
                    "line": None,
                })

    # ─────────────────────────────────────────────────────────
    # NEW CHECK: api.js exports vs routes.py endpoints
    # Every major route group should have a corresponding
    # api.js export. Catches the case where api.js is built
    # before routes.py and misses endpoints.
    # ─────────────────────────────────────────────────────────
    if api_code and routes_code:
        route_paths = re.findall(r"@routes\.route\(['\"]([^'\"]+)['\"]", routes_code)
        api_export_names = list(api_exports)

        # Build a set of resource names from route paths (e.g. 'products', 'orders', 'users')
        route_resources = set()
        for rp in route_paths:
            parts = [p for p in rp.strip("/").split("/") if p and not p.startswith("<")]
            if len(parts) >= 2:  # /api/resource
                route_resources.add(parts[1])  # e.g. 'products'

        # For each resource, check that at least one api export references it
        for resource in route_resources:
            has_export = any(resource.lower() in name.lower() for name in api_export_names)
            if not has_export:
                errors.append({
                    "file": "frontend/src/api.js",
                    "type": "missing_api_export",
                    "message": (
                        f"routes.py has endpoints for '{resource}' but api.js has no exported "
                        f"function referencing '{resource}'. Add get{resource.capitalize()}(), "
                        f"create{resource.capitalize()}(), etc. to api.js."
                    ),
                    "line": None,
                })

    # Cross-file import check
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
                    if re.search(r"[./]api[./]?", import_path) or import_path.endswith("/api"):
                        continue
                    if imported_name in api_exports:
                        continue
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

    # Deduplicate by (file, line, type)
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
    if test_result["passed"]:
        print("\n✅ TESTER: All checks passed!")
        return
    print(f"\n❌ TESTER: {test_result['summary']}")
    for e in test_result["errors"]:
        loc = f" (line {e['line']})" if e.get("line") else ""
        print(f"  [{e['type']}] {e['file']}{loc}")
        print(f"    → {e['message']}")