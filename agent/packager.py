import os

# ─────────────────────────────────────────────
#  PACKAGER AGENT
#  Generates Docker + deploy config for every project.
#  Runs after Builder + Tester/Debugger.
#  Adds: Dockerfile, frontend/Dockerfile,
#        docker-compose.yml, .dockerignore
# ─────────────────────────────────────────────

def generate_backend_dockerfile():
    return """\
# ── Backend Dockerfile ──────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "-m", "backend.app"]
"""

def generate_frontend_dockerfile():
    return """\
# ── Frontend Dockerfile ─────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install --frozen-lockfile 2>/dev/null || npm install

# Copy source and build
COPY . .
RUN npm run build

# ── Production image ────────────────────────
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx config for React Router (handle client-side routes)
RUN echo 'server { \\n\\
    listen 80; \\n\\
    location / { \\n\\
        root /usr/share/nginx/html; \\n\\
        index index.html; \\n\\
        try_files $uri $uri/ /index.html; \\n\\
    } \\n\\
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
"""

def generate_docker_compose(project_name, blueprint):
    """Generate docker-compose.yml based on the project blueprint."""
    db_name = project_name.lower().replace("-", "_")

    # Collect all API endpoints to show in comment
    endpoints = blueprint.get("api_endpoints", [])
    endpoint_count = len(endpoints)

    return f"""\
# ── docker-compose.yml ───────────────────────
# Run everything with: docker-compose up --build
# App will be available at:
#   Frontend: http://localhost:3000
#   Backend:  http://localhost:5000
#   API docs: http://localhost:5000/health
# {endpoint_count} API endpoints included

version: '3.9'

services:

  # ── PostgreSQL Database ──────────────────
  db:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: {db_name}
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${{POSTGRES_PASSWORD:-postgres}}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── Flask Backend ────────────────────────
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      DATABASE_URL: postgresql://postgres:${{POSTGRES_PASSWORD:-postgres}}@db:5432/{db_name}
      SECRET_KEY: ${{SECRET_KEY:-change-me-in-production}}
      JWT_SECRET_KEY: ${{JWT_SECRET_KEY:-change-me-in-production}}
      DEBUG: "false"
      FLASK_ENV: production
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ── React Frontend ───────────────────────
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "3000:80"
    environment:
      REACT_APP_API_URL: http://localhost:5000
    depends_on:
      - backend

volumes:
  postgres_data:
"""

def generate_dockerignore():
    return """\
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.eggs/
.pytest_cache/
.mypy_cache/

# Virtual environments
venv/
env/
.venv/
agent_env/

# Environment files (never include secrets in images)
.env
.env.local
.env.production

# Database files
*.db
*.sqlite
*.sqlite3

# Node
node_modules/
frontend/node_modules/
npm-debug.log*

# Build outputs
frontend/build/
frontend/.cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Test files
.coverage
htmlcov/

# Git
.git/
.gitignore
"""

def generate_makefile(project_name):
    """Generate a Makefile with common dev commands."""
    return f"""\
# ── Makefile ─────────────────────────────────
# Quick commands for {project_name}
# Usage: make <command>

.PHONY: up down build logs shell-backend shell-db clean

## Start all services
up:
\tdocker-compose up --build

## Start in background
up-detached:
\tdocker-compose up --build -d

## Stop all services
down:
\tdocker-compose down

## Stop and remove volumes (WARNING: deletes database)
down-clean:
\tdocker-compose down -v

## View logs
logs:
\tdocker-compose logs -f

## View backend logs only
logs-backend:
\tdocker-compose logs -f backend

## Open shell in backend container
shell-backend:
\tdocker-compose exec backend /bin/bash

## Open PostgreSQL shell
shell-db:
\tdocker-compose exec db psql -U postgres -d {project_name.lower()}

## Run backend tests
test:
\tdocker-compose exec backend python -m pytest

## Clean up Docker resources
clean:
\tdocker-compose down -v --rmi all --remove-orphans
"""

def generate_deploy_readme(project_name, blueprint):
    """Generate DEPLOY.md with deployment instructions."""
    stack = blueprint.get("stack", {})
    return f"""\
# Deploying {project_name.replace('_', ' ').title()}

## 🐳 Docker (Recommended)

### Prerequisites
- Docker Desktop installed
- Docker Compose v2+

### Quick Start
```bash
# 1. Clone / extract the project
cd {project_name}

# 2. Set environment variables
cp .env.example .env
# Edit .env and set SECRET_KEY and JWT_SECRET_KEY

# 3. Start everything
docker-compose up --build

# App is now running at:
# Frontend: http://localhost:3000
# Backend:  http://localhost:5000
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | ⚠️ Change this! |
| `JWT_SECRET_KEY` | JWT signing key | ⚠️ Change this! |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `DATABASE_URL` | Auto-set by compose | — |

---

## ☁️ Deploy to Render (Free)

### Backend
1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python -m backend.app`
5. Add environment variables from `.env.example`
6. Add a **PostgreSQL** database and copy the connection string to `DATABASE_URL`

### Frontend
1. Create a new **Static Site** on Render
2. Set build command: `cd frontend && npm install && npm run build`
3. Set publish directory: `frontend/build`
4. Add environment variable: `REACT_APP_API_URL=https://your-backend.onrender.com`

---

## ☁️ Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

---

## Stack
- Frontend: {stack.get('frontend', 'React')}
- Backend: {stack.get('backend', 'Flask')}
- Database: {stack.get('database', 'PostgreSQL')}
"""

def run_packager(project_path, blueprint, files):
    """
    Main entry point.
    Takes project_path, blueprint, and existing files dict.
    Returns updated files dict with Docker files added.
    """
    project_name = blueprint.get("project_name", "app")
    print(f"\n📦 PACKAGER: Generating Docker + deploy config...")

    new_files = {}

    # Backend Dockerfile
    new_files["Dockerfile"] = generate_backend_dockerfile()
    print("  ✅ Dockerfile (backend)")

    # Frontend Dockerfile
    new_files["frontend/Dockerfile"] = generate_frontend_dockerfile()
    print("  ✅ frontend/Dockerfile")

    # docker-compose.yml
    new_files["docker-compose.yml"] = generate_docker_compose(project_name, blueprint)
    print("  ✅ docker-compose.yml")

    # .dockerignore
    new_files[".dockerignore"] = generate_dockerignore()
    print("  ✅ .dockerignore")

    # Makefile
    new_files["Makefile"] = generate_makefile(project_name)
    print("  ✅ Makefile")

    # DEPLOY.md
    new_files["DEPLOY.md"] = generate_deploy_readme(project_name, blueprint)
    print("  ✅ DEPLOY.md")

    # Write all files to disk
    for file_path, content in new_files.items():
        full_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    print(f"  🐳 Docker config complete — run with: docker-compose up --build")

    # Return merged files
    return {**files, **new_files}