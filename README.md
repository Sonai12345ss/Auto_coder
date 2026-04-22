# 🚀 Autocoder AI

> **Turn a single sentence into a full-stack production-ready application.**

Autocoder AI is an intelligent AI-powered system that generates complete full-stack applications (Frontend + Backend + Database + Deployment) from a simple natural language prompt.

---

## ⚡ What Makes Autocoder AI Different?

Unlike basic code generators, **Autocoder AI is a self-building system**:

* 🧠 Generates complete project architecture
* 🔁 Self-heals using an automated Tester + Debugger loop
* ⚙️ Uses multiple LLM providers with fallback logic
* 📦 Produces Docker-ready deployable applications
* ☁️ Stores builds permanently using cloud storage

---

## 🧠 Core Features

### 🔥 1. Prompt → Full Stack App

Describe your idea like:

```bash
"Build a task manager with login and dashboard"
```

Autocoder AI generates:

* React Frontend (with routing + UI)
* Flask Backend (API + business logic)
* PostgreSQL Database (schema + relationships)
* Authentication system

---

### 🔁 2. Self-Healing System (Game-Changer)

Autocoder AI doesn’t just generate code — it **fixes itself**.

* Runs automated tests
* Detects errors
* Sends issues back to LLM
* Rewrites broken code
* Repeats until build succeeds

> This makes it closer to an **autonomous developer**, not just a generator.

---

### ⚙️ 3. Multi-LLM Provider System

To ensure reliability and cost-efficiency:

* Primary models (fast + free)
* Fallback models (backup)
* Paid models (last resort)

Supported providers include:

* Groq (LLaMA variants)
* Gemini (Flash / Pro)
* OpenRouter models

---

### 📦 4. Dockerized Output

Every generated project includes:

* Dockerfile
* Backend setup
* Dependency management

👉 Ready for deployment on:

* Render
* Railway
* AWS
* Any container-based platform

---

### ☁️ 5. Persistent Build Storage

* Builds are stored using **Supabase Storage**
* Users can download ZIP anytime
* No data loss after server restart

---

## 🏗️ System Architecture

```
User Prompt
     ↓
Planner (LLM)
     ↓
Code Generator
     ↓
Tester Engine
     ↓
Debugger Engine
     ↓
Build Validator
     ↓
ZIP Packaging
     ↓
Cloud Storage (Supabase)
```

---

## 🔄 How It Works

1. User sends a prompt
2. System breaks it into tasks
3. Code is generated module-by-module
4. Tester runs validations
5. Debugger fixes errors automatically
6. Final app is packaged and stored

---

## 📡 API Endpoints

### 🔹 Create Build

```http
POST /build
```

**Response:**

```json
{
  "build_id": "abc123"
}
```

---

### 🔹 Track Build Status

```http
GET /status/<build_id>
```

Returns:

* Progress
* Logs
* Errors (if any)
* Completion status

---

## ⚙️ Tech Stack

### 🧠 AI & Backend

* Python
* FastAPI
* LLM APIs (Groq, Gemini, OpenRouter)

### 🌐 Frontend (Generated)

* React
* Tailwind CSS
* React Router v6

### 🗄 Database

* PostgreSQL
* SQLAlchemy ORM

### ☁️ Storage

* Supabase Storage

### 🐳 DevOps

* Docker

---

## 🛠️ Local Setup

```bash
git clone https://github.com/your-username/autocoder-ai.git
cd autocoder-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```

---

## 🎯 Why This Project Stands Out

* 🚀 Not just code generation — **autonomous correction system**
* 🧠 Real-world AI system design (LLM orchestration)
* ⚙️ Production-ready architecture
* 💡 Solves real developer pain: time + debugging

This is closer to building a **mini AI software engineer**.

---

## 🧭 Roadmap

* [ ] Build history dashboard
* [ ] UI improvements
* [ ] Template system
* [ ] Iterative editing (edit generated apps)
* [ ] Authentication for users
* [ ] One-click deployment

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo
* Create a feature branch
* Submit a PR

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Sonai Mondal**

* Aspiring Machine Learning Engineer
* Building real-world AI systems

---

## ⭐ Final Thought

> "Autocoder AI is not just a tool — it's a step toward autonomous software creation."

---

⭐ If you like this project, consider starring the repo!
