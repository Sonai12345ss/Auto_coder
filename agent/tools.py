import subprocess
import os
import chromadb
from chromadb.utils import embedding_functions

# Ensure sandbox exists
os.makedirs("sandbox", exist_ok=True)

SANDBOX_DIR = "sandbox"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="agent_memories",
    embedding_function=default_ef
)

def read_file(path):
    """Reads a file and returns its content."""
    full_path = os.path.join(SANDBOX_DIR, path)
    try:
        with open(full_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: File '{path}' not found in sandbox."
    except Exception as e:
        return f"ERROR: {str(e)}"

def write_file(path, content):
    """Writes content to a file in sandbox."""
    full_path = os.path.join(SANDBOX_DIR, path)
    try:
        with open(full_path, "w") as f:
            f.write(content)
        return f"SUCCESS: Written to {path}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def execute_python_code(code, filename="temp_script.py"):
    """Saves code to a file in sandbox and runs it."""
    full_path = os.path.join(SANDBOX_DIR, filename)
    with open(full_path, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["python3", full_path],  # Fixed: python3 not python
            capture_output=True,
            text=True,
            timeout=10
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "ERROR: Code timed out after 10 seconds.", "exit_code": 1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": 1}

def add_experience(task, solution, error=None, status="success"):
    """Stores task + solution in ChromaDB."""
    document = f"Task: {task}\nSolution: {solution}"
    if error:
        document += f"\nError that was fixed: {error}"
    collection.add(
        documents=[document],
        metadatas=[{"type": "code_snippet", "status": status}],
        ids=[str(abs(hash(task)))]  # Fixed: abs() to avoid negative hash ids
    )

def query_experience(current_task):
    """Finds top 2 similar past tasks from memory."""
    try:
        results = collection.query(
            query_texts=[current_task],
            n_results=2
        )
        return results['documents']
    except Exception:
        return [[]]