import chromadb
from chromadb.utils import embedding_functions
import os

os.makedirs("./chroma_db", exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="agent_memories",
    embedding_function=default_ef
)

def add_experience(task, solution, error=None, status="success"):
    """Stores successful code and task in ChromaDB."""
    document = f"Task: {task}\nSolution: {solution}"
    if error:
        document += f"\nError that was fixed: {error}"
    collection.add(
        documents=[document],
        metadatas=[{"type": "code_snippet", "status": status}],
        ids=[str(abs(hash(task)))]
    )
    print("💾 Experience saved to memory.")

def query_experience(current_task, n_results=2):
    """Finds similar past tasks from memory."""
    try:
        results = collection.query(
            query_texts=[current_task],
            n_results=n_results
        )
        return results['documents']
    except Exception:
        return [[]]

def add_project_blueprint(description, blueprint_json):
    """Stores a successful project blueprint in memory."""
    import json
    document = f"Project: {description}\nBlueprint: {json.dumps(blueprint_json)}"
    collection.add(
        documents=[document],
        metadatas=[{"type": "blueprint"}],
        ids=[str(abs(hash(description)))]
    )
    print("💾 Blueprint saved to memory.")

def query_similar_blueprint(description):
    """Finds similar past project blueprints."""
    try:
        results = collection.query(
            query_texts=[description],
            n_results=1,
            where={"type": "blueprint"}
        )
        return results['documents']
    except Exception:
        return [[]]