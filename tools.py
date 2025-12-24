import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def search_knowledge_base(query: str):
    """
    Tools that searches the project documentation for answers.
    """
    # --- SETUP LOCAL EMBEDDINGS (Must match ingest.py) ---
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = None # FORCE DISABLE OPENAI
    # -----------------------------------------------------

    # 1. Connect to existing DB
    db = chromadb.PersistentClient(path="./db")
    chroma_collection = db.get_collection("project_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Create Index from existing vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store
    )
    
    # 3. Create Query Engine
    # 4. Create Query Engine (no external LLM usage)
    try:
        # Walkthrough Update: Deep search (Top-15) for holistic coverage
        query_engine = index.as_query_engine(similarity_top_k=15)
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"(Error during retrieval): {e}"