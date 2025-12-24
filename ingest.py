import os
import sys
import base64
from dotenv import load_dotenv
from typing import List
from groq import Groq

# Load environment variables
load_dotenv()

# Check for Groq Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY is missing in .env. Needed for Audio/Image processing.")
    sys.exit(1)

# FORCE DISABLE OPENAI to prevent accidental usage and quota errors
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Check for Llama Cloud Key
if not os.getenv("LLAMA_CLOUD_API_KEY"):
    print("❌ ERROR: LLAMA_CLOUD_API_KEY is missing in .env. Needed for PDF parsing.")
    sys.exit(1)

# --- Imports ---
from llama_parse import LlamaParse
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def process_audio(file_path: str) -> Document:
    print(f"🎤 Transcribing Audio: {os.path.basename(file_path)}...")
    try:
        with open(file_path, "rb") as audio_file:
            # Groq Whisper
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="json", 
                language="en", 
                temperature=0.0 
            )
            text = transcription.text
            return Document(text=text, metadata={"file_name": os.path.basename(file_path), "file_type": "audio"})
    except Exception as e:
        print(f"❌ Error processing audio {file_path}: {e}")
        return None

class CustomAudioReader:
    def load_data(self, file, extra_info=None):
        doc = process_audio(str(file))
        return [doc] if doc else []



def ingest_documents():
    print("--- STARTING MULTIMODAL INGESTION (GROQ POWERED) ---")
    
    # 1. Configure Settings to use LOCAL Embeddings
    print("⬇️  Loading local embedding model (BAAI/bge-small-en-v1.5)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = None 
    
    # 2. Setup LlamaParse for PDFs (Multimodal Tables/Text)
    # 2. Setup PyMuPDF for PDFs (Faster, Local)
    print("📄 Setting up PDF Parser (PyMuPDF)...")
    # parser = LlamaParse(result_type="markdown", verbose=True, language="en") # Disabled
    pdf_reader = PyMuPDFReader()
    # Still keep LlamaParse for Images if needed, or remove if user wants pure PyMuPDF
    # Since user said "use pymupdf to embed the image", likely means PyMuPDF for PDFs.
    # For images (.jpg), PyMuPDFReader doesn't support them directly.
    # LlamaParse is still best for images unless we switch to something else.
    image_parser = LlamaParse(result_type="markdown", verbose=True, language="en") 


    # 2. Setup Custom Readers
    # 2. Setup Custom Readers
    audio_reader = CustomAudioReader()
    # image_reader removed, using LlamaParse

    # 3. Load Data with File Extractors
    file_extractor = {
        ".pdf": pdf_reader,
        ".mp3": audio_reader,
        ".wav": audio_reader,
        ".m4a": audio_reader,
        ".jpg": image_parser,
        ".jpeg": image_parser,
        ".png": image_parser,
        ".ppt": image_parser,
        ".pptx": image_parser,
    }
    
    print("📂 Scanning ./data for PDFs, Word Docs, Images, and Audio...")
    reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
    documents = reader.load_data()
    print(f"✅ Loaded {len(documents)} document fragments.")

    # 4. Setup Vector Database (ChromaDB)
    db = chromadb.PersistentClient(path="./db")
    chroma_collection = db.get_or_create_collection("project_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Create Embeddings & Save
    if documents:
        print("🧠 Indexing documents into Vector Store...")
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print("--- INGESTION COMPLETE: Data saved to ./db ---")
    else:
        print("⚠️ No documents found to ingest!")

if __name__ == "__main__":
    ingest_documents()