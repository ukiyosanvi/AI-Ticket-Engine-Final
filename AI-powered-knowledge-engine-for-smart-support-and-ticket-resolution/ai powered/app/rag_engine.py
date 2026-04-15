import os
import shutil
import logging
import ollama
import math
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# Configure Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Path Handling
BASE_DIR = os.getcwd()
DATA_ROOT = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_ROOT, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
FAISS_INDEX_PATH = os.path.join(DATA_PROCESSED_DIR, "faiss_index")

class OllamaEmbeddings(Embeddings):
    """Custom Embeddings class to use Ollama natively."""
    def __init__(self, model="llama3.2:1b"):
        self.model = model

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            resp = ollama.embeddings(model=self.model, prompt=text)
            embeddings.append(resp['embedding'])
        return embeddings

    def embed_query(self, text):
        resp = ollama.embeddings(model=self.model, prompt=text)
        return resp['embedding']

def ingest_documents():
    """Ingests documents, splits them, and updates FAISS index."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    files = [f for f in os.listdir(DATA_RAW_DIR) if os.path.isfile(os.path.join(DATA_RAW_DIR, f))]
    
    if not files:
        logging.info("No new documents to ingest.")
        return

    logging.info(f"Found {len(files)} new documents. Starting ingestion...")
    
    documents = []
    for f in files:
        file_path = os.path.join(DATA_RAW_DIR, f)
        try:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif f.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            logging.error(f"Failed to load {f}: {e}")

    if not documents:
        return

    # Optimized splitting for higher accuracy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings()
    
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
            logging.info("Updated existing FAISS index.")
        except Exception as e:
            db = FAISS.from_documents(docs, embeddings)
    else:
        db = FAISS.from_documents(docs, embeddings)
        logging.info("Created new FAISS index.")

    db.save_local(FAISS_INDEX_PATH)
    
    # Move files to avoid double processing
    for f in files:
        try:
            shutil.move(os.path.join(DATA_RAW_DIR, f), os.path.join(DATA_PROCESSED_DIR, f))
        except Exception:
            pass

    logging.info("Ingestion complete.")

def get_relevant_context(query, k=3):
    """Retrieves context and calculates an improved confidence score."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return {"context_text": "", "kb_context_found": False, "retrieval_score": 0.0, "matches": []}

    embeddings = OllamaEmbeddings()
    try:
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        # We retrieve more (k) to get better context
        docs_with_scores = db.similarity_search_with_score(query, k=k)

        matches = []
        for doc, distance in docs_with_scores:
            # IMPROVED SCORE MATH: Using a sigmoid-style decay
            # This makes a distance of 0.5 roughly 85% and 1.0 roughly 60%
            similarity_score = 1 / (1 + math.exp(float(distance) - 1.5))
            
            # Boost the score slightly for the UI
            similarity_score = min(1.0, similarity_score * 1.2)

            matches.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": round(similarity_score, 3),
            })

        context_text = "\n\n---\n\n".join(match["content"] for match in matches)
        # Use the highest score as the main retrieval score
        main_score = matches[0]["similarity_score"] if matches else 0.0
        
        return {
            "context_text": context_text,
            "kb_context_found": bool(matches),
            "retrieval_score": main_score,
            "matches": matches,
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"context_text": "", "kb_context_found": False, "retrieval_score": 0.0, "matches": []}

if __name__ == "__main__":
    ingest_documents()