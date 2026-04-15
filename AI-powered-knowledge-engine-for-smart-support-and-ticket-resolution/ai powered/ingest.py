import os
import sys
import logging
from tqdm import tqdm

# Ensure we can import from local app dir
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'app'))

import rag_engine
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import shutil

# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

def manual_ingest():
    print(f"Checking {rag_engine.DATA_RAW_DIR} for documents...")
    
    # Ensure dirs exist
    if not os.path.exists(rag_engine.DATA_RAW_DIR):
        try:
            os.makedirs(rag_engine.DATA_RAW_DIR)
        except: 
            pass
            
    if not os.path.exists(rag_engine.DATA_PROCESSED_DIR):
        os.makedirs(rag_engine.DATA_PROCESSED_DIR, exist_ok=True)

    files = [f for f in os.listdir(rag_engine.DATA_RAW_DIR) if os.path.isfile(os.path.join(rag_engine.DATA_RAW_DIR, f))]
    
    if not files:
        print("No new documents found to ingest.")
        return

    print(f"Found {len(files)} documents. Loading...")
    
    documents = []
    
    # Load with progress bar
    for f in tqdm(files, desc="Loading Files"):
        file_path = os.path.join(rag_engine.DATA_RAW_DIR, f)
        try:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif f.lower().endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not documents:
        print("No valid content extracted.")
        return

    print(f"Splitting {len(documents)} pages/documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Total Chunks: {len(docs)}")
    
    # Embed and Store with progress bar
    print("Generating Embeddings (This uses CPU and may take time)...")
    embeddings = rag_engine.OllamaEmbeddings(model="tinyllama")
    
    # Batch processing for progress bar
    batch_size = 5
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    if os.path.exists(rag_engine.FAISS_INDEX_PATH):
        try:
            db = FAISS.load_local(rag_engine.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing index.")
        except:
             # Create new if fail
             db = None
    else:
        db = None
    
    # We can't easily hook into FAISS.add_documents for a progress bar unless we batch manually
    # So we will add batch by batch
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding Chunks"):
        batch = docs[i : i + batch_size]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
            
    print("Saving Vector Index...")
    db.save_local(rag_engine.FAISS_INDEX_PATH)
    
    # Move files
    print("Cleaning up...")
    for f in files:
        src = os.path.join(rag_engine.DATA_RAW_DIR, f)
        dst = os.path.join(rag_engine.DATA_PROCESSED_DIR, f)
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"Failed to move {f}: {e}")

    print("\n✅ Ingestion Complete! You can now run the main app.")

if __name__ == "__main__":
    manual_ingest()