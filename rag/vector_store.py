import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.join(BASE_DIR, "chroma_db"))

print("Using MODEL_NAME:", MODEL_NAME)
print("Using PERSIST_DIR:", PERSIST_DIR)


# ------------------------
# EMBEDDINGS (SINGLE SOURCE OF TRUTH)
# ------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ------------------------
# CREATE VECTOR DB
# ------------------------
def create_vectordb(documents):
    if not documents:
        raise ValueError("❌ No documents provided. Ingestion failed.")
    embedding_function = get_embeddings()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    print("✅ Vector DB created successfully")
    print("📥 Number of documents:", len(documents))
    print("📦 Saving to:", PERSIST_DIR)
    print("🔄 Creating embeddings...")
    print("📄 Sample doc:", documents[0].page_content[:100]) 
    return vectordb


# ------------------------
# LOAD VECTOR DB
# ------------------------
def load_vectordb():
    embedding_function = get_embeddings()

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function
    )

    print("📦 Vector DB loaded")
    return vectordb